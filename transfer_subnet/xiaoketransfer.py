"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
sys.path.append('./segmentation')
import os
import tqdm
import argparse

import torch
from torchvision.utils import save_image
import torch.nn as nn
# from model import WaveEncoder, WaveDecoder

from utils.core import feature_wct
from utils.core import feature_adin
from utils.io import Timer, open_image, load_segment, compute_label_info
from xiaokemodel import XiaoKeEncoder, XiaoKeDecoder
import numpy as np
import torchvision.transforms as transforms
from segmentation.dataset import round2nearest_multiple
from segmentation.models import ModelBuilder, SegmentationModule
from scipy.io import loadmat
colors = loadmat('segmentation/data/color150.mat')['colors']
from PIL import Image
from scipy.misc import imread, imresize
import cv2
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import datetime

def overlay(img, pred_color, blend_factor=0.4):
    edges = cv2.Canny(pred_color, 20, 40)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=1)
    out = (1-blend_factor)*img + blend_factor * pred_color
    edge_pixels = (edges==255)
    new_color = [0,0,255]
    for i in range(0,3):
        timg = out[:,:,i]
        timg[edge_pixels]=new_color[i]
        out[:,:,i] = timg
    return out


def visualize_result(label_map):
    label_map = label_map.astype('int')
    label_map_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label in np.unique(label_map):
        label_map_rgb += (label_map == label)[:, :, np.newaxis] * \
            np.tile(colors[label],(label_map.shape[0], label_map.shape[1], 1))
    return label_map_rgb


class SegReMapping:
    def __init__(self, mapping_name, min_ratio=0.02):
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio
        print('label_mapping 是什么',self.label_mapping.shape)

    def cross_remapping(self, cont_seg, styl_seg):
        cont_label_info = []
        new_cont_label_info = []
        for label in np.unique(cont_seg):
            cont_label_info.append(label)
            new_cont_label_info.append(label)

        style_label_info = []
        new_style_label_info = []
        for label in np.unique(styl_seg):
            style_label_info.append(label)
            new_style_label_info.append(label)

        cont_set_diff = set(cont_label_info) - set(style_label_info)
        # Find the labels that are not covered by the style
        # Assign them to the best matched region in the style region
        # 尝试找到content中没有但是最接近的那个label
        for s in cont_set_diff:
            cont_label_index = cont_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in style_label_info:
                    new_cont_label_info[cont_label_index] = new_label
                    break
        new_cont_seg = cont_seg.copy()
        for i,current_label in enumerate(cont_label_info):
            new_cont_seg[(cont_seg == current_label)] = new_cont_label_info[i]

        cont_label_info = []
        for label in np.unique(new_cont_seg):
            cont_label_info.append(label)

        # 这边是style有，但是cont没有的
        styl_set_diff = set(style_label_info) - set(cont_label_info)
        valid_styl_set = set(style_label_info) - set(styl_set_diff)
        for s in styl_set_diff:
            style_label_index = style_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in valid_styl_set:
                    new_style_label_info[style_label_index] = new_label
                    break
        new_styl_seg = styl_seg.copy()
        for i,current_label in enumerate(style_label_info):
            # print("%d -> %d" %(current_label,new_style_label_info[i]))
            new_styl_seg[(styl_seg == current_label)] = new_style_label_info[i]

        return new_cont_seg, new_styl_seg

    def self_remapping(self, seg):
        #表示一张语义图本身的操作,将小的语义分割合并到大的部分，但是对于嘴唇部分，人脸的检测并不可取。

        init_ratio = self.min_ratio
        # Assign label with small portions to label with large portion
        new_seg = seg.copy()
        [h,w] = new_seg.shape
        n_pixels = h*w
        # First scan through what are the available labels and their sizes
        label_info = []
        ratio_info = []
        new_label_info = []
        for label in np.unique(seg):
            ratio = np.sum(np.float32((seg == label))[:])/n_pixels
            label_info.append(label)
            new_label_info.append(label)
            ratio_info.append(ratio)
        for i,current_label in enumerate(label_info):
            if ratio_info[i] < init_ratio:
                for j in range(self.label_mapping.shape[0]):
                    new_label = self.label_mapping[j,current_label]
                    if new_label in label_info:
                        index = label_info.index(new_label)
                        if index >= 0:
                            if ratio_info[index] >= init_ratio:
                                new_label_info[i] = new_label
                                break
        for i,current_label in enumerate(label_info):
            new_seg[(seg == current_label)] = new_label_info[i]
        return new_seg

class MySegReMapping:
    def __init__(self, mapping_name, min_ratio=0.02):
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio
        # print('label_mapping 是什么',self.label_mapping.shape) # shape 是 150x150

    # 这里针对多张style 图像来完成remapping
    def cross_remapping(self, cont_seg, styl_segs):
        cont_label_info = []
        new_cont_label_info = []
        for label in np.unique(cont_seg):
            cont_label_info.append(label)
            new_cont_label_info.append(label)

        style_label_info = []
        new_style_label_info = []
        total_dict = {}
        for styl_seg in styl_segs:
            for label in np.unique(styl_seg):
                style_label_info.append(label)
                new_style_label_info.append(label)

        cont_set_diff = set(cont_label_info) - set(style_label_info)
        # Find the labels that are not covered by the style
        # Assign them to the best matched region in the style region
        # 尝试找到content中没有但是最接近的那个label
        for s in cont_set_diff:
            cont_label_index = cont_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in style_label_info:
                    new_cont_label_info[cont_label_index] = new_label
                    break
        new_cont_seg = cont_seg.copy()
        # 第一个for循环更新content的语义分割图，第二个for更新content的label信息
        for i,current_label in enumerate(cont_label_info):
            new_cont_seg[(cont_seg == current_label)] = new_cont_label_info[i]
        cont_label_info = []
        for label in np.unique(new_cont_seg):
            cont_label_info.append(label)

        return new_cont_seg, styl_segs

    def self_remapping(self, seg):
        #表示一张语义图本身的操作,将小的语义分割合并到大的部分，但是对于嘴唇部分，人脸的检测并不可取。

        init_ratio = self.min_ratio
        # Assign label with small portions to label with large portion
        new_seg = seg.copy()
        [h,w] = new_seg.shape
        n_pixels = h*w
        # First scan through what are the available labels and their sizes
        label_info = []
        ratio_info = []
        new_label_info = []
        for label in np.unique(seg):
            ratio = np.sum(np.float32((seg == label))[:])/n_pixels
            label_info.append(label)
            new_label_info.append(label)
            ratio_info.append(ratio)
        for i,current_label in enumerate(label_info):
            if ratio_info[i] < init_ratio:
                for j in range(self.label_mapping.shape[0]):
                    new_label = self.label_mapping[j,current_label]
                    if new_label in label_info:
                        index = label_info.index(new_label)
                        if index >= 0:
                            if ratio_info[index] >= init_ratio:
                                new_label_info[i] = new_label
                                break
        for i,current_label in enumerate(label_info):
            new_seg[(seg == current_label)] = new_label_info[i]
        return new_seg


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'], option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'
        model_path = './xiaoke_video_checkpoints/'
        encoder_path = 'xiaoke_encoder.pth'
        decoder_path = 'xiaoke_decoder_0.0001_4.pth'

        model_path = './xiaoke_checkpoints/'
        encoder_path = 'xiaoke_encoder.pth'
        decoder_path = 'xiaoke_decoder_87.pth'
        self.device = torch.device(device)
        self.verbose = verbose
        # self.encoder = WaveEncoder(option_unpool).to(self.device)
        # self.decoder = WaveDecoder(option_unpool).to(self.device)
        # self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        # self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

        self.encoder = XiaoKeEncoder(option_unpool).to(self.device)
        self.decoder = XiaoKeDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path,encoder_path),map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path,decoder_path),map_location=lambda storage, loc: storage))    


    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1,is_wct=False):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                if is_wct:
                    content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
                else:
                    content_feat = feature_adin(content_feat, style_feats['encoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
                self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                if is_wct:
                    content_skips[skip_level] = feature_wct(content_skips[skip_level], style_skips[skip_level],
                                                                    content_segment, style_segment,
                                                                    label_set, label_indicator,
                                                                    alpha=alpha, device=self.device)
                else :
                    content_skips[skip_level] = feature_adin(content_skips[skip_level], style_skips[skip_level],
                                                                    content_segment, style_segment,
                                                                    label_set, label_indicator,
                                                                    alpha=alpha, device=self.device)
                self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                if is_wct:
                    content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
                else :
                    content_feat = feature_adin(content_feat, style_feats['decoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
                self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder']:
        for d in ['decoder']:
            for s in ['skip']:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret

# def get_single_transfer():
#     return ['encoder', 'decoder', 'skip']

def segment_this_img(f):
    # imread method of scipy.misc
    img = imread(f, mode='RGB')
    img = img[:, :, ::-1]  # BGR to RGB!!!
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    # imgsize [300,400,500,600]
    for this_short_size in config.imgSize:
        scale = this_short_size / float(min(ori_height, ori_width))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, config.padding_constant)
        target_width = round2nearest_multiple(target_width, config.padding_constant)
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input = dict()
    input['img_ori'] = img.copy()
    input['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (img.shape[0],img.shape[1])
    with torch.no_grad():
        pred = torch.zeros(1, config.num_class, segSize[0], segSize[1])
        for timg in img_resized_list:
            feed_dict = dict()
            feed_dict['img_data'] = timg.cuda()
            feed_dict = async_copy_to(feed_dict, config.gpu_id)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp.cpu() / len(config.imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0))
    return preds

def run_bulk(label_remapping):
    accurate_segment = True
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')
    cw, ch =  640,400                 
    sw, sh = 640,400 
    # cw, ch =  400,400              
    # sw, sh = 400,400
    # The filenames of the content and style pair should match
    c_transforms = transforms.Compose([transforms.Resize((ch,cw), interpolation=Image.NEAREST),transforms.CenterCrop((ch // 16 * 16, cw // 16 * 16)),transforms.ToTensor()])
    c_seg_transforms = transforms.Compose([transforms.Resize((ch,cw), interpolation=Image.NEAREST),transforms.CenterCrop((ch // 16 * 16, cw // 16 * 16))])

    fnames = os.listdir(config.content)
    fnames.sort()
    sample_fnames= fnames[:50]
    print('transfer at ~~~~',transfer_at)
    style = Image.open(config.style).convert('RGB')
    _style_segment = segment_this_img(config.style)
    _style_segment_path = os.path.join(config.style_segment,'style_segment.png')
    print('style_segment_path is~~~~~~',_style_segment_path)
    cv2.imwrite(_style_segment_path,_style_segment)
    style = c_transforms(style).unsqueeze(0).to(device)

    for fname in tqdm.tqdm(sample_fnames):
        if not is_image_file(fname):
            print('invalid file (is not image), ', fname)
            continue
        print('config.wct  is   ',config.is_wct)

        # content
        _content = os.path.join(config.content, fname)
        content = Image.open(_content).convert('RGB') # 别忘了这边的to(device)
                   
        content = c_transforms(content).unsqueeze(0).to(device)
        print('current frame {} and shape is {}'.format(fname,content.shape))




        _content_segment = segment_this_img(_content)
        _content_segment_path = os.path.join(config.content_segment,fname)
        cv2.imwrite(_content_segment_path, _content_segment )

        # _content_segment = os.path.join(config.content_segment, fname) if config.content_segment else None
        # _style_segment = os.path.join(config.style_segment, fname) if config.style_segment else None
        _output = os.path.join(config.output, fname)
        print('_content_segment',_content_segment.shape)
        if accurate_segment:
            try:
                content_segment = Image.open(_content_segment_path)
                style_segment = Image.open(_style_segment_path)


            except:
                content_segment = []
                style_segment = []
        _content_segment = c_seg_transforms(content_segment)
        _style_segment = c_seg_transforms(style_segment)

        # 重新调整语义label
        content_segment = np.asarray(_content_segment)
        style_segment = np.asarray(_style_segment)
        # print('content and style shape',content.shape, style.shape)
        content_segment = label_remapping.self_remapping(content_segment)
        style_segment = label_remapping.self_remapping(style_segment)
        content_segment, style_segment = label_remapping.cross_remapping(content_segment, style_segment)
        print('content and style segment shape',content_segment.shape,style_segment.shape)



        if not config.transfer_all:
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                postfix = '_'.join(sorted(list(transfer_at)))
                fname_output = _output.replace('.png', '_{}_{}.png'.format(config.option_unpool, postfix))
                print('------ transfer:', _output)
                wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
                with torch.no_grad():
                    img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha,is_wct=config.is_wct)

                save_image(img.clamp_(0, 1), fname_output, padding=0)
        else:
            for _transfer_at in get_all_transfer():
                print('location for transfer at~~~~',_transfer_at)
                with Timer('Elapsed time in whole WCT: {}', config.verbose):
                    postfix = '_'.join(sorted(list(_transfer_at)))
                    fname_output = _output.replace('.png', '_{}_{}.png'.format(config.option_unpool, postfix))
                    print('------ transfer:', fname,'-',_transfer_at)
                    wct2 = WCT2(transfer_at=_transfer_at, option_unpool=config.option_unpool, device=device, verbose=config.verbose)
                    # print('wct2 model encoder ',wct2.encoder)
                    # print('wcr2 model decoder ',wct2.decoder)
                    with torch.no_grad():
                        starttime = datetime.datetime.now()
                        img = wct2.transfer(content, style, content_segment, style_segment, alpha=config.alpha,is_wct=config.is_wct)
                        endtime = datetime.datetime.now()
                        print('xiaoke with adin 运行时间为----',(endtime - starttime))
                    save_image(img.clamp_(0, 1), fname_output, padding=0)
                # break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./examples/content')
    parser.add_argument('--content_segment', type=str, default='./examples/content_segment')
    parser.add_argument('--style', type=str, default='./examples/style')
    parser.add_argument('--style_segment', type=str, default='./examples/style_segment')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--is_wct',action='store_true')
    parser.add_argument('--label_mapping', type=str, default='ade20k_semantic_rel.npy')
    parser.add_argument('--model_path', help='folder to model path', default='baseline-resnet50_dilated8-ppm_bilinear_deepsup')
    parser.add_argument('--arch_encoder', default='resnet50_dilated8', help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='ppm_bilinear_deepsup', help="architecture of net_decoder")
    parser.add_argument('--suffix', default='_epoch_20.pth', help="which snapshot to load")
    parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')
    parser.add_argument('--num_class', default=150, type=int, help='number of classes')
    parser.add_argument('--padding_constant', default=8, type=int, help='maxmimum downsampling rate of the network')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')
    parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')

    # 不能定义两次同样的参数
    config = parser.parse_args()
    segReMapping = MySegReMapping(config.label_mapping)

    # Absolute paths of segmentation model weights  
    SEG_NET_PATH = 'segmentation'                                  
    config.weights_encoder = os.path.join(SEG_NET_PATH,config.model_path, 'encoder' + config.suffix)
    config.weights_decoder = os.path.join(SEG_NET_PATH,config.model_path, 'decoder' + config.suffix)
    config.arch_encoder = 'resnet50_dilated8'
    config.arch_decoder = 'ppm_bilinear_deepsup'
    config.fc_dim = 2048

    # Load semantic segmentation network module
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=config.arch_encoder, fc_dim=config.fc_dim, weights=config.weights_encoder)
    net_decoder = builder.build_decoder(arch=config.arch_decoder, fc_dim=config.fc_dim, num_class=config.num_class, weights=config.weights_decoder, use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    segmentation_module.eval()
    transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
    print(config)
    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))
    run_bulk(segReMapping)

'''
CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
'''



'''
python xiaoketransfer.py --content ./examples/demo_content/ --style ./examples/demo_style/ -a --output ./examples/demo_stylization --is_wct --image_size 400
CUDA_VISIBLE_DEVICES=1 python xiaoketransfer.py --content ./examples/dataset/alley_2/ --style ./examples/dataset/fangao.png -a --output ./examples/stylization 

CUDA_VISIBLE_DEVICES=1 python xiaoketransfer.py --content ./examples/data/MPI-Sintel-complete/training/clean/temple_2 --style ./examples/data/fangao.png -a --output ./examples/stylization 


'''