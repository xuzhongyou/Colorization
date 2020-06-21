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
from utils.core import feature_adin_without_segment
from utils.core import feature_wct_without_segment
from utils.io import Timer, open_image, load_segment, compute_label_info
from xiaokemodel import XiaoKeEncoder, XiaoKeDecoder
import numpy as np
import torchvision.transforms as transforms

from scipy.io import loadmat
from PIL import Image
from scipy.misc import imread, imresize
import cv2
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy, mark_volatile
import datetime


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
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']
        label_set,label_indicator = None, None
        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                if is_wct:
                    content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                            content_segment, style_segment,
                                            label_set, label_indicator,
                                            alpha=alpha, device=self.device)
                else:
                    content_feat = feature_adin_without_segment(content_feat, style_feats['encoder'][level],
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
                    content_skips[skip_level] = feature_adin_without_segment(content_skips[skip_level], style_skips[skip_level],
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
                    content_feat = feature_adin_without_segment(content_feat, style_feats['decoder'][level],
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



def run_bulk():
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
    # cw, ch =  640,360                  
    cw, ch =  640,400                  

    # The filenames of the content and style pair should match
    c_transforms = transforms.Compose([transforms.Resize((ch,cw), interpolation=Image.NEAREST),transforms.CenterCrop((ch // 16 * 16, cw // 16 * 16)),transforms.ToTensor()])

    fnames = os.listdir(config.content)
    fnames.sort()
    print('transfer at ~~~~',transfer_at)
    style = Image.open(config.style).convert('RGB')
    style = c_transforms(style).unsqueeze(0).to(device)
    sample_fnames = fnames[:50]
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


        # _content_segment = os.path.join(config.content_segment, fname) if config.content_segment else None
        # _style_segment = os.path.join(config.style_segment, fname) if config.style_segment else None
        _output = os.path.join(config.output, fname)

        content_segment,style_segment = None,None
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


    transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
    print(config)
    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))
    run_bulk()

# 树林的图片
# 171124_D1_HD_01
# 170216A_122_ForestTrail_1080
# 170216A_070_LookingUpThroughForest_1080  
# 180705_01_0
# 190416_10_Drone1_0
# Forest_15_1_Videv
# Forest_15_4_Videv
# on
# WalkingThroughTreesatSunsetVidev


# 树叶
# Autumn_leaves_in_motion_0
# autumn_leaves
# autumn-leaves-blowing-in-the-wind-H264
# 180705_01_0

# 海浪
# 46234354
# walking_on_the_beac

# 雪山
# 180607_A_00

# 开车
# 180607_A_10

# 飞机
# Airbus_A380_Landing_2__Videv
# Evening_landin
# PlaneLand

# 海边瑜伽
# Ao_Nang_Beach_Yoga_MP4_HDV_1080p25__TanuriX_Stock_Footage_N
# MVI_126

# 水稻
# Barley_3_Videv
# HandStrokin

# wild_gras
# windygrassnoaudi-

# 船
# beach1
# sailing_boa

# 天空
# Becco_di_Filadonna_su_Vall
# Blue_Sky_and_Clouds_Timelapse_0892__Videv

# 老鼠
# CotswoldSequence


# 奶牛
# cow
# Cow_Mother_and_cal
# Cows_
# Limousin_Cows_1__VIdev
# Limousin_Cows_2__Videv

# 日落
# Lonely_tree_at_sunset_CCBY_NatureCli
# MilkyWaywithTreeVidev
# SilhouetteJogge
# Sun_to_Sea_Model__Pan_Down_MP4_HDV_1080p25__TanuriX_Stock_Footage_W
# Sunris
# TimelapseSunse
# Wakeboarding_on_the_Lak


# 马
# Dirty_Hors

# 黑白鸟
# Pigeon-Stock-Vide
# Red_fod
# Weave

# 海鸥
# seagul-H264
# seagulls_on_the_beac


# 建筑
# Run_5_wo_metadata_h264420_720p_UH

# 鸭子
# SeaBirdsSwimming_
# Swans__1287_

# 羊
# Shee

'''
CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
'''



'''
python xiaoketransfer.py --content ./examples/demo_content/ --style ./examples/demo_style/ -a --output ./examples/demo_stylization --is_wct --image_size 400
CUDA_VISIBLE_DEVICES=1 python xiaoketransfer.py --content ./examples/dataset/alley_2/ --style ./examples/dataset/fangao.png -a --output ./examples/stylization 

CUDA_VISIBLE_DEVICES=1 python xiaoketransfer2.py --content ./examples/data/MPI-Sintel-complete/training/clean/temple_2 --style ./examples/data/fangao.png -a --output ./examples/stylization 

CUDA_VISIBLE_DEVICES=1 python xiaoketransfer2.py --content ./examples/data/MPI-Sintel-complete/training/clean/mountain_1 --style ./examples/data/fangao.png -a --output ./examples/stylization 

CUDA_VISIBLE_DEVICES=1 python xiaoketransfer2.py --content ./examples/data/MPI-Sintel-complete/training/clean/temple_2 --style ./examples/data/fangao.png -a --output ./examples/stylization --is_wct 

'''


'''
'../data/video-picture/160825_26_WindTurbines4_1080'
python xiaoketransfer2.py --content ../data/video-picture/160825_26_WindTurbines4_1080 --style ./examples/data/fangao.png -a --output ./examples/160825_26_WindTurbines4_1080_adain  

'''

'''
'../data/video-picture/xxx'
python xiaoketransfer2.py --content ../data/video-picture/180705_01_0 --style ../data/reference/tar0056_orange_forest.png -a --output ./examples/Forest_15_4_Videv  

python xiaoketransfer.py --content ../data/video-picture/Red_fod --style ../data/video-picture/Weave/frame_0001.png -a --output ./examples/Red_fod_seg


python xiaoketransfer.py --content ../data/video-picture/seagulls_on_the_beac --style ../data/video-picture/seagul-H264/frame_0001.png -a --output ./examples/seagulls_on_the_beac_seg

python xiaoketransfer2.py --content ../data/video-picture/HandStrokin --style ../data/video-picture/Barley_3_Videv/frame_0001.png -a --output ./examples/HandStrokin

python xiaoketransfer.py --content ../data/video-picture/Swans__1287_ --style ../data/video-picture/SeaBirdsSwimming_/frame_0001.png -a --output ./examples/Swans__1287_

python xiaoketransfer.py --content ../data/video-picture/Becco_di_Filadonna_su_Vall --style ../data/video-picture/Blue_Sky_and_Clouds_Timelapse_0892__Videv/frame_0001.png -a --output ./examples/Becco_di_Filadonna_su_Vall

python xiaoketransfer.py --content ../data/video-picture/Sun_to_Sea_Model__Pan_Down_MP4_HDV_1080p25__TanuriX_Stock_Footage_W --style ../data/video-picture/Lonely_tree_at_sunset_CCBY_NatureCli/frame_0004.png -a --output ./examples/Sun_to_Sea_Model__Pan_Down_MP4_HDV_1080p25__TanuriX_Stock_Footage_W


python xiaoketransfer2.py --content ../data/video-picture/Wakeboarding_on_the_Lak --style ../data/video-picture/Sunris/frame_0004.png -a --output ./examples/Wakeboarding_on_the_Lak

# Barley_3_Videv
# HandStrokin

# Pigeon-Stock-Vide
# Red_fod
'''