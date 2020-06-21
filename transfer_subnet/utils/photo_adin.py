"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from models.models import VGGEncoder, VGGDecoder


class PhotoADIN(nn.Module):
    def __init__(self):
        super(PhotoADIN, self).__init__()
        self.e1 = VGGEncoder(1)
        self.d1 = VGGDecoder(1)
        self.e2 = VGGEncoder(2)
        self.d2 = VGGDecoder(2)
        self.e3 = VGGEncoder(3)
        self.d3 = VGGDecoder(3)
        self.e4 = VGGEncoder(4)
        self.d4 = VGGDecoder(4)
    
    def transform(self, cont_img, styl_img, cont_seg, styl_seg):
        self.__compute_label_info(cont_seg, styl_seg)
        # channel 512 256 128 64 
        sF4, sF3, sF2, sF1 = self.e4.forward_multiple(styl_img)
        # print('sF4,sF3,sF2,sF1---',sF4.shape,sF3.shape,sF2.shape,sF1.shape)
        cF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3 = self.e4(cont_img)
        # print('cF4,pool_index,cpool1--',cF4.shape)
        # print('pool_index,cpool--',cpool_idx,cpool1)
        # print('cpool1.shape--',cpool1.shape)
        sF4 = sF4.data.squeeze(0)
        cF4 = cF4.data.squeeze(0)
        # print(cont_seg)
        # csF4 shape is 512
        csF4 = self.__feature_adin(cF4, sF4, cont_seg, styl_seg)
        # Im4 is the same size with Im4.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        Im4 = self.d4(csF4, cpool_idx, cpool1, cpool_idx2, cpool2, cpool_idx3, cpool3)
        cF3, cpool_idx, cpool1, cpool_idx2, cpool2 = self.e3(Im4)
        sF3 = sF3.data.squeeze(0)
        cF3 = cF3.data.squeeze(0)
        csF3 = self.__feature_adin(cF3, sF3, cont_seg, styl_seg)
        Im3 = self.d3(csF3, cpool_idx, cpool1, cpool_idx2, cpool2)

        cF2, cpool_idx, cpool = self.e2(Im3)
        sF2 = sF2.data.squeeze(0)
        cF2 = cF2.data.squeeze(0)
        csF2 = self.__feature_adin(cF2, sF2, cont_seg, styl_seg)
        Im2 = self.d2(csF2, cpool_idx, cpool)

        cF1 = self.e1(Im2)
        sF1 = sF1.data.squeeze(0)
        cF1 = cF1.data.squeeze(0)
        csF1 = self.__feature_adin(cF1, sF1, cont_seg, styl_seg)
        Im1 = self.d1(csF1)
        return Im1

    def __compute_label_info(self, cont_seg, styl_seg):
        if cont_seg.size == False or styl_seg.size == False:
            return
        max_label = np.max(cont_seg) + 1
        # print('cont_seg---',cont_seg,cont_seg.shape)
        self.label_set = np.unique(cont_seg)
        self.label_indicator = np.zeros(max_label)
        # print('label_set---',self.label_set,self.label_set.shape)
        # print('label_indicator',self.label_indicator)
        for l in self.label_set:
            # if l==0:
            #   continue
            is_valid = lambda a, b: a > 10 and b > 10 and a / b < 100 and b / a < 100
            # np.where return the index of the reshaped cont_seg
            o_cont_mask = np.where(cont_seg.reshape(cont_seg.shape[0] * cont_seg.shape[1]) == l)
            o_styl_mask = np.where(styl_seg.reshape(styl_seg.shape[0] * styl_seg.shape[1]) == l)
            # judge the label mask 
            self.label_indicator[l] = is_valid(o_cont_mask[0].size, o_styl_mask[0].size)
    

    def __feature_adin(self,cont_feat,styl_feat,cont_seg,styl_seg):
        # 这里的 feature 是三维的
        cont_c, cont_h, cont_w = cont_feat.size(0), cont_feat.size(1), cont_feat.size(2)
        styl_c, styl_h, styl_w = styl_feat.size(0), styl_feat.size(1), styl_feat.size(2)
        # clone 的作用
        cont_feat_view = cont_feat.view(cont_c, -1).clone()
        styl_feat_view = styl_feat.view(styl_c, -1).clone()

        if cont_seg.size == False or styl_seg.size == False:
            target_feature = self.__adaptive_instance_normalization(cont_feat_view, styl_feat_view)
        else:
            target_feature = cont_feat.view(cont_c, -1).clone()
            if len(cont_seg.shape) == 2:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg).resize((cont_w, cont_h), Image.NEAREST))
            else:
                t_cont_seg = np.asarray(Image.fromarray(cont_seg, mode='RGB').resize((cont_w, cont_h), Image.NEAREST))
            if len(styl_seg.shape) == 2:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg).resize((styl_w, styl_h), Image.NEAREST))
            else:
                t_styl_seg = np.asarray(Image.fromarray(styl_seg, mode='RGB').resize((styl_w, styl_h), Image.NEAREST))

            for l in self.label_set:
                if self.label_indicator[l] == 0:
                    # label 0 isn't the label wanted
                    continue
                # 这里的 cont_mask 是一个tuple，np.where 返回一个tuple，但是只需要用到索引部分,如：cont_mask[0]
                cont_mask = np.where(t_cont_seg.reshape(t_cont_seg.shape[0] * t_cont_seg.shape[1]) == l)
                styl_mask = np.where(t_styl_seg.reshape(t_styl_seg.shape[0] * t_styl_seg.shape[1]) == l)
                # print('cont_mask---',cont_mask)
                if cont_mask[0].size <= 0 or styl_mask[0].size <= 0:
                    continue

                cont_indi = torch.LongTensor(cont_mask[0])
                styl_indi = torch.LongTensor(styl_mask[0])
                # print('cont_indi shape ',cont_indi)
                if self.is_cuda:
                    cont_indi = cont_indi.cuda()
                    styl_indi = styl_indi.cuda()
                # print('cont_feat_view--styl_feat_view',cont_feat_view.shape,styl_feat_view.shape) 也是二维的
                cFFG = torch.index_select(cont_feat_view, 1, cont_indi)
                sFFG = torch.index_select(styl_feat_view, 1, styl_indi)
                # 这里的cFFG 和 sFFG 的维度需要一样吗？
                # print('ccfg----ssfg',cFFG.shape,sFFG.shape) 二维

                # print(len(cont_indi))
                # print(len(styl_indi))
                tmp_target_feature = self.__adaptive_instance_normalization(cFFG, sFFG)
                # print('tmp_target_feature---', tmp_target_feature.shape)# 也是二维的
                if torch.__version__ >= "0.4.0":
                    # This seems to be a bug in PyTorch 0.4.0 to me.
                    new_target_feature = torch.transpose(target_feature, 1, 0)
                    # print('new_target_feature--',new_target_feature.shape)
                    new_target_feature.index_copy_(0, cont_indi, \
                            torch.transpose(tmp_target_feature,1,0)) 
                    target_feature = torch.transpose(new_target_feature, 1, 0)
                    # print('final_target_feature--',target_feature.shape)
                else:
                    target_feature.index_copy_(1, cont_indi, tmp_target_feature)

        target_feature = target_feature.view_as(cont_feat)
        ccsF = target_feature.float().unsqueeze(0)
        return ccsF



    def __cal_mean_std(self,features,eps=1e-5):
        C,t = features.size()
        feature_var = features.var(dim=1) + eps
        feature_std = feature_var.sqrt().view(C,1)
        feature_mean = features.mean(dim=1).view(C,1)
        return feature_mean,feature_std

    def __adaptive_instance_normalization(self,content_feat, style_feat):
        cont_mean,cont_std = self.__cal_mean_std(content_feat)
        sty_mean,sty_std = self.__cal_mean_std(style_feat)
        normalized_feat = (content_feat-cont_mean.expand_as(content_feat)) / (cont_std.expand_as(content_feat))
        adin = sty_std.expand_as(content_feat) * normalized_feat + sty_mean.expand_as(content_feat)
        return adin


    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, *input):
        pass

