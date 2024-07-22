from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *
from .newcrf import NewCRF
from .spm import SPM
from .LGFI import LGFI

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, pretrained_x1='Final_models/depth.pth', freeze_x1=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.freeze_x1 = freeze_x1  

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.convs_train = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs_train[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs_train[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            
            self.convs_train[("reduced", i)] = Conv1x1(num_ch_out * 2, num_ch_out)
            self.convs_train[("last",i)] = Conv1x1(2, 1)
            
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs_train[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.decoder_training = nn.ModuleList(list(self.convs_train.values()))

        self.sigmoid = nn.Sigmoid()
        

        self.pretrained_x1 = pretrained_x1
        if self.pretrained_x1 is not None:
            self.load_pretrained_model(self.pretrained_x1)

    def load_pretrained_model(self, pretrained_model):
        """
        Load pretrained weights into the upper part of the decoder.
        """
        pretrained_dict = torch.load(pretrained_model)
        model_dict = self.state_dict()
        
        model_dict.update(pretrained_dict)
        # print(model_dict.keys())
        self.load_state_dict(model_dict)
        print("Loaded pretrained x1 weights")

       
        if self.freeze_x1:
            for name, param in self.named_parameters():
                if "decoder." in name:
                    param.requires_grad = False
            print("Freezing pre-trained weights")    

    def forward(self, input_features):
        self.outputs = {}

        self.pretrained_outputs = self.forward_pretrained(input_features)

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs_train[("upconv", i, 0)](x)           
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)


            x_temp = self.convs_train[("upconv", i, 1)](x)
            x3 = self.pretrained_outputs[('upconv', i, 1)]
            
            x = torch.cat([x_temp,x3], dim=1)
   
            x = self.convs_train[('reduced', i)](x)         
            if i in self.scales:                
                self.outputs[("disp_1", i)] = self.sigmoid(self.convs_train[("dispconv", i)](x))
                #Concatenation of pretrained and current output

                # self.outputs[("disp", i)] = torch.cat((self.pretrained_outputs[("disp", i)],self.outputs[("disp_1", i)]),dim=1)
                # self.outputs[("disp", i)] = self.convs_train[("last",i)](self.outputs[("disp", i)])

                self.outputs[("disp", i)] = (self.pretrained_outputs[("disp", i)] + 0.5 * self.outputs[("disp_1", i)])

               
        return self.outputs

    def forward_pretrained(self, input_features):
        
        self.outputs = {}
       
        x = input_features[-1]
        for i in range(4, -1, -1):

            x = self.convs[("upconv", i, 0)](x)

            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            self.outputs[("upconv", i, 1)] = self.convs[("upconv", i, 1)](x)
            x = self.convs[("upconv", i, 1)](x)

        
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs