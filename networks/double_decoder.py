from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *
from .newcrf import NewCRF

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, pretrained_x1='Models/2024-06-26-11-20-09/models/weights_0/depth.pth', freeze_x1=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.freeze_x1 = freeze_x1  # Flag to freeze x1

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        self.training = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.training[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.training[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            
            self.training[("reduced", i)] = Conv1x1(num_ch_out * 2, num_ch_out)
        # for s in self.scales:
        #     self.convs[('reduced', s)] = Conv1x1(num_output_channels * 2, num_output_channels)
        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.training[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder_freeze = nn.ModuleList(list(self.convs.values()))
        self.decoder_training = nn.ModuleList(list(self.training.values()))

        self.sigmoid = nn.Sigmoid()

        # Load pretrained weights for x1 if specified
        self.pretrained_x1 = pretrained_x1
        if self.pretrained_x1 is not None:
            self.load_pretrained_model(self.pretrained_x1)

    def load_pretrained_model(self, pretrained_model):
        """
        Load pretrained weights into the x1 part of the decoder.
        """
        pretrained_dict = torch.load(pretrained_model)
        model_dict = self.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print("Loaded pretrained x1 weights")

       
        if self.freeze_x1:
            for name, param in self.named_parameters():
                if "decoder_freeze" in name:
                    param.requires_grad = False
            print("Freezing x1 weights")    

    def forward(self, input_features):
        self.outputs = {}

        self.pretrained_outputs = self.forward_pretrained(input_features)
        # print("Pretrained outputs:", self.convs_outputs.keys()) 
        # print("Pretrained outputs:", self.pretrained_outputs.keys())

        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.training[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.training[("upconv", i, 1)](x)

            x_2 = self.pretrained_outputs[('upconv',i,1)]
            
            x = torch.cat([x,x_2], dim=1)
            # if i in self.scales:
            x = self.training[('reduced', i)](x)

            if i in self.scales:
                
                self.outputs[("disp", i)] = self.sigmoid(self.training[("dispconv", i)](x))
                """
                Concatenate the two depth map predictions and apply a 1x1 convolution
                """
                # self.outputs[("disp", i)] = torch.cat((self.outputs[("disp", i)], 0.5 * self.pretrained_outputs[("disp", i)]),dim =1)
                # self.outputs[("disp", i)] = self.convs[('reduced', i)](self.outputs[("disp", i)])

                """
                Weighted sum of the two depth map predictions
                """
                # self.outputs[("disp", i)] = self.outputs[("disp", i)] * 0.4 + 0.6 * self.pretrained_outputs[("disp", i)]

        return self.outputs

    def forward_pretrained(self, input_features):
        
        self.outputs = {}
       
        x = input_features[-1]
        for i in range(4, -1, -1):

            self.outputs[("upconv", i, 0)] = self.convs[("upconv", i, 0)](x)
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