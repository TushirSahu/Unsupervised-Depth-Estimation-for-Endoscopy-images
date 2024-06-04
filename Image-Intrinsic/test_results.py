from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_diff = np.mean(np.abs(gt - pred))
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_diff, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate_single_image(opt, image_path):
    """Evaluates a pretrained model on a single image and plots the output depth map"""
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    input_image = Image.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((encoder_dict['width'], encoder_dict['height']), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0).cuda()

    with torch.no_grad():
        output = depth_decoder(encoder(input_image))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
        pred_disp = pred_disp.cpu()[:, 0].numpy()

    pred_disp_resized = cv2.resize(pred_disp[0], (original_width, original_height))
    pred_depth = 1 / pred_disp_resized

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(input_image[0].permute(1, 2, 0).cpu().numpy())
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_depth, cmap='magma')
    plt.title("Predicted Depth")
    plt.axis('off')
  
    plt.show()
    cv2.imwrite("new1.pgm",pred_depth)
    
if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    image_path = "daVinci/test/image_0/000001.png"  # Replace with your image path
    evaluate_single_image(opt, image_path)
