# TODO: convert into torchscript

# import copy
# from operator import mod
from lib2to3.pgen2.tokenize import TokenError
import os
# import time
# from collections import Counter, deque

# import cv2
# import face_alignment
import numpy as np
import torch
# import torch.nn.functional as F
# import torchvision
# from imutils import face_utils
# from PIL import Image
# from scipy.spatial import distance as dist
# from scipy.stats import entropy
# from torchvision import transforms, utils

# import util
from configs.image_cfg import _C as cfg
# from dataset import make_data_loader
# from dataset.transforms import build_transforms
# from model.graph_net import Graph_Net
from model.overall_net import Net

from torch.utils.mobile_optimizer import optimize_for_mobile

import random
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)



# os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True

label_template = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
num_classes    = len(label_template)

with torch.no_grad():

    # define and load model
    model_path = cfg.MODEL.SAVE_WEIGHT_PATH+'.pth'
    model = Net(cfg, num_classes)#.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # output = model(f_tensor, c_tensor)
    # Input:
    # - f_tensor.size(): [batch_size, 3, 112, 112]
    # - c_tensor.size(): [batch_size, 3, 112, 112]
    # Output:
    # - output.size(): [batch_size, 7]
    # f_tensor = torch.rand(1, 3, 112, 112)#.to(device)
    # c_tensor = torch.rand(1, 3, 112, 112)#.to(device)
    # f_tensor = torch.ones(1, 3, 112, 112)#.to(device)
    # c_tensor = torch.ones(1, 3, 112, 112)#.to(device)
    # torch.use_deterministic_algorithms(True)
    # output = model(f_tensor, c_tensor)
    # print(output.size())
    # print(output)
    
    # traced_script_module = torch.jit.script(Net(cfg, num_classes))  # ERROR: Don't use this, wrong way to script!
    traced_script_module = torch.jit.script(model)
    
    # traced_script_module = torch.jit.trace(model, (f_tensor, c_tensor))
    optimized_traced_model = optimize_for_mobile(traced_script_module)
    optimized_traced_model._save_for_lite_interpreter("./model_scripted.pt")
    # traced_script_module.save("./model_scripted.pt")
