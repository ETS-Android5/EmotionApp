import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import cv2
import torchvision.transforms as T
from configs.image_cfg import _C as cfg
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import List

# fix random seed to get consistent results, or else all three type of model have random results
torch.manual_seed(0)

class EmotionRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.fa = torch.jit.load("./FaceAlignment_scripted.pt")
        # self.face_detector = torch.jit.load("./s3fd_scripted.pt")
        self.model = torch.jit.load("./model_scripted.pt")
        # self.model = torch.jit.load("./model_traced.pt")
        # self.fa.eval()
        self.model.eval()
        self.normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        # self.f_resize = T.Resize(cfg.INPUT.SIZE_TEST, interpolation=T.InterpolationMode.BILINEAR)
        self.f_transform = nn.Sequential(
            # T.Resize(cfg.INPUT.SIZE_TEST, interpolation=T.InterpolationMode.BILINEAR),
            self.normalize_transform
        )
        # self.c_resize = T.Resize((128, 171), interpolation=T.InterpolationMode.BILINEAR)
        self.c_transform = nn.Sequential(
            # T.Resize((128, 171), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(cfg.INPUT.SIZE_TEST),
            self.normalize_transform
        )
    
    def resize(self, x, size: List[int]):
        return F.resize(x, size)
    
    @torch.no_grad()
    def forward(self, frame: torch.Tensor):
        frame = self.resize(frame, [112,])
        frame = frame.squeeze().permute(1, 2, 0)  # TODO: need to shirnk? cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        face, context, landmark = self.detFace(frame)
        if len(landmark.size()) == 0:
            return torch.tensor([[0.]])
        else:
            face, context = face.permute(2, 0, 1).to(torch.float32).unsqueeze(0).div(255), context.permute(2, 0, 1).to(torch.float32).unsqueeze(0).div(255)
            # face, context = face.permute(2, 0, 1).unsqueeze(0), context.permute(2, 0, 1).unsqueeze(0)
            f_tensor = self.resize(face, [112, 112])
            f_tensor = self.f_transform(f_tensor)
            c_tensor = self.resize(context, [128, 171])
            c_tensor = self.c_transform(c_tensor)
            # f, axarr = plt.subplots(2,1) 
            # axarr[0].imshow(f_tensor[0].permute(1, 2, 0))#f_tensor[0].permute(1, 2, 0).clamp(0, 1))
            # axarr[1].imshow(c_tensor[0].permute(1, 2, 0))#c_tensor[0].permute(1, 2, 0).clamp(0, 1))
            # plt.show()
            # print(f"{f_tensor.size()=}, {c_tensor.size()=}, {f_tensor.dtype}")
            output = self.model(f_tensor, c_tensor)
            return output
            # return landmark[:,0]


    def detFace(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = frame.clone().mul(std).add(mean).mul(255).to(torch.uint8)  # Inverse normalization and range 0~255
        # plt.imshow(image.to(torch.uint8))
        # plt.show()
        w, h, c = image.shape
        # preds = model.get_landmarks_from_image(process_image)

        # preds = [torch.tensor([[ 40.,  52.],
        # [ 43.,  59.],
        # [ 46.,  65.],
        # [ 50.,  71.],
        # [ 56.,  76.],
        # [ 62.,  80.],
        # [ 68.,  82.],
        # [ 77.,  84.],
        # [ 87.,  84.],
        # [ 94.,  79.],
        # [ 96.,  75.],
        # [ 98.,  71.],
        # [100.,  63.],
        # [100.,  55.],
        # [100.,  47.],
        # [ 99.,  42.],
        # [ 97.,  35.],
        # [ 49.,  39.],
        # [ 53.,  35.],
        # [ 57.,  32.],
        # [ 61.,  30.],
        # [ 64.,  30.],
        # [ 78.,  26.],
        # [ 81.,  25.],
        # [ 84.,  24.],
        # [ 89.,  24.],
        # [ 92.,  26.],
        # [ 73.,  33.],
        # [ 75.,  38.],
        # [ 76.,  41.],
        # [ 77.,  45.],
        # [ 72.,  50.],
        # [ 74.,  50.],
        # [ 78.,  49.],
        # [ 80.,  48.],
        # [ 82.,  47.],
        # [ 57.,  41.],
        # [ 59.,  39.],
        # [ 62.,  38.],
        # [ 65.,  38.],
        # [ 63.,  40.],
        # [ 60.,  41.],
        # [ 79.,  33.],
        # [ 82.,  31.],
        # [ 87.,  30.],
        # [ 89.,  31.],
        # [ 87.,  32.],
        # [ 83.,  33.],
        # [ 65.,  63.],
        # [ 70.,  58.],
        # [ 76.,  53.],
        # [ 79.,  53.],
        # [ 81.,  52.],
        # [ 88.,  52.],
        # [ 92.,  57.],
        # [ 90.,  63.],
        # [ 87.,  66.],
        # [ 82.,  68.],
        # [ 78.,  69.],
        # [ 73.,  68.],
        # [ 66.,  63.],
        # [ 75.,  57.],
        # [ 79.,  56.],
        # [ 83.,  55.],
        # [ 91.,  57.],
        # [ 85.,  63.],
        # [ 81.,  64.],
        # [ 77.,  65.]])]

        
        preds = self.fa(image)
        # print(preds)
        max_landmark = torch.tensor(0.)
        if preds is not None and len(preds) > 0:
            max_face_area = 0
            max_face_info = [0, 0, 0, 0]
            
            # print(f"{len(preds)=}, {type(preds)=}")
            for i, landmark in enumerate(preds):  # landmark.size(): torch.Size([68, 2])
                x1 = torch.min(landmark[:, 0])
                y1 = torch.min(landmark[:, 1])
                x2 = torch.max(landmark[:, 0])
                y2 = torch.max(landmark[:, 1])

                box_len = int((float(torch.max(x2 - x1, y2 - y1)) / 2.) * 1.3)
                center_x = int(float(x1 + x2) / 2.)
                center_y = int(float(y1 + y2) / 2.)
                
                if box_len > max_face_area:
                    max_face_area = box_len
                    max_face_info = [center_x, center_y]
                    max_landmark = landmark

            sx = max_face_info[0] - max_face_area
            ex = max_face_info[0] + max_face_area
            sy = max_face_info[1] - max_face_area
            ey = max_face_info[1] + max_face_area
            face = image[sx:ex, sy:ey, :]
            wf, hf, cf = face.shape
            # face = image[sy:ey, sx:ex, :]  # FIXME: original order of h and w are inversed!
            # hf, wf, cf = face.shape  # FIXME: original order of h and w are inversed!
            result = torch.full((max_face_area * 2, max_face_area * 2, c), 0, dtype=torch.float32)#uint8)
            xx = (max_face_area * 2 - wf) // 2
            yy = (max_face_area * 2 - hf) // 2
            result[xx:xx+wf, yy:yy+hf] = image[sx:ex, sy:ey, :]
            # result[xx:xx+wf, yy:yy+hf] = frame[sx:ex, sy:ey, :]
            # result[yy:yy+hf, xx:xx+wf] = image[sy:ey, sx:ex, :]  # FIXME: original order of h and w are inversed!
            # FIXME: However, correcting the w, h order will make trained model worse. So don't change it
            
            # print(result.size(), image.size())
            
            return result, image, max_landmark  # FIXME: only return the landmark for last detected face. What if mutiple faces?
            # return result, frame, preds
            # return result, image, preds
        else:
            # print("NO face detected")
            mid_w = w // 2
            mid_h = h // 2
            box_len = min(mid_h, mid_w) // 2
            return image[(mid_w - box_len):(mid_w + box_len), (mid_h - box_len):(mid_h + box_len), :], image, torch.tensor(0.)


if __name__ == "__main__":

    emotion_recognition = EmotionRecognition()

    with torch.no_grad():
        input = cv2.imread("./test.jpg")  # read as uint8, shape: HxWx3, same as original input
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # plt.imshow(input)
        # plt.show()
        # input = io.imread("./test.jpg")
        input = torch.tensor(input, dtype=torch.float32)  # remember to set as float32
        input = input.div(255)  # set range to 0~1
        input = input.permute(2, 0, 1).unsqueeze(0)
        normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        input = normalize(input)  # RGB, 0~1, normalized & shape: 1x3xHxW, same as output of bitmapToFloat32Tensor
        # print(torch.max(input), torch.min(input))
        # plt.imshow(input[0].permute(1,2,0))
        # print(input.size(), input.mean(), input.std())
        # input = torch.rand(112, 112, 3)
        # output = emotion_recognition(input)
        # print(f"{len(output)=}, {output[0].shape}")
        # print(output)

        traced_script_module = torch.jit.script(emotion_recognition, input)
        # traced_script_module = torch.jit.trace(emotion_recognition, input)
        # traced_script_module = torch.jit.trace(emotion_recognition, input, strict=False)
        optimized_traced_model = optimize_for_mobile(traced_script_module)
        optimized_traced_model._save_for_lite_interpreter("./EmotionRecognition_scripted.pt")
        
        # # run torchscript
        # torchscript = torch.jit.load("./EmotionRecognition_scripted.pt")
        # # print(torchscript.code)
        # output = torchscript(input)
        # # print(f"{len(output)=}, {output[0].shape}")
        # # print(len(output), [o.shape for o in output])
        # print(output)
        

    # pip installed model, result should be the same
    # face_alignment = SFDDetector(device="cpu")
    # output = face_alignment.detect_from_image(input)
