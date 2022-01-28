import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import List, Tuple
from skimage import io
from face_alignment.detection.sfd.sfd_detector import SFDDetector


def nms(dets: torch.Tensor, thresh:float) -> List[int]:
    if 0 == len(dets):
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = torch.jit.annotate(List[int], [])
    while order.size(0) > 0:
        i = order[0]
        keep.append(i.item())
        xx1, yy1 = torch.maximum(x1[i], x1[order[1:]]), torch.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = torch.minimum(x2[i], x2[order[1:]]), torch.minimum(y2[i], y2[order[1:]])
        
        w, h = torch.maximum(torch.zeros_like(xx1), xx2 - xx1 + 1), torch.maximum(torch.zeros_like(xx1), yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]
    # print(type(keep[0].item()))

    return keep


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def decode(loc, priors, variances: List[float]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes






def get_predictions(olist: List[torch.Tensor], batch_size: int):
    bboxlists = [[[float(torch.tensor(0.).item())]]]
    variances = [0.1, 0.2]
    # bboxlist = torch.jit.annotate(List[List[torch.Tensor]], [])
    for j in range(batch_size):
        bboxlist = [[float(torch.tensor(0.).item())]]
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            # poss = zip(*torch.where(ocls[:, 1, :, :] > 0.05))
            
            # poss = []
            _poss = torch.where(ocls[:, 1, :, :] > 0.05)
            # print(_poss)
            # for i in range(len(_poss[0])):
            #     p = []
            #     for _p in _poss:
            #         p.append(_p[i])
            #     poss.append(p)
            for i in range(len(_poss[0])):
            # for Iindex, hindex, windex in poss:
                Iindex, hindex, windex = _poss[0][i], _poss[1][i], _poss[2][i]
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].clone().reshape(1, 4)
                priors = torch.tensor([[float(axc.item()) / 1.0, float(ayc.item()) / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0][0], box[0][1], box[0][2], box[0][3]
                bboxlist.append([float(x1.item()), float(y1.item()), float(x2.item()), float(y2.item()), float(score.item())])
            # print(type(bboxlist[-1]))
        bboxlists.append(bboxlist[1:])

    bboxlists_tensor = torch.tensor(bboxlists[1:])
    # print(bboxlists_tensor.size())  # torch.Size([1, 36, 5])
    return bboxlists_tensor



class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.empty(self.n_channels).fill_(self.scale))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class s3fd(nn.Module):
    def __init__(self):
        super(s3fd, self).__init__()
        
        self.device = 'cpu'
        self.filter_threshold = 0.5
        
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

        self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.detect_from_image(x)

    def face_detector(self, x):
        h = F.relu(self.conv1_1(x), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        f3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        f4_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        f5_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.fc6(h), inplace=True)
        h = F.relu(self.fc7(h), inplace=True)
        ffc7 = h
        h = F.relu(self.conv6_1(h), inplace=True)
        h = F.relu(self.conv6_2(h), inplace=True)
        f6_2 = h
        h = F.relu(self.conv7_1(h), inplace=True)
        h = F.relu(self.conv7_2(h), inplace=True)
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)

        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

    def _filter_bboxes(self, bboxlist):
        bboxlist_tmp = []
        if len(bboxlist) > 0:
            keep = nms(bboxlist, 0.3)
            # keep = keep.tolist()
            bboxlist = bboxlist[keep, :]
            # bboxlist = [x for x in bboxlist if x[-1] > self.filter_threshold]
            for x in bboxlist:
                if x[-1] > self.filter_threshold:
                    bboxlist_tmp.append(x)
                # else:
                #     bboxlist_tmp = bboxlist_tmp

        return bboxlist_tmp

    def detect_from_image(self, tensor):
        image = tensor

        bboxlist = self.detect(image)[0]
        bboxlist = self._filter_bboxes(bboxlist)

        return bboxlist

    def detect_from_batch(self, tensor):
        bboxlists = self.batch_detect(tensor, device=self.device)

        new_bboxlists = []
        for i in range(bboxlists.shape[0]):
            bboxlist = bboxlists[i]
            bboxlist = self._filter_bboxes(bboxlist)
            new_bboxlists.append(bboxlist)

        return new_bboxlists


    def detect(self, img: torch.Tensor) -> torch.Tensor:
        img = img.permute(2, 0, 1)
        # Creates a batch of 1
        img = img.unsqueeze(0)

        return self.batch_detect(img)


    def batch_detect(self, img_batch):
        """
        Inputs:
            - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
        """

        batch_size = int(img_batch.size(0))
        # img_batch = img_batch.to(device, dtype=torch.float32)

        img_batch = img_batch.flip(-3)  # RGB to BGR
        img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)

        # with torch.no_grad():
        olist = self.face_detector(img_batch)  # patched uint8_t overflow error

        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)

        # olist = [oelem.data.cpu().numpy() for oelem in olist]

        # print(type(batch_size))
        bboxlists = get_predictions(olist, batch_size)
        return bboxlists


    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0

models_urls = {
    's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
}

model_weights = load_url(models_urls['s3fd'])
face_detector = s3fd()
face_detector.load_state_dict(model_weights)
face_detector.to('cpu')
face_detector.eval()

with torch.no_grad():
    input = io.imread("./noFace.jpg")
    input = torch.tensor(input)
    # input = torch.rand(112, 112, 3)
    output = face_detector(input)
    print(len(output), [o.shape for o in output])
    print(output)

    # traced_script_module = torch.jit.script(face_detector, input)
    # traced_script_module = torch.jit.trace(face_detector, input)
    # traced_script_module = torch.jit.trace(face_detector, input, strict=False)
    # optimized_traced_model = optimize_for_mobile(traced_script_module)
    # optimized_traced_model._save_for_lite_interpreter("./s3fd_scripted.pt")
    
    # # run torchscript
    # torchscript = torch.jit.load("./s3fd_scripted.pt")
    # # print(torchscript.code)
    # output = torchscript(input)
    # print(len(output), [o.shape for o in output])
    # print(output)
    

# pip installed model, result should be the same
# face_detector = SFDDetector(device="cpu")
# output = face_detector.detect_from_image(input)