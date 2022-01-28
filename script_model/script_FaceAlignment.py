from numpy import float32
import torch
from torch import nn
import torch.nn.functional as F
# from utils import crop, get_preds_fromhm
from torch.utils.mobile_optimizer import optimize_for_mobile
import cv2
from typing import List




default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

class FaceAlignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'

        # if LooseVersion(torch.__version__) < LooseVersion('1.5.0'):
        #     raise ImportError(f'Unsupported pytorch version detected. Minimum supported version of pytorch: 1.5.0\
        #                     Either upgrade (recommended) your pytorch setup, or downgrade to face-alignment 1.2.0')

        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        # Get the face detector
        self.face_detector = torch.jit.load("./s3fd_scripted.pt")

        # Initialise the face alignemnt networks
        network_name = '2DFAN-4'
        
        # self.face_alignment_net = torch.jit.load("C:/Users/Alvin/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip")
        self.face_alignment_net = torch.jit.load("./face_alignment_net_scripted.pt")

        self.face_alignment_net.to(self.device)
        self.face_alignment_net.eval()
            
    @torch.no_grad()
    def forward(self, image_or_path: torch.Tensor):
        
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """

        image = image_or_path
        detected_faces = self.face_detector(image.clone())

        if len(detected_faces) == 0:
            # print("No faces were detected.")
            # return None
            nothing = torch.jit.annotate(List[torch.Tensor], [])
            return nothing

        landmarks = []
        landmarks_scores = []
        for i, d in enumerate(detected_faces):
            center = torch.tensor(
                [float((d[2] - (d[2] - d[0]) / 2.0).item()), float((d[3] - (d[3] - d[1]) / 2.0).item())])
            center[1] = center[1] - (d[3] - d[1]) * 0.12
            scale = (d[2] - d[0] + d[3] - d[1]) / 195

            inp = self.crop(image, center, float(scale.item()))
            inp = inp.permute(2, 0, 1).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)
            
            # print(inp.size())

            out = self.face_alignment_net(inp).detach()
            
            out = out.cpu()

            pts, pts_img, scores = self.get_preds_fromhm(out, center, float(scale.item()))
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            scores = scores.squeeze(0)

            landmarks.append(pts_img)
            landmarks_scores.append(scores)


        return landmarks


    # @torch.jit.script
    def transform(self, point: List[float], center, scale: float, resolution: float):
        
        """Generate and affine transformation matrix.

        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.

        Arguments:
            point {torch.tensor} -- the input 2D point
            center {torch.tensor or numpy.array} -- the center around which to perform the transformations
            scale {float} -- the scale of the face/object
            resolution {float} -- the output resolution

        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)

        t = self.inverse_3X3_matrix(t)
        # t = torch.inverse(t)

        new_point = (torch.matmul(t, _pt))[0:2]

        return new_point.int()


    # @torch.jit.script
    def crop(self, image: torch.Tensor, center: torch.Tensor, scale: float, resolution: float = 256.0) -> torch.Tensor:
        """Center crops an image or set of heatmaps

        Arguments:
            image {numpy.array} -- an rgb image
            center {numpy.array} -- the center of the object, usually the same as of the bounding box
            scale {float} -- scale of the face

        Keyword Arguments:
            resolution {float} -- the size of the output cropped image (default: {256.0})

        Returns:
            [type] -- [description]
        """  # Crop around the center point
        """ Crops the image around the center. Input is expected to be an np.ndarray """
        ul = self.transform([1., 1.], center, scale, resolution)
        br = self.transform([resolution, resolution], center, scale, resolution)
        # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
        if image.ndim > 2:
            newDim = [int((br[1] - ul[1]).item()), int((br[0] - ul[0]).item()), image.shape[2]]
            newImg = torch.zeros(newDim, dtype=torch.float32)
        else:
            newDim = [int((br[1] - ul[1]).item()), int((br[0] - ul[0]).item())]
            newImg = torch.zeros(newDim, dtype=torch.float32)
        ht = image.shape[0]
        wd = image.shape[1]
        newX = [max(1, -ul[0].item() + 1), min(br[0].item(), wd) - ul[0]]
        newY = [max(1, -ul[1].item() + 1), min(br[1].item(), ht) - ul[1]]
        oldX = [max(1, ul[0].item() + 1), min(br[0].item(), wd)]
        oldY = [max(1, ul[1].item() + 1), min(br[1].item(), ht)]
        newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
            ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
        # print('--------------', newImg.shape, newImg.dtype, image.dtype)
        newImg = newImg.permute(2, 0, 1)
        newImg = newImg.unsqueeze(1)
        # print('--------------', newImg.shape)
        newImg = F.interpolate(newImg, size=(int(resolution), int(resolution)), mode='bilinear')
        newImg = newImg.squeeze().permute(1, 2, 0)
        # print('--------------', newImg.shape)

        return newImg


    def inverse_3X3_matrix(self, _I_Q_list: torch.Tensor):
        # I_Q_list = torch.jit.annotate(List[List[float]], _I_Q_list.tolist())
        I_Q_list = _I_Q_list
        det_ = I_Q_list[0, 0] * (
                (I_Q_list[1, 1] * I_Q_list[2, 2]) - (I_Q_list[1, 2] * I_Q_list[2, 1])) - \
            I_Q_list[0, 1] * (
                    (I_Q_list[1, 0] * I_Q_list[2, 2]) - (I_Q_list[1, 2] * I_Q_list[2, 0])) + \
            I_Q_list[0, 2] * (
                    (I_Q_list[1, 0] * I_Q_list[2, 1]) - (I_Q_list[1, 1] * I_Q_list[2, 0]))
        co_fctr_1 = torch.stack(((I_Q_list[1, 1] * I_Q_list[2, 2]) - (I_Q_list[1, 2] * I_Q_list[2, 1]),
                    -((I_Q_list[1, 0] * I_Q_list[2, 2]) - (I_Q_list[1, 2] * I_Q_list[2, 0])),
                    (I_Q_list[1, 0] * I_Q_list[2, 1]) - (I_Q_list[1, 1] * I_Q_list[2, 0])))

        co_fctr_2 = torch.stack((-((I_Q_list[0, 1] * I_Q_list[2, 2]) - (I_Q_list[0, 2] * I_Q_list[2, 1])),
                    (I_Q_list[0, 0] * I_Q_list[2, 2]) - (I_Q_list[0, 2] * I_Q_list[2, 0]),
                    -((I_Q_list[0, 0] * I_Q_list[2, 1]) - (I_Q_list[0, 1] * I_Q_list[2, 0]))))

        co_fctr_3 = torch.stack(((I_Q_list[0, 1] * I_Q_list[1, 2]) - (I_Q_list[0, 2] * I_Q_list[1, 1]),
                    -((I_Q_list[0, 0] * I_Q_list[1, 2]) - (I_Q_list[0, 2] * I_Q_list[1, 0])),
                    (I_Q_list[0, 0] * I_Q_list[1, 1]) - (I_Q_list[0, 1] * I_Q_list[1, 0])))

        inv_list = torch.stack((torch.stack((1 / det_ * (co_fctr_1[0]), 1 / det_ * (co_fctr_2[0]), 1 / det_ * (co_fctr_3[0]))),
                    torch.stack((1 / det_ * (co_fctr_1[1]), 1 / det_ * (co_fctr_2[1]), 1 / det_ * (co_fctr_3[1]))),
                    torch.stack((1 / det_ * (co_fctr_1[2]), 1 / det_ * (co_fctr_2[2]), 1 / det_ * (co_fctr_3[2])))))

        # print(inv_list)
        # return torch.tensor(inv_list)
        return inv_list


    def transform_np(self, point, center, scale: float, resolution: int):
        """Generate and affine transformation matrix.

        Given a set of points, a center, a scale and a targer resolution, the
        function generates and affine transformation matrix. If invert is ``True``
        it will produce the inverse transformation.

        Arguments:
            point {numpy.array} -- the input 2D point
            center {numpy.array} -- the center around which to perform the transformations
            scale {float} -- the scale of the face/object
            resolution {float} -- the output resolution

        Keyword Arguments:
            invert {bool} -- define wherever the function should produce the direct or the
            inverse transformation matrix (default: {False})
        """
        _pt = torch.ones(3)
        _pt[0] = point[0]
        _pt[1] = point[1]

        h = 200.0 * scale
        t = torch.eye(3)
        t[0, 0] = resolution / h
        t[1, 1] = resolution / h
        t[0, 2] = resolution * (-center[0] / h + 0.5)
        t[1, 2] = resolution * (-center[1] / h + 0.5)
        # print(t)
        # t = torch.linalg.pinv(t)
        t = self.inverse_3X3_matrix(t)
        # print(t)
        
        new_point = torch.matmul(t, _pt)[0:2]

        return new_point.type(torch.int32)


    def get_preds_fromhm(self, hm: torch.Tensor, center: torch.Tensor, scale: float):
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        B, C, H, W = hm.shape
        hm_reshape = hm.reshape(B, C, H * W)
        idx = torch.argmax(hm_reshape, dim=-1)
        scores = torch.take_along_dim(hm_reshape, torch.unsqueeze(idx, dim=-1), dim=-1).squeeze(-1)
        # hm = hm.numpy()
        # idx = idx.numpy()
        # scores = scores.numpy()
        preds, preds_orig = self._get_preds_fromhm(hm, idx, center, scale)

        return preds, preds_orig, scores


    def _get_preds_fromhm(self, hm: torch.Tensor, idx: torch.Tensor, center: torch.Tensor, scale: float):
        """Obtain (x,y) coordinates given a set of N heatmaps and the
        coresponding locations of the maximums. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        B, C, H, W = hm.shape
        idx += 1
        preds = idx.repeat_interleave(2).reshape(B, C, 2).type(torch.float32)
        preds[:, :, 0] = (preds[:, :, 0] - 1) % W + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / H) + 1

        for i in range(B):
            for j in range(C):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.tensor(
                        [float((hm_[pY, pX + 1] - hm_[pY, pX - 1]).item()),
                        float((hm_[pY + 1, pX] - hm_[pY - 1, pX]).item())])
                    preds[i, j] += torch.sign(diff) * 0.25

        preds -= 0.5

        preds_orig = torch.zeros_like(preds)
        if center is not None and scale is not None:
            for i in range(B):
                for j in range(C):
                    preds_orig[i, j] = self.transform_np(
                        preds[i, j], center, scale, H)

        return preds, preds_orig


face_alignment = FaceAlignment()

with torch.no_grad():
    input = cv2.imread("./noFace.jpg")  # read as uint8
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    # input = io.imread("./test.jpg")
    input = torch.tensor(input, dtype=torch.float32)  # remember to set as float32
    # input = torch.rand(112, 112, 3)
    # output = face_alignment(input)
    # print(f"{len(output)=}, {output[0].shape}")

    # traced_script_module = torch.jit.script(face_alignment, input)
    # # traced_script_module = torch.jit.trace(face_alignment, input)
    # # traced_script_module = torch.jit.trace(face_alignment, input, strict=False)
    # optimized_traced_model = optimize_for_mobile(traced_script_module)
    # optimized_traced_model._save_for_lite_interpreter("./FaceAlignment_scripted.pt")
    
    # # run torchscript
    torchscript = torch.jit.load("./FaceAlignment_scripted.pt")
    # print(torchscript.code)
    output = torchscript(input)
    print(f"{len(output)=}, {output[0].shape}")
    # print(len(output), [o.shape for o in output])
    # print(output)
    

# pip installed model, result should be the same
# face_alignment = SFDDetector(device="cpu")
# output = face_alignment.detect_from_image(input)
