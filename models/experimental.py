import numpy as np
import random
import torch
import torch.nn as nn
import torchvision

from models.common import Conv, DWConv
from utils.google_utils import attempt_download

from .transform import YOLOTransform, _get_shape_onnx

from typing import Tuple, List, Optional, Dict, Any, Callable


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output





class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None, n_classes=80):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, n_classes=80):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


class YOLOv7Full(nn.Module):
    """
    Wrapping the pre-processing (`LetterBox`) into the YOLO models.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes but they will be resized
    to a fixed size that maintains the aspect ratio before passing it to the backbone.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows, where ``N`` is the number of detections:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each detection
        - scores (Tensor[N]): the scores for each detection
    Example:
        Demo pipeline for YOLOv5 Inference.
        .. code-block:: python
            from yolort.models import YOLOv5
            # Load the yolov5s version 6.0 models
            arch = 'yolov5_darknet_pan_s_r60'
            model = YOLOv5(arch=arch, pretrained=True, score_thresh=0.35)
            model = model.eval()
            # Perform inference on an image file
            predictions = model.predict('bus.jpg')
            # Perform inference on a list of image files
            predictions2 = model.predict(['bus.jpg', 'zidane.jpg'])
        We also support loading the custom checkpoints trained from ultralytics/yolov5
        .. code-block:: python
            from yolort.models import YOLOv5
            # Your trained checkpoint from ultralytics
            checkpoint_path = 'yolov5n.pt'
            model = YOLOv5.load_from_yolov5(checkpoint_path, score_thresh=0.35)
            model = model.eval()
            # Perform inference on an image file
            predictions = model.predict('bus.jpg')
    Args:
        model (nn.Module): YOLO model.
        size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
            Default: (640, 640)
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(
            self,
            model: nn.Module,
            size: Tuple[int, int] = (640, 640),
            size_divisible: int = 32,
            fixed_shape: Optional[Tuple[int, int]] = None,
            fill_color: int = 114,
            **kwargs: Any):
        super().__init__()
        self.model = model
        
        self.is_trt_mode = False
        if isinstance(self.model, End2End):
            self.is_trt_mode = isinstance(self.model.end2end, ONNX_TRT) 
        

        self.transform = YOLOTransform(
            size[0],
            size[1],
            size_divisible=size_divisible,
            fixed_shape=fixed_shape,
            fill_color=fill_color,
            is_trt_mode=self.is_trt_mode,
        )
        # self.transform.to(self.model.device)

        # used only on torchscript mode
        self._has_warned = False

    def forward(
        self,
        inputs: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        # get the original image sizes
        if torchvision._is_tracing():
            original_image_size: Tuple[int, int] = _get_shape_onnx(inputs)
        else:
            original_image_size: Tuple[int, int] = torch.tensor(inputs.shape[-2:])

        # Transform the input
        transformed_inputs = self.transform(inputs)
        print(f'!!!!!!!!!!!!!!!!! inputs={transformed_inputs.shape}')
        # Compute the detections
        result = self.model(transformed_inputs)
        print(f'!!!!!!!!!!!!!!!!!!!!!! result from model ({len(result)})= {result}')

        if torchvision._is_tracing():
            im_shape = _get_shape_onnx(transformed_inputs)
        else:
            im_shape = torch.tensor(transformed_inputs.shape[-2:])

        detections = self.transform.postprocess(result, im_shape, original_image_size)
        
        # TODO: Remove if-else
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("YOLOv5 always returns a (Losses, Detections) tuple in scripting.")
                self._has_warned = True
            return detections
        else:
            return detections

    @torch.no_grad()
    def predict(self, x: Any, image_loader: Optional[Callable] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Predict function for raw data or processed data
        Args:
            x: Input to predict. Can be raw data or processed data.
            image_loader: Utility function to convert raw data to Tensor.
        Returns:
            The post-processed model predictions.
        """
        image_loader = image_loader or self.default_loader
        images = self.collate_images(x, image_loader)
        return self.forward(images)

    def default_loader(self, img_path: str) -> torch.Tensor:
        """
        Default loader of read a image path.
        Args:
            img_path (str): a image path
        Returns:
            Tensor, processed tensor for prediction.
        """
        return read_image(img_path, mode=ImageReadMode.RGB) / 255.0

    def collate_images(self, samples: Any, image_loader: Callable) -> List[torch.Tensor]:
        """
        Prepare source samples for inference.
        Args:
            samples (Any): samples source, support the following various types:
                - str or List[str]: a image path or list of image paths.
                - Tensor or List[Tensor]: a tensor or list of tensors.
        Returns:
            List[Tensor], The processed image samples.
        """
        p = next(self.parameters())  # for device and type
        if isinstance(samples, torch.Tensor):
            return [samples.to(p.device).type_as(p)]

        if contains_any_tensor(samples):
            return [sample.to(p.device).type_as(p) for sample in samples]

        if isinstance(samples, str):
            samples = [samples]

        if isinstance(samples, (list, tuple)) and all(isinstance(p, str) for p in samples):
            outputs = []
            for sample in samples:
                output = image_loader(sample).to(p.device).type_as(p)
                outputs.append(output)
            return outputs

        raise NotImplementedError(
            f"The type of the sample is {type(samples)}, we currently don't support it now, the "
            "samples should be either a tensor, list of tensors, a image path or list of image paths."
        )

    @classmethod
    def load_from_yolov5(
        cls,
        checkpoint_path: str,
        *,
        size: Tuple[int, int] = (640, 640),
        size_divisible: int = 32,
        fixed_shape: Optional[Tuple[int, int]] = None,
        fill_color: int = 114,
        **kwargs: Any,
    ):
        """
        Load custom checkpoints trained from YOLOv5.
        Args:
            checkpoint_path (str): Path of the YOLOv5 checkpoint model.
            size: (Tuple[int, int]): the minimum and maximum size of the image to be rescaled.
                Default: (640, 640)
            size_divisible (int): stride of the models. Default: 32
            fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
                the image will be padded to shape `fixed_shape` if specified. Instead the image will
                be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
                is divisible by `size_divisible` if it is not specified. Default: None
            fill_color (int): fill value for padding. Default: 114
        """
        model = YOLO.load_from_yolov5(checkpoint_path, **kwargs)
        yolov5 = cls(
            model=model,
            size=size,
            size_divisible=size_divisible,
            fixed_shape=fixed_shape,
            fill_color=fill_color,
        )
        return yolov5



def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


