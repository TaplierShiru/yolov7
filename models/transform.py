# Copyright (c) 2020, yolort team. All rights reserved.
# Taken from:
#            https://github.com/zhiqwang/yolov5-rt-stack/blob/main/yolort/models/transform.py

import math

from typing import cast, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import box_convert


class NestedTensor(NamedTuple):
    """
    Structure that holds images as a single tensor (with equal size as big tensor).
    This works by padding the images to the same size, and storing in a
    field the original sizes of each image.
    """

    tensor: Tensor
    image_origin_size: Tuple[int, int]


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[-2:]

@torch.jit.unused
def _get_batch_size_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators

    return operators.shape_as_tensor(image)[0]


@torch.jit.unused
def _tracing_int_onnx(v: Tensor) -> int:
    """
    ONNX requires a tensor type for Tensor.item() in tracing mode, so we cast
    its type to int here.
    """

    return cast(int, v)


@torch.jit.unused
def _tracing_float_onnx(v: Tensor) -> float:
    """
    ONNX requires a tensor type for Tensor.item() in tracing mode, so we cast
    its type to float here.
    """
    return cast(float, v)



class YOLOTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a YOLO model. It plays
    the same role of `LetterBox` in YOLOv5. YOLOv5 use (0, 1, RGB) as the default mean, std and
    color channel mode. We do not normalize below, the inputs need to be scaled down to float
    in [0-1] from uint8 in [0-255] and transpose the color channel to RGB before being fed to
    this transformation. We use the `torch.nn.functional.interpolate` and `torch.nn.functional.pad`
    ops to implement the `LetterBox` to make it jit traceable and scriptable.
    The transformations it perform are:
        - resizing input / target that maintains the aspect ratio to match `min_size / max_size`
        - letterboxing padding the input / target that can be divided by `size_divisible`
    It returns a `NestedTensor` for the inputs, and a List[Dict[Tensor]] for the targets.
    Args:
        min_size (int): minimum size of the image to be rescaled
        max_size (int): maximum size of the image to be rescaled
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(
            self,
            min_size: int,
            max_size: int,
            *,
            size_divisible: int = 32,
            fixed_shape: Optional[Tuple[int, int]] = None,
            fill_color: int = 114,
            is_trt_mode=False):

        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.size_divisible = size_divisible
        self.fixed_shape = fixed_shape
        self.fill_color = fill_color / 255
        self.is_trt_mode = is_trt_mode
        
    def forward(
        self,
        images: Tensor
    ) -> Tensor:
        """
        Perform letterboxing transformation.
        Args:
            images (Tensor): Images to be processed. In general the type of images is a list
                of 3-dim `Tensor`, except for the dataloader in training and evaluation, the `images`
                will be a 4-dim `Tensor` in that case. Check out the belows link for more details:
                https://github.com/zhiqwang/yolov5-rt-stack/pull/308#pullrequestreview-878689796
        Returns:
            result (NestedTensor): preprocessed image for the yolo model.
            
        """
        print('!!!!!!!!!!!!!!!!!! preprocess...')
        images = self.resize(images)
        images = self.batch_images(images)

        print('!!!!!!!!!!!!!!!!!! preprocess END...')

        return images

    def resize(
        self,
        images: Tensor
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        self_min_size = float(self.min_size)
        self_max_size = float(self.max_size)
        
        if torchvision._is_tracing():
            im_shape = _get_shape_onnx(images)
        else:
            im_shape = torch.tensor(images.shape[-2:])

        scale_factor: Optional[float] = None

        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _tracing_float_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

        images = F.interpolate(
            images,
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=recompute_scale_factor,
            align_corners=False,
        )
        return images

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: Tensor) -> Tensor:
        img_h, img_w = _get_shape_onnx(images)
        
        if self.fixed_shape is not None:
            max_size = torch.tensor(self.fixed_shape)
        else:
            # We assume that images have equal size
            max_size = [img_h, img_w]
            stride = self.size_divisible
            max_size[0] = (torch.ceil((max_size[0].to(torch.float32)) / stride) * stride).to(torch.int32)
            max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int32)

        # work around for
        # batched_imgs[i, :channel, dh : dh + img_h, dw : dw + img_w].copy_(img)
        # which is not yet supported in onnx

        dh = (max_size[1] - img_w) / 2
        dw = (max_size[0] - img_h) / 2

        padding = (
            _tracing_int_onnx(torch.round(dh - 0.1).to(dtype=torch.int32)),
            _tracing_int_onnx(torch.round(dh + 0.1).to(dtype=torch.int32)),
            _tracing_int_onnx(torch.round(dw - 0.1).to(dtype=torch.int32)),
            _tracing_int_onnx(torch.round(dw + 0.1).to(dtype=torch.int32)),
        )
        #padded_img = F.pad(img, padding, value=self.fill_color)
        # Change F.pad to cat with zeros due to error while convert to TensorRT:
        #   https://github.com/NVIDIA/TensorRT/issues/2487
        batch_size = _get_batch_size_onnx(images)
        im_padded = torch.cat((torch.zeros((batch_size, 3, img_h, padding[0])).to(images.device), images), dim=3)
        im_padded = torch.cat((im_padded, torch.zeros((batch_size, 3, img_h, padding[1])).to(images.device)), dim=3)

        im_padded = torch.cat((im_padded, torch.zeros((batch_size, 3, padding[2], img_w + padding[0] + padding[1])).to(images.device)), dim=2)
        im_padded = torch.cat((torch.zeros((batch_size, 3, padding[3], img_w + padding[0] + padding[1])).to(images.device), im_padded), dim=2)

        return im_padded

    def batch_images(self, images: Tensor) -> Tensor:
        """
        Nest a list of tensors. It plays the same role of the letterbox function.
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images)
        bs, channel, img_h, img_w = images.shape
        # TODO: Rewrite code below as in onnx for full batched tensor
        if self.fixed_shape is not None:
            max_size = [3, *(self.fixed_shape)]
        else:
            stride = float(self.size_divisible)
            max_size = [channel, img_h, img_w]
            max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
            max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [bs] + max_size
        batched_imgs = images[0].new_full(batch_shape, self.fill_color)
        # divide padding into 2 sides below
        dh = (max_size[1] - img_h) / 2
        dh = int(round(dh - 0.1))

        dw = (max_size[2] - img_w) / 2
        dw = int(round(dw - 0.1))

        batched_imgs[:, :channel, dh : dh + img_h, dw : dw + img_w].copy_(images)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: Tensor,
        original_image_size: Tuple[int, int],
    ) -> List[Dict[str, Tensor]]:
        """
        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = scale_coords(boxes, image_shapes, o_im_s)
            result[i]["boxes"] = boxes
        """
        print(f'!!!!!!!!!!!!!!!!!! postprocess is_trt_mode={self.is_trt_mode} on result({len(result)})={result}...')
        # TODO: If not is_trt_mode - only batch equal to 1 is possible
        if self.is_trt_mode:
            boxes = result[1]
        else:
            boxes = result[:, :, :4] # TODO: Its should be 1:5 ? Right???
        boxes = scale_coords_batch(boxes, image_shapes, original_image_size)

        if self.is_trt_mode:
            result = (result[0], boxes, *result[2:])
        else:
            result[:, :, :4] = boxes  # TODO: Its should be 1:5 ? Right???
        print('!!!!!!!!!!!!!!!!!! postprocess END...')
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        _indent = "\n    "
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size})"
        format_string += "\n)"
        return format_string



def _resize_image_and_masks(
    image: Tensor,
    self_min_size: float,
    self_max_size: float
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    scale_factor: Optional[float] = None

    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale = torch.min(self_min_size / min_size, self_max_size / max_size)

    if torchvision._is_tracing():
        scale_factor = _tracing_float_onnx(scale)
    else:
        scale_factor = scale.item()
    recompute_scale_factor = True

    image = F.interpolate(
        image[None],
        size=None,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    return image

    
class YOLOTransform_OLD(nn.Module):
    """
    Performs input / target transformation before feeding the data to a YOLO model. It plays
    the same role of `LetterBox` in YOLOv5. YOLOv5 use (0, 1, RGB) as the default mean, std and
    color channel mode. We do not normalize below, the inputs need to be scaled down to float
    in [0-1] from uint8 in [0-255] and transpose the color channel to RGB before being fed to
    this transformation. We use the `torch.nn.functional.interpolate` and `torch.nn.functional.pad`
    ops to implement the `LetterBox` to make it jit traceable and scriptable.
    The transformations it perform are:
        - resizing input / target that maintains the aspect ratio to match `min_size / max_size`
        - letterboxing padding the input / target that can be divided by `size_divisible`
    It returns a `NestedTensor` for the inputs, and a List[Dict[Tensor]] for the targets.
    Args:
        min_size (int): minimum size of the image to be rescaled
        max_size (int): maximum size of the image to be rescaled
        size_divisible (int): stride of the models. Default: 32
        fixed_shape (Tuple[int, int], optional): Padding mode for letterboxing. If set to `True`,
            the image will be padded to shape `fixed_shape` if specified. Instead the image will
            be padded to a minimum rectangle to match `min_size / max_size` and each of its edges
            is divisible by `size_divisible` if it is not specified. Default: None
        fill_color (int): fill value for padding. Default: 114
    """

    def __init__(
            self,
            min_size: int,
            max_size: int,
            *,
            size_divisible: int = 32,
            fixed_shape: Optional[Tuple[int, int]] = None,
            fill_color: int = 114,
            is_trt_mode=False):

        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.size_divisible = size_divisible
        self.fixed_shape = fixed_shape
        self.fill_color = fill_color / 255
        self.is_trt_mode = is_trt_mode
        
    def forward(
        self,
        images: List[Tensor]
    ) -> Tuple[NestedTensor, Optional[Tensor]]:
        """
        Perform letterboxing transformation.
        Args:
            images (List[Tensor]): Images to be processed. In general the type of images is a list
                of 3-dim `Tensor`, except for the dataloader in training and evaluation, the `images`
                will be a 4-dim `Tensor` in that case. Check out the belows link for more details:
                https://github.com/zhiqwang/yolov5-rt-stack/pull/308#pullrequestreview-878689796
            targets (List[Dict[Tensor]], optional): ground-truth boxes present in the image for
                training. Default: None
        Returns:
            result (List[BoxList] or Dict[Tensor]): the output from the model.
                During training, it returns a Dict[Tensor] which contains the losses.
                During testing, it returns List[BoxList] contains additional fields
                like `scores`, `labels` and `boxes`.
        """
        print('!!!!!!!!!!!!!!!!!! preprocess...')
        # TODO: Do not use images as List, use it as big tensor
        #       Need to rewrite prerocess(resize) and postprocess to batch version
        assert 1 ==3, 'todo above'
        device = images[0].device
        images = [img for img in images]

        for i in range(len(images)):
            image = images[i]

            if image.dim() != 3:
                raise ValueError(
                    "images is expected to be a list of 3d tensors of "
                    f"shape [C, H, W], but got '{image.shape}'."
                )

            image = self.resize(image)
            images[i] = image

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = NestedTensor(tensors=images, image_sizes=image_sizes_list)

        print('!!!!!!!!!!!!!!!!!! preprocess END...')

        return image_list

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]

    def resize(
        self,
        image: Tensor
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        # TODO: Make all needed code here and resize full batch
        h, w = image.shape[-2:]
        min_size = float(self.min_size)
        max_size = float(self.max_size)
        image = _resize_image_and_masks(image, min_size, max_size)

        return image

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: List[Tensor]) -> Tensor:
        if self.fixed_shape is not None:
            max_size = torch.tensor(self.fixed_shape)
        else:
            max_size = []
            for i in range(1, images[0].dim()):
                max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32))
                max_size.append(max_size_i.to(torch.int32))
            stride = self.size_divisible
            max_size[0] = (torch.ceil((max_size[0].to(torch.float32)) / stride) * stride).to(torch.int32)
            max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int32)

        # work around for
        # batched_imgs[i, :channel, dh : dh + img_h, dw : dw + img_w].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            img_h, img_w = _get_shape_onnx(img)

            dh = (max_size[1] - img_w) / 2
            dw = (max_size[0] - img_h) / 2

            padding = (
                _tracing_int_onnx(torch.round(dh - 0.1).to(dtype=torch.int32)),
                _tracing_int_onnx(torch.round(dh + 0.1).to(dtype=torch.int32)),
                _tracing_int_onnx(torch.round(dw - 0.1).to(dtype=torch.int32)),
                _tracing_int_onnx(torch.round(dw + 0.1).to(dtype=torch.int32)),
            )
            #padded_img = F.pad(img, padding, value=self.fill_color)
            # Change F.pad to cat with zeros due to error while convert to TensorRT:
            #   https://github.com/NVIDIA/TensorRT/issues/2487
            im_padded = torch.cat((torch.zeros((3, img_h, padding[0])).to(img.device), img), dim=2)
            im_padded = torch.cat((im_padded, torch.zeros((3, img_h, padding[1])).to(img.device)), dim=2)

            im_padded = torch.cat((im_padded, torch.zeros((3, padding[2], img_w + padding[0] + padding[1])).to(img.device)), dim=1)
            im_padded = torch.cat((torch.zeros((3, padding[3], img_w + padding[0] + padding[1])).to(img.device), im_padded), dim=1)
            
            padded_imgs.append(im_padded)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor]) -> Tensor:
        """
        Nest a list of tensors. It plays the same role of the letterbox function.
        """

        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images)

        if self.fixed_shape is not None:
            max_size = [3, *(self.fixed_shape)]
        else:
            stride = float(self.size_divisible)
            max_size = self.max_by_axis([list(img.shape) for img in images])
            max_size = list(max_size)
            max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
            max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, self.fill_color)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            channel, img_h, img_w = img.shape
            # divide padding into 2 sides below
            dh = (max_size[1] - img_h) / 2
            dh = int(round(dh - 0.1))

            dw = (max_size[2] - img_w) / 2
            dw = int(round(dw - 0.1))

            batched_imgs[i, :channel, dh : dh + img_h, dw : dw + img_w].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: Tensor,
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        """
        for i, (pred, o_im_s) in enumerate(zip(result, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = scale_coords(boxes, image_shapes, o_im_s)
            result[i]["boxes"] = boxes
        """
        print(f'!!!!!!!!!!!!!!!!!! postprocess is_trt_mode={self.is_trt_mode} on result({len(result)})={result}...')
        # TODO: If not is_trt_mode - only batch equal to 1 is possible
        # TODO: Remove for - scale coordinates as big vector
        for i, o_im_s in enumerate(original_image_sizes):
            if self.is_trt_mode:
                boxes = result[1][i]
            else:
                boxes = result[i, :, :4]
            boxes = scale_coords(boxes, image_shapes, o_im_s)
            
            if self.is_trt_mode:
                result[1][i] = boxes
            else:
                result[i, :, :4] = boxes
        print('!!!!!!!!!!!!!!!!!! postprocess END...')
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        _indent = "\n    "
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size})"
        format_string += "\n)"
        return format_string


def scale_coords(boxes: Tensor, new_size: Tensor, original_size: Tuple[int, int]) -> Tensor:
    """
    Rescale boxes (xyxy) from new_size to original_size
    """
    gain = torch.min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    pad = (new_size[1] - original_size[1] * gain) / 2, (new_size[0] - original_size[0] * gain) / 2
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = (xmin - pad[0]) / gain
    xmax = (xmax - pad[0]) / gain
    ymin = (ymin - pad[1]) / gain
    ymax = (ymax - pad[1]) / gain

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def scale_coords_batch(boxes_batch: Tensor, new_size: Tensor, original_size: Tuple[int, int]) -> Tensor:
    """
    Rescale boxes (xyxy) from new_size to original_size
    """
    gain = torch.min(new_size[0] / original_size[0], new_size[1] / original_size[1])
    pad = (new_size[1] - original_size[1] * gain) / 2, (new_size[0] - original_size[0] * gain) / 2
    # (N, count, 4) -> Four tensors with shape (N, 4)
    xmin, ymin, xmax, ymax = boxes_batch.unbind(2)

    xmin = (xmin - pad[0]) / gain
    xmax = (xmax - pad[0]) / gain
    ymin = (ymin - pad[1]) / gain
    ymax = (ymax - pad[1]) / gain

    return torch.stack((xmin, ymin, xmax, ymax), dim=2)


# TODO: Seems like not used function - delete?
#  In original repo this func was used for target data, but here its not needed - DELETE!
def normalize_boxes(boxes: Tensor, original_size: List[int]) -> Tensor:
    height = torch.tensor(original_size[0], dtype=torch.float32, device=boxes.device)
    width = torch.tensor(original_size[1], dtype=torch.float32, device=boxes.device)
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin / width
    xmax = xmax / width
    ymin = ymin / height
    ymax = ymax / height
    boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
    # Convert xyxy to cxcywh
    return box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
