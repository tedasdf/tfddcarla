import numpy as np

def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def batched_nms(boxes: Tensor,
                scores: Tensor,
                idxs: Tensor,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (
                max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]

        # This assumes `dets` has arbitrary dimensions where
        # the last dimension is score.
        # Currently it supports bounding boxes [x1, y1, x2, y2, score] or
        # rotated boxes [cx, cy, w, h, angle_radian, score].

        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop('max_num', -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        # Some type of nms would reweight the score, such as SoftNMS
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


    def force_fp32(apply_to: Optional[Iterable] = None,
               out_fp16: bool = False) -> Callable:
        """Decorator to convert input arguments to fp32 in force.

        This decorator is useful when you write custom modules and want to support
        mixed precision training. If there are some inputs that must be processed
        in fp32 mode, then this decorator can handle it. If inputs arguments are
        fp16 tensors, they will be converted to fp32 automatically. Arguments other
        than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
        torch.cuda.amp is used as the backend, otherwise, original mmcv
        implementation will be adopted.

        Args:
            apply_to (Iterable, optional): The argument names to be converted.
                `None` indicates all arguments.
            out_fp16 (bool): Whether to convert the output back to fp16.

        Example:

            >>> import torch.nn as nn
            >>> class MyModule1(nn.Module):
            >>>
            >>>     # Convert x and y to fp32
            >>>     @force_fp32()
            >>>     def loss(self, x, y):
            >>>         pass

            >>> import torch.nn as nn
            >>> class MyModule2(nn.Module):
            >>>
            >>>     # convert pred to fp32
            >>>     @force_fp32(apply_to=('pred', ))
            >>>     def post_process(self, pred, others):
            >>>         pass
        """

        def force_fp32_wrapper(old_func):

            @functools.wraps(old_func)
            def new_func(*args, **kwargs):
                # check if the module has set the attribute `fp16_enabled`, if not,
                # just fallback to the original method.
                if not isinstance(args[0], torch.nn.Module):
                    raise TypeError('@force_fp32 can only be used to decorate the '
                                    'method of nn.Module')
                if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                    return old_func(*args, **kwargs)
                # get the arg spec of the decorated method
                args_info = getfullargspec(old_func)
                # get the argument names to be casted
                args_to_cast = args_info.args if apply_to is None else apply_to
                # convert the args that need to be processed
                new_args = []
                if args:
                    arg_names = args_info.args[:len(args)]
                    for i, arg_name in enumerate(arg_names):
                        if arg_name in args_to_cast:
                            new_args.append(
                                cast_tensor_type(args[i], torch.half, torch.float))
                        else:
                            new_args.append(args[i])
                # convert the kwargs that need to be processed
                new_kwargs = dict()
                if kwargs:
                    for arg_name, arg_value in kwargs.items():
                        if arg_name in args_to_cast:
                            new_kwargs[arg_name] = cast_tensor_type(
                                arg_value, torch.half, torch.float)
                        else:
                            new_kwargs[arg_name] = arg_value
                # apply converted arguments to the decorated method
                if (TORCH_VERSION != 'parrots' and
                        digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
                    with autocast(enabled=False):
                        output = old_func(*new_args, **new_kwargs)
                else:
                    output = old_func(*new_args, **new_kwargs)
                # cast the results back to fp32 if necessary
                if out_fp16:
                    output = cast_tensor_type(output, torch.float, torch.half)
                return output

            return new_func

        return force_fp32_wrapper

