#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : TextDetection.py
# @Author   : jade
# @Date     : 2022/6/15 10:56
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sys
sys.path.append("python_lib")
import logging
import sys
import os
import copy
from copy import deepcopy
import numpy as np
import cv2
import platform
from shapely.geometry import Polygon
import paddle
import paddle.distributed as dist
from paddle import inference
import pyclipper

def get_infer_gpuid():
    sysstr = platform.system()
    if sysstr == "Windows":
        return 0

    if not paddle.fluid.core.is_compiled_with_rocm():
        cmd = "env | grep CUDA_VISIBLE_DEVICES"
    else:
        cmd = "env | grep HIP_VISIBLE_DEVICES"
    env_cuda = os.popen(cmd).readlines()
    if len(env_cuda) == 0:
        return 0
    else:
        gpu_id = env_cuda[0].strip().split("=")[1]
        return int(gpu_id[0])

logger_initialized = {}
def get_logger(name='ppocr', log_file=None, log_level=logging.DEBUG):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified a FileHandler will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt="%Y/%m/%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_file is not None and dist.get_rank() == 0:
        log_file_folder = os.path.split(log_file)[0]
        os.makedirs(log_file_folder, exist_ok=True)
        file_handler = logging.FileHandler(log_file, 'a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if dist.get_rank() == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    logger_initialized[name] = True
    logger.propagate = False
    return logger
logger = get_logger()

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def build_post_process(config, global_config=None):
    support_dict = [
        'DBPostProcess'
    ]
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch

class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]

class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data



class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        print(op_name,param)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

def get_output_tensors(predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return output_tensors

def create_predictor(gpu_mem,model_path, logger):
    model_dir = model_path
    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))
    if not os.path.exists(params_file_path):
        raise ValueError("not find params file path {}".format(
            params_file_path))

    config = inference.Config(model_file_path, params_file_path)

    precision = inference.PrecisionType.Float32
    gpu_id = get_infer_gpuid()
    if gpu_id is None:
        logger.warning(
            "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jetson."
        )
    config.enable_use_gpu(gpu_mem, 0)
    # skip the minmum trt subgraph
    use_dynamic_shape = True
    min_input_shape = {
        "x": [1, 3, 50, 50],
        "conv2d_92.tmp_0": [1, 120, 20, 20],
        "conv2d_91.tmp_0": [1, 24, 10, 10],
        "conv2d_59.tmp_0": [1, 96, 20, 20],
        "nearest_interp_v2_1.tmp_0": [1, 256, 10, 10],
        "nearest_interp_v2_2.tmp_0": [1, 256, 20, 20],
        "conv2d_124.tmp_0": [1, 256, 20, 20],
        "nearest_interp_v2_3.tmp_0": [1, 64, 20, 20],
        "nearest_interp_v2_4.tmp_0": [1, 64, 20, 20],
        "nearest_interp_v2_5.tmp_0": [1, 64, 20, 20],
        "elementwise_add_7": [1, 56, 2, 2],
        "nearest_interp_v2_0.tmp_0": [1, 256, 2, 2]
    }
    max_input_shape = {
        "x": [1, 3, 1536, 1536],
        "conv2d_92.tmp_0": [1, 120, 400, 400],
        "conv2d_91.tmp_0": [1, 24, 200, 200],
        "conv2d_59.tmp_0": [1, 96, 400, 400],
        "nearest_interp_v2_1.tmp_0": [1, 256, 200, 200],
        "conv2d_124.tmp_0": [1, 256, 400, 400],
        "nearest_interp_v2_2.tmp_0": [1, 256, 400, 400],
        "nearest_interp_v2_3.tmp_0": [1, 64, 400, 400],
        "nearest_interp_v2_4.tmp_0": [1, 64, 400, 400],
        "nearest_interp_v2_5.tmp_0": [1, 64, 400, 400],
        "elementwise_add_7": [1, 56, 400, 400],
        "nearest_interp_v2_0.tmp_0": [1, 256, 400, 400]
    }
    opt_input_shape = {
        "x": [1, 3, 640, 640],
        "conv2d_92.tmp_0": [1, 120, 160, 160],
        "conv2d_91.tmp_0": [1, 24, 80, 80],
        "conv2d_59.tmp_0": [1, 96, 160, 160],
        "nearest_interp_v2_1.tmp_0": [1, 256, 80, 80],
        "nearest_interp_v2_2.tmp_0": [1, 256, 160, 160],
        "conv2d_124.tmp_0": [1, 256, 160, 160],
        "nearest_interp_v2_3.tmp_0": [1, 64, 160, 160],
        "nearest_interp_v2_4.tmp_0": [1, 64, 160, 160],
        "nearest_interp_v2_5.tmp_0": [1, 64, 160, 160],
        "elementwise_add_7": [1, 56, 40, 40],
        "nearest_interp_v2_0.tmp_0": [1, 256, 40, 40]
    }
    min_pact_shape = {
        "nearest_interp_v2_26.tmp_0": [1, 256, 20, 20],
        "nearest_interp_v2_27.tmp_0": [1, 64, 20, 20],
        "nearest_interp_v2_28.tmp_0": [1, 64, 20, 20],
        "nearest_interp_v2_29.tmp_0": [1, 64, 20, 20]
    }
    max_pact_shape = {
        "nearest_interp_v2_26.tmp_0": [1, 256, 400, 400],
        "nearest_interp_v2_27.tmp_0": [1, 64, 400, 400],
        "nearest_interp_v2_28.tmp_0": [1, 64, 400, 400],
        "nearest_interp_v2_29.tmp_0": [1, 64, 400, 400]
    }
    opt_pact_shape = {
        "nearest_interp_v2_26.tmp_0": [1, 256, 160, 160],
        "nearest_interp_v2_27.tmp_0": [1, 64, 160, 160],
        "nearest_interp_v2_28.tmp_0": [1, 64, 160, 160],
        "nearest_interp_v2_29.tmp_0": [1, 64, 160, 160]
    }
    min_input_shape.update(min_pact_shape)
    max_input_shape.update(max_pact_shape)
    opt_input_shape.update(opt_pact_shape)
    if use_dynamic_shape:
        config.set_trt_dynamic_shape_info(
            min_input_shape, max_input_shape, opt_input_shape)

    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_tensors = get_output_tensors(predictor)
    return predictor, input_tensor, output_tensors, config


class TextDetector(object):
    def __init__(self, model_path,gpu_mem=500):
        self.model_path = model_path
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = "False"
        postprocess_params["score_mode"] = "fast"
        self.det_resize_for_test_op = DetResizeForTest(**pre_process_list[0]["DetResizeForTest"])
        self.normalize_image_op = NormalizeImage(**pre_process_list[1]["NormalizeImage"])
        self.to_chw_image_op = ToCHWImage()
        self.keep_kesy_op = KeepKeys(**pre_process_list[3]["KeepKeys"])
        self.preprocess_op = [self.det_resize_for_test_op,self.normalize_image_op,self.to_chw_image_op,self.keep_kesy_op]
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = create_predictor(gpu_mem,model_path, logger)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['maps'] = outputs[0]
        # self.predictor.try_shrink_memory()
        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes

if __name__ == '__main__':
    from jade import *
    from opencv_tools import ReadChinesePath
    from opencv_tools.jade_visualize import draw_text_det_res
    text_detector = TextDetector("models/container_number_det/2022-06-12")
    image_root_path = r"/mnt/e/Data/字符检测识别数据集/箱号关键点数据集/2021-12-05/image"
    image_path_list = GetAllImagesPath(image_root_path)
    for image_path in image_path_list:
        image = ReadChinesePath(image_path)
        dt_boxes = text_detector(image)
        image = draw_text_det_res(image,dt_boxes)
        cv2.namedWindow("result",0)
        cv2.imshow("result",image)
        cv2.waitKey(0)
