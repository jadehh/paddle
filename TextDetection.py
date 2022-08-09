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
import logging as logger
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from shapely.geometry import Polygon
import pyclipper
import cv2
import os
import sys
import numpy as np
import operator
class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self, params):
        self.thresh = params['thresh']
        self.box_thresh = params['box_thresh']
        self.max_candidates = params['max_candidates']
        self.unclip_ratio = params['unclip_ratio']
        self.min_size = 3

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
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

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

    def __call__(self, outs_dict, ratio_list):
        pred = outs_dict['maps']

        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = pred.shape[-2:]
            tmp_boxes, tmp_scores = self.boxes_from_bitmap(
                pred[batch_index], segmentation[batch_index], width, height)

            boxes = []
            for k in range(len(tmp_boxes)):
                if tmp_scores[k] > self.box_thresh:
                    boxes.append(tmp_boxes[k])
            if len(boxes) > 0:
                boxes = np.array(boxes)

                ratio_h, ratio_w = ratio_list[batch_index]
                boxes[:, :, 0] = boxes[:, :, 0] / ratio_w
                boxes[:, :, 1] = boxes[:, :, 1] / ratio_h

            boxes_batch.append(boxes)
        return boxes_batch


class DBProcessTest(object):
    """
    DB pre-process for Test mode
    """

    def __init__(self, params):
        super(DBProcessTest, self).__init__()
        self.resize_type = 0
        if 'test_image_shape' in params:
            self.image_shape = params['test_image_shape']
            # print(self.image_shape)
            self.resize_type = 1
        if 'max_side_len' in params:
            self.max_side_len = params['max_side_len']
        else:
            self.max_side_len = 2400

    def resize_image_type0(self, im):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        max_side_len = self.max_side_len
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            if resize_h > resize_w:
                ratio = float(max_side_len) / resize_h
            else:
                ratio = float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)
        if resize_h % 32 == 0:
            resize_h = resize_h
        elif resize_h // 32 <= 1:
            resize_h = 32
        else:
            resize_h = (resize_h // 32 - 1) * 32
        if resize_w % 32 == 0:
            resize_w = resize_w
        elif resize_w // 32 <= 1:
            resize_w = 32
        else:
            resize_w = (resize_w // 32 - 1) * 32
        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            im = cv2.resize(im, (int(resize_w), int(resize_h)))
        except:
            print(im.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image_type1(self, im):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = im.shape[:2]  # (h, w, c)
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        return im, (ratio_h, ratio_w)

    def normalize(self, im):
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        im = im.astype(np.float32, copy=False)
        im = im / 255
        im[:, :, 0] -= img_mean[0]
        im[:, :, 1] -= img_mean[1]
        im[:, :, 2] -= img_mean[2]
        im[:, :, 0] /= img_std[0]
        im[:, :, 1] /= img_std[1]
        im[:, :, 2] /= img_std[2]
        channel_swap = (2, 0, 1)
        im = im.transpose(channel_swap)
        return im

    def __call__(self, im):
        if self.resize_type == 0:
            im, (ratio_h, ratio_w) = self.resize_image_type0(im)
        else:
            im, (ratio_h, ratio_w) = self.resize_image_type1(im)
        im = self.normalize(im)
        im = im[np.newaxis, :]
        return [im, (ratio_h, ratio_w)]


class TextDetector():
    def __init__(self, model_path,gpu_memory_fraction=1000):
        self.model_path = model_path
        self.gpu_mem = gpu_memory_fraction
        if self.gpu_mem > 0:
            self.use_gpu = True
        else:
            self.use_gpu = False
        preprocess_params = {'max_side_len': 960}
        postprocess_params = {}
        self.preprocess_op = DBProcessTest(preprocess_params)
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.5
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 2.0
        self.postprocess_op = DBPostProcess(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors = self.create_predictor()
        self.out_wh_h = (400, 80)
        self.out_wh_v = (80, 400)
        logger.info("读取文本检测模型成功")

    def create_predictor(self):
        model_file_path = self.model_path + "/model"
        params_file_path = self.model_path + "/params"
        if not os.path.exists(model_file_path):
            logger.error("没能找到模型路径地址 {}".format(model_file_path))
            sys.exit(0)
        if not os.path.exists(params_file_path):
            logger.error("没能找到模型参数地址 {}".format(params_file_path))
        config = AnalysisConfig(model_file_path, params_file_path)
        config.disable_glog_info()
        if self.use_gpu:
            config.enable_use_gpu(self.gpu_mem, 0)
            config.enable_memory_optim()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        predictor = create_paddle_predictor(config)
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_tensor(input_names[0])
        output_names = predictor.get_output_names()
        output_tensors = []
        for output_name in output_names:
            output_tensor = predictor.get_output_tensor(output_name)
            output_tensors.append(output_tensor)
        return predictor, input_tensor, output_tensors

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(4):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect


    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img = img/255.0
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                                 np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                                  np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0],
                              [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                      borderValue=.0)

        # dst_img = cv2.cvtColor(dst_img,cv2.COLOR_RGB2BGR)
        return dst_img*255.0


    def  expand_image(self,image):
        img_wh = np.array([image.shape[1],image.shape[0]])
        n_wh = img_wh
        n_pimg = image
        # one side scale to max
        if n_wh[0] < n_wh[1]:
            adj_radio = np.array(self.out_wh_v, dtype=float) / n_wh
        else:
            adj_radio = np.array(self.out_wh_h, dtype=float) / n_wh

        if (adj_radio > 1.0).all():
            adj_radio = min(adj_radio)
            n_wh = (adj_radio * n_wh).astype(int)

            n_pimg = cv2.resize(n_pimg, tuple(n_wh))

        if n_wh[0] > n_wh[1] and n_wh[0] / n_wh[1] < 4.0:
            n_wh[0] = n_wh[0] * 1.7
            n_wh = n_wh.astype(int)
            n_pimg = cv2.resize(n_pimg, tuple(n_wh))
        #

        if n_wh[0] < n_wh[1]:
            expand_wh = ((self.out_wh_v - n_wh) * .5).astype(int)
            expand_wh = np.floor(np.maximum(expand_wh, 0.)).astype(int)
            if expand_wh.any() > 0:
                n_pimg = cv2.copyMakeBorder(n_pimg, expand_wh[1], expand_wh[1], expand_wh[0], expand_wh[0],
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if (n_wh % 2).any() != .0 or operator.eq(tuple(n_wh), self.out_wh_v) != 0:
                n_pimg = cv2.resize(n_pimg, self.out_wh_v)
        else:
            expand_wh = ((self.out_wh_h - n_wh) * .5).astype(int)
            expand_wh = np.floor(np.maximum(expand_wh, 0.)).astype(int)
            if expand_wh.any() > 0 or operator.eq(tuple(n_wh), self.out_wh_h) != 0:
                n_pimg = cv2.copyMakeBorder(n_pimg, expand_wh[1], expand_wh[1], expand_wh[0], expand_wh[0],
                                            cv2.BORDER_CONSTANT, value=[0, 0, 0])

            if (n_wh % 2).any() != .0:
                n_pimg = cv2.resize(n_pimg, self.out_wh_h)
        return n_pimg

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 10 or rect_height <= 10:
                continue
            dt_boxes_new.append(box)

        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict(self, img):
        ori_im = img.copy()
        im, ratio_list = self.preprocess_op(img)
        if im is None:
            return None, 0
        im = im.copy()
        self.input_tensor.copy_from_cpu(im)
        self.predictor.zero_copy_run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
        outs_dict = {}
        outs_dict['maps'] = outputs[0]
        dt_boxes_list = self.postprocess_op(outs_dict, [ratio_list])
        dt_boxes = dt_boxes_list[0]
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes

if __name__ == '__main__':
    from jade import *
    from opencv_tools import ReadChinesePath
    from opencv_tools.jade_visualize import draw_text_det_res
    text_detector = TextDetector("models/TextDetModels/2022-05-27")
    image_root_path = r"/mnt/e/Data/字符检测识别数据集/箱号关键点数据集/2021-12-05/image"
    image_path_list = GetAllImagesPath(image_root_path)
    for image_path in image_path_list:
        image = ReadChinesePath(image_path)
        dt_boxes = text_detector.predict(image)
        image = draw_text_det_res(image,dt_boxes)
        cv2.namedWindow("result",0)
        cv2.imshow("result",image)
        cv2.waitKey(0)
