#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File     : main.py
# @Author   : jade
# @Date     : 2021/5/6 10:05
# @Email    : jadehh@1ive.com
# @Software : Samples
# @Desc     :
import sys
sys.path.append("/mnt/h/Libs/paddle/v2.2")
import paddle
import paddle.incubate.nn.layer
# import cv2
#
# from jade import *
# if __name__ == '__main__':
#     model_dir = "/mnt/h/PycharmProjects/Gitlab/samples/car_plate_ocr/models/qr_detect/2022-02-18"
#     from detector import Detector
#     qrdetector = Detector(
#         model_dir)
#     image_path_list = GetAllImagesPath("/mnt/c/Users/Administrator/Desktop/2022-02-24/2022-02-24")
#     wrong_count = 0
#
#     for image_path in image_path_list:
#         image = cv2.imread(image_path)
#         results = qrdetector.predict(image)
#         print(results)