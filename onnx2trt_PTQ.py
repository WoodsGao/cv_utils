#!/usr/bin/python3
import argparse
import os
import os.path as osp

import cv2
import numpy as np
from tqdm import tqdm

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger()


def build_blob(size,
               img,
               rect=False,
               mean=[123.675, 116.28, 103.53],
               std=[58.395, 57.12, 57.375]):
    if rect:
        ratio = min(size[1] / img.shape[0], size[0] / img.shape[1])
        img = cv2.resize(
            img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    else:
        img = cv2.resize(img, (size[0], size[1]))
    img = img.astype(np.float32)[..., ::-1]
    pad = np.int32([size[1] - img.shape[0], size[0] - img.shape[1]])
    lpad = pad // 2
    rpad = pad - lpad
    img = img.transpose(2, 0, 1)
    img -= np.float32(mean).reshape(3, 1, 1)
    img /= np.float32(std).reshape(3, 1, 1)
    pad = np.concatenate([lpad, rpad])
    return img, pad


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator):
    def __init__(self,
                 img_dir,
                 img_size,
                 cache_file='image.cache',
                 batch_size=32,
                 rect=False):
        super(ImageEntropyCalibrator, self).__init__()
        self.cache_file = cache_file
        self.data = []
        for dirpath, dirnames, filenames in os.walk(img_dir):
            for filename in filenames:
                if osp.splitext(filename)[-1] in ['.png', '.jpg', '.jpeg', '.tiff']:
                    self.data.append(osp.join(dirpath, filename))
        self.img_size = img_size
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * 8 *
                                           img_size[0] * img_size[1])
        self.pbar = iter(tqdm(range(len(self.data) // self.batch_size)))
        self.rect = rect

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None
        batch = []
        for data in self.data[self.current_index:self.current_index +
                              self.batch_size]:
            img = cv2.imread(data)
            img, pad = build_blob(self.img_size, img, self.rect)
            batch.append(img)
        batch = np.ascontiguousarray(np.stack(batch, 0))
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        next(self.pbar)
        print(self.current_index)
        return [self.device_input]

    def read_calibration_cache(self):
        if osp.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def onnx2trt_PTQ(onnx_model, output_path, img_dir, img_size, batch_size):
    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size
        builder.int8_mode = True
        builder.int8_calibrator = ImageEntropyCalibrator(img_dir, img_size)
        # Build engine and do int8 calibration.
        network_creation_flag = 0
        # network_creation_flag |= 1 << int(
        #     trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        network_creation_flag |= 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_creation_flag)
        parser = trt.OnnxParser(network, TRT_LOGGER)
        # Parse model file
        with open(onnx_model, 'rb') as model:
            parser.parse(model.read())
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            # last_layer = network.get_layer(network.num_layers - 1)
            # network.mark_output(last_layer.get_output(0))
        engine = builder.build_cuda_engine(network)
        with open(output_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', type=str)
    parser.add_argument('output', type=str)
    parser.add_argument('img_dir', type=str)
    parser.add_argument('-s', '--img_size', type=int, nargs=2, default=[224, 224])
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    opt = parser.parse_args()
    print(opt)
    onnx2trt_PTQ(opt.onnx, opt.output, opt.img_dir, opt.img_size, opt.batch_size)
