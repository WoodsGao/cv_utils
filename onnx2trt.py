#!/usr/bin/python3
import argparse

import tensorrt as trt

TRT_LOGGER = trt.Logger()


def onnx2trt(onnx_model, output_path):
    with trt.Builder(TRT_LOGGER) as builder:
        builder.max_workspace_size = 1 << 30  # 256MiB
        builder.max_batch_size = 1
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
    opt = parser.parse_args()
    onnx2trt(opt.onnx, opt.output)
