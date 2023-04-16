import torch
import torchvision
import tensorflow as tf
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import numpy as np
import tf2onnx


class TensorRTModel:
    def __init__(self, model, model_type="pytorch", input_shape=(1, 3, 224, 224), fp16_mode=False):
        self.model = model
        self.model_type = model_type
        self.input_shape = input_shape
        self.fp16_mode = fp16_mode

        self.engine = self._build_engine()
        self.context = self.engine.create_execution_context()

    def _build_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        if self.model_type.lower() == "pytorch":
            onnx_model = self._convert_pytorch_to_onnx()
        elif self.model_type.lower() == "tensorflow":
            onnx_model = self._convert_tensorflow_to_onnx()
        else:
            raise ValueError("Invalid model_type. Supported types: 'pytorch', 'tensorflow'.")

        parser.parse(onnx_model.SerializeToString())

        builder.max_workspace_size = 1 << 28
        builder.fp16_mode = self.fp16_mode

        engine = builder.build_cuda_engine(network)

        return engine

    def _convert_pytorch_to_onnx(self):
        model = self.model.eval()
        dummy_input = torch.randn(self.input_shape)
        torch.onnx.export(model, dummy_input, "temp.onnx", verbose=False, opset_version=11)
        onnx_model = onnx.load("temp.onnx")
        return onnx_model

    def _convert_tensorflow_to_onnx(self):
        model = self.model
        model_path = "temp.pb"
        tf.saved_model.save(model, model_path)
        onnx_model, _ = tf2onnx.convert.from_saved_model(model_path, opset=11)
        return onnx_model

    def forward(self, input_data):
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        elif not isinstance(input_data, np.ndarray):
            raise ValueError("Input data must be a NumPy array or a PyTorch tensor.")

        inputs, outputs, bindings, stream = self._allocate_buffers()

        np.copyto(inputs[0].host, input_data.ravel())

        with self.context as context:
            cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
            stream.synchronize()

        output_shape = tuple(self.engine.get_binding_shape(binding) for binding in self.engine if not self.engine.binding_is_input(binding))[0]
        return torch.tensor(outputs[0].host.reshape(output_shape))

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(trt.PluginTensorDesc(host_mem, device_mem))
            else:
                outputs.append(trt.PluginTensorDesc(host_mem, device_mem))
        return inputs, outputs, bindings, stream
