import whisper
import base64
from io import BytesIO
import torch
from collections import OrderedDict
import tensorrt as trt
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util
import numpy as np
from copy import copy
import os

class Engine():
    def __init__(
        self,
        model_name,
        engine_dir,
    ):
        self.engine_path = os.path.join(engine_dir, model_name+'.plan')
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def build(self, onnx_path, fp16, input_profile=None, enable_preview=False):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        preview_features = []
        if enable_preview:
            trt_version = [int(i) for i in trt.__version__.split(".")]
            # FASTER_DYNAMIC_SHAPES_0805 should only be used for TRT 8.5.1 or above.
            if trt_version[0] > 8 or \
                (trt_version[0] == 8 and (trt_version[1] > 5 or (trt_version[1] == 5 and trt_version[2] >= 1))):
                preview_features = [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]

        engine = engine_from_network(network_from_onnx_path(onnx_path), config=CreateConfig(fp16=True,max_workspace_size=8100654080, profiles=[p],
            preview_features=preview_features))
        save_engine(engine, path=self.engine_path)

    def activate(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, device='cuda'):
        for idx in range(trt_util.get_bindings_per_profile(self.engine)):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)
            dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            # Workaround to convert np dtype to torch
            np_type_tensor = np.empty(shape=[], dtype=dtype)
            torch_type_tensor = torch.from_numpy(np_type_tensor)
            tensor = torch.empty(tuple(shape), dtype=torch_type_tensor.dtype).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)

    def infer(self, feed_dict, stream):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors




model = whisper.load_model("base")
audio_enc = model.encoder
inputs = torch.randn(1,80,3000, dtype=torch.float16, device='cuda')
torch.onnx.export(audio_enc,               # model being run
                     inputs,                         # model input (or a tuple for multiple inputs)
                     "whisper_encoder.onnx",   # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=12,          # the ONNX version to export the model to
                     do_constant_folding=True,  # whether to execute constant folding for optimization
                     input_names = ['input_0'],
                     output_names = ['output_0'])

encoder_engine = Engine("whisper_encoder","./trtengine")
encoder_engine.build("whisper_encoder.onnx",fp16=False,input_profile=None,enable_preview=False)
encoder_engine.activate()
encoder_engine.allocate_buffers(shape_dict={"input_0":(1,80,3000)}, device="cuda:0")
model.encoder = encoder_engine
print(model.transcribe("test.mp3"))
# import tensorrt as trt
# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# builder = trt.Builder(TRT_LOGGER)
# network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
# parser = trt.OnnxParser(network, TRT_LOGGER)
