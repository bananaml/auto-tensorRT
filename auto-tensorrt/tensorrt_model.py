import torch
import torchvision
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_model_to_trt(model, input_shape, output_path, fp16_mode=False):
    # Convert the model to TorchScript
    model.eval()
    scripted_model = torch.jit.trace(model, torch.zeros(input_shape))

    # Create a TensorRT builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the TorchScript model to TensorRT
    with torch.no_grad():
        onnx_model = torch.onnx.export(scripted_model, torch.zeros(input_shape), None, verbose=False, opset_version=11, do_constant_folding=True)
    parser.parse(onnx_model.SerializeToString())

    # Configure the builder and create the engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    if fp16_mode:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1,) + input_shape[1:], input_shape, (1,) + input_shape[1:])
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)

    # Save the TensorRT engine to the specified output path
    with open(output_path, "wb") as f:
        f.write(engine.serialize())

    return output_path
