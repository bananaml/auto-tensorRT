from setuptools import setup, find_packages

setup(
    name='auto-tensorrt',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tensorflow',
        'tensorrt',
        'pycuda',
        'onnx'
    ],
)
