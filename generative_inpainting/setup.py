from setuptools import find_packages, setup

setup(
    name="generative_inpainting",
    version="0.0.1",
    author="Xiangyi Gao",
    description="inpainting image hole",
    packages=find_packages(),
    install_requires=[
        "torch==1.8.1",
        "torchvision==0.9.1",
    ],
)
