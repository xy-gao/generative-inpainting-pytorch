from setuptools import setup

setup(
    name="generative_inpainting",
    version="0.0.0",
    author="Xiangyi Gao",
    description="inpainting image hole",
    packages=["generative_inpainting"],
    install_requires=[
        "torch==1.8.1",
        "torchvision==0.9.1",
        ],
)
