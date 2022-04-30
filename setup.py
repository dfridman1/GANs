#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="GANs",
    version="0.0.1",
    description="GANs playground",
    packages=find_packages(include=["dcgan", "pix2pix"]),
    install_requires=[
        "torch",
        "torchvision",
        "tensorboard",
        "tensorflow",
        "ipython",
        "tqdm",
        "jupyter"
    ],
)
