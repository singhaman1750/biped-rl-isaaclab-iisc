from setuptools import find_packages, setup

setup(
    name="co_optimisation",
    version="0.0.1",
    author="Junfeng Long, Zirui Wang, Nikita Rudin",
    author_email="",
    license="BSD-3-Clause",
    packages=find_packages(),
    description="Fast and simple RL algorithms implemented in pytorch, with hybrid internal model",
    python_requires=">=3.7.16",
    install_requires=["torch>=1.4.0", "torchvision>=0.5.0", "numpy>=1.16.4"],
)
