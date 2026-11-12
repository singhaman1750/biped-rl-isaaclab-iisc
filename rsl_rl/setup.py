from setuptools import find_packages, setup

setup(
    name="rsl_rl_debug",
    version="0.0.1",
    author="",
    author_email="",
    license="BSD-3-Clause",
    packages=find_packages(),
    description=(
        "Instrumented rsl_rl OnPolicyRunner subclass timing environment "
        "collection and PPO learning per phase, for training throughput "
        "debugging"
    ),
    python_requires=">=3.7.16",
    install_requires=["torch>=1.4.0"],
)
