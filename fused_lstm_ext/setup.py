import os
import sys

from setuptools import setup
from mlx import extension

# CMake's FindPython prefers a system Python.framework over the env running this
# build, so it would look for mlx/nanobind in the wrong interpreter. mlx's
# CMakeBuild forwards CMAKE_ARGS, so pin the interpreter to this one.
os.environ["CMAKE_ARGS"] = (
    f"-DPython_EXECUTABLE={sys.executable} " + os.environ.get("CMAKE_ARGS", "")
).strip()

if __name__ == "__main__":
    setup(name="mlx_fused_lstm", version="0.1.0",
          description="Input-fused LSTM (fwd + fused BPTT bwd) MLX extension.",
          ext_modules=[extension.CMakeExtension("mlx_fused_lstm._ext")],
          cmdclass={"build_ext": extension.CMakeBuild},
          packages=["mlx_fused_lstm"],
          package_data={"mlx_fused_lstm": ["*.so", "*.dylib", "*.metallib"]},
          zip_safe=False, python_requires=">=3.8")
