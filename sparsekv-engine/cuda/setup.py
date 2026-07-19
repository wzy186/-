from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sparsekv_cuda",
    ext_modules=[
        CUDAExtension(
            "sparsekv_cuda",
            sources=["sparse_attention.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090/4090
                    "-gencode=arch=compute_89,code=sm_89",  # RTX 4090 / L40
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
