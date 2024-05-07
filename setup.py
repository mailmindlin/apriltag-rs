from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="apriltag_rs",
    version="1.0",
    rust_extensions=[
        RustExtension(
            target="apriltag_rs.apriltag_rs_native",
            features=["cpython", "debug", "wgpu"],
            binding=Binding.RustCPython,
            debug=True,
        )
    ],
    packages=["apriltag_rs"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)