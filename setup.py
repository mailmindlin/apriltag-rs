from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="apriltag_rs",
    version="1.0",
    rust_extensions=[
        RustExtension(
            target="apriltag_rs.apriltag_rs",
            features=["cpython"],
            binding=Binding.RustCPython
        )
    ],
    packages=["apriltag_rs"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)