from setuptools import setup

setup(
    name="dominant_color",
    version="0.0.0",
    author="Xiangyi Gao",
    description="extract dominant colors form an image.",
    packages=["dominant_color"],
    install_requires=["opencv-python", "scikit-learn"],
    extras_require={"dev": ["pytest"]},
)
