from setuptools import setup

setup(
    name="dominant_color",
    version="0.0.0",
    author="Xiangyi Gao",
    description="extract dominant colors form an image.",
    packages=["dominant_color"],
    install_requires=["opencv-python==4.5.1.48", "scikit-learn==0.24.1"],
    extras_require={"dev": ["pytest"]},
)
