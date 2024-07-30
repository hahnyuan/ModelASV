from setuptools import setup, find_packages
from setuptools import setup, find_packages

setup(
    name="lm-view",
    version="0.1",
    description="LM-View: Your Description Here",
    author="Zhihang Yuan",
    author_email="hahnyuan@gmail.com",
    url="https://github.com/hahnyuan/LM-View",
    packages=find_packages(),
    install_requires=["torch"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    package_dir={"lm_view": "path/to/lm_view"},  # Replace "path/to/lm_view" with the actual path to the lm_view folder
    package_data={"lm_view": ["*.txt", "*.csv"]},  # Include any additional data files if needed
)
