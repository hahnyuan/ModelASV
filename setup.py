from setuptools import setup, find_packages

setup(
    name="modelasv",
    version="0.2",
    description="ModelASV: Analyze, Simulate and Visulize the Performance of Large Neural Network Models",
    author="Zhihang Yuan",
    author_email="hahnyuan@gmail.com",
    url="https://github.com/hahnyuan/ModelASV",
    packages=find_packages(),
    install_requires=["torch", "transformers", "diffusers"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
