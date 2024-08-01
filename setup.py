from setuptools import setup, find_packages

setup(
    name="lm-view",
    version="0.1",
    description="LM-View: Your Description Here",
    author="Zhihang Yuan",
    author_email="hahnyuan@gmail.com",
    url="https://github.com/hahnyuan/LM-View",
    packages=find_packages(),
    install_requires=["torch", "transformers", "diffusers"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
