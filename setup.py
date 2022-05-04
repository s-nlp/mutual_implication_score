import os
import setuptools

description = "Yet another common wrapper for Alice/Salut skills and Facebook/Telegram/VK bots"
long_description = description
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()


setuptools.setup(
    name="mutual_imp",
    version="0.0.0",
    author="Nikolai Babakov and David Dale",
    author_email="dale.david@mail.ru",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skoltech-nlp/mutual_implication_score",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['transformers>=4.13.0', 'pytorch', 'tqdm'],
)
