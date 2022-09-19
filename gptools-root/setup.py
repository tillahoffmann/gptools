from setuptools import setup

with open("../README.rst") as fp:
    long_description = fp.read()
long_description = long_description.replace(":doc:", ":code:").replace(".. toctree::", "..")

setup(
    name="gp-tools",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    version="0.1.0",
    install_requires=[
        "gp-tools-stan",
        "gp-tools-torch",
        "gp-tools-util",
    ],
    tests_require=[
        "twine",
    ],
)
