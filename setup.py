import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="gestualis",
    version="0.0.5",
    author="Sujatro Ganguli",
    author_email="iamsurjog@gmail.com",
    packages=["gestualis"],
    description="A sample test package",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamsurjog/Gestualis",
    license=None,
    python_requires='>=3.11',
    install_requires=[]
)