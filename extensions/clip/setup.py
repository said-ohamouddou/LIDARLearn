from setuptools import setup, find_packages

setup(
    name="clip",
    version="1.0",
    description="OpenAI CLIP",
    author="OpenAI",
    py_modules=["clip"],
    packages=find_packages(exclude=["tests*"]),
    install_requires=["ftfy", "packaging", "regex"],
    include_package_data=True,
    # Ship the BPE vocabulary with non-editable installs too.
    package_data={"clip": ["bpe_simple_vocab_16e6.txt.gz"]},
)
