from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pyphylon",
    version="0.0.1",
    author="Siddharth M Chauhan",
    author_email="smchauhan@ucsd.edu",
    description="Python package for constructing, analyzing, & visualizing co-occuring gene / allele sets (phylons) within a pangenome.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SBRG/pyphylon/",
    packages=find_packages(exclude=["tests", "examples"]),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={
        'cd-hit': ['cd-hit'],
        'tests': ['pytest']
    },
    python_requires='>=3.11',
)
