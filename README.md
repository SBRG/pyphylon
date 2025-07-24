# pyphylon

Python package for analyzing and visualizing co-occuring gene / allele sets (phylons) within a pangenome.

## Features

- Automated workflows for downloading genomic data from sources like NCBI and BV-BRC.
- Extensive data preprocessing including QC/QA, annotation with tools like bakta, Mash, MLST, and serotype analysis.
- Generation of pangenomes and comprehensive eggNOG annotations.
- Detailed analysis using various forms of normalized and binary data matrices.
- Object-oriented design for extensible and scalable development.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/SBRG/pyphylon.git
```

Navigate into the project directory:

```bash
cd pyphylon
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install pyphylon
```
pip install -e .
```

## Usage

We recommend working through the <a href="https://github.com/SBRG/pyphylon/blob/streamline-configs/examples/README.md">example</a>

## Running Docker:
1. Build the container with: `docker build -t pyphylon .`
2. Run the container interactively with: `docker run --privileged -it -v %cd%/examples/data:/data -v %cd%/workflow:/workflow pyphylon`
3. from INSIDE the container cd to /workflow (`cd workflow`)
4. Run snakemake with: `snakemake -d /data --use-singularity -c 5`

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. Tests are performed using pytest, so please install 'pytest' in addition to the package requirements if you wish to contribute and run tests. Calling `pytest` in the terminal from the root of the package should run all tests implemented to this point. 

Install pytest:
```
pip install pytest
```

## License

This project is licensed under the [MIT License](LICENSE).

