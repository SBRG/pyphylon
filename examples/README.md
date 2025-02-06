

# Instructions for Running the Example (S. pyogenes)

## There are two options for running the example:
1. Run all bioinformatics yourself using the attached Snakemake Workflows (in folder workflow)
    * Download <a href="https://www.dropbox.com/s/ksvdgi8xgfx5m2r/spyogenes_metadata_summary.tar?dl=0">metadata</a>
    * Untar these two files into a data/metadata directory in the "examples" folder
1. Download the result of all bioinformatics scripts, copy and extract to the "data" folder
    * Download <a href="https://www.dropbox.com/s/5clg2tq6w4u67u7/SPyogenes_example.tar?dl=0">all</a>
    * Unzip all files into the data directory in the "examples" folder

## Running the Example (once you've downloaded and extracted one of the two files above)
#### Note if you select option 2 above you can skip steps 3 and 5 below

1. Run notebook 1a. This will prepare genomes for download from BV-BRC
    * Output is: interim/genome_summary_1a.csv and interim/genome_metadata_1a.csv
1. Run notebook 1b - this will download the filtered genomes from BV-BRC (note you can skip running this notebook if you downloaded the data file)
    * Output is data/interim/genome_summary_1b.csv and data/interim/genome_metadata_1b.csv
1. Now run mash workflow to filter out any genomes that aren't in the species:
      1. Build the container with: `docker build -t pyphylon .`
      1. Run the container interactively with: `docker run --privileged -it -v %cd%/examples/data:/data -v %cd%/workflow:/workflow pyphylon`
      1. from INSIDE the container cd to /workflow (`cd workflow`)
      1. for the MASH workflow cd to the "mash" folder: `cd mash`
      1. Run snakemake with: `snakemake -d /examples/data --use-singularity -c 10` 
    * This will generate results in processed/mash
1. Run notebooks 2a and 2b. This will use mash to filter strains
    * Output is interim/mash_scrubbed_species_summary_2b.csv and interim/mash_scrubbed_species_metadata_2b.csv
    * NOTE these files are used in WORKFLOW TWO to run CDHIT, annotation and MLST
1. Now run CDHIT, ANNOTATION and MLST
    1. From inside the same container from step 5 above cd to /workflow/anno_mlst_cdhit (`cd /workflow/anno_mlst_cdhit`)
    1. Run snakemake with: `snakemake -d /examples/data --use-singularity -c 10` 
    * Output is processed/cd-hit-results, processed/bakta, processed/mlst
1. Run notebook 2c - this will load the cd-hit-results into a pg matrix saved as csv
    - Output is: processed/ch-hit-results: SPyogenes_strain_by_gene.pickle.gz, etc.
1. Run notebook 2d to enrich with MLST
    - Output is: processed/interim/enriched_metadata_2d.csv
1. Run notebooks 3a and b to characterize core, accessory and rare genomes
    - Output is: processed/CAR_genomes/df_acc.csv, df_core.csv and df_rare.csv
1. Run notebooks 4a for NMF
    - Output is: processed/nmf-outputs/A.csv, L.csv
1. Run notebooks 5a and 5b to characterized NMF, L and A matrices
    - Output is: processed/nmf-outputs/A_norm.csv, L_norm.csv, A_bin.csv, L_bin.csv



Folder Structure (within the examples folder):
```
├── data
  ├── raw
    ├── genomes
      ├── fna # genomes in fasta format with .fna extension
  ├── interim # csv files for interim metadata
  ├── processed # post bioinformatics scripts and notebook processing
    ├── bakta # result of bakta annotation, each genome results in a folder with ID
    ├── cd-hit-results # results of CD-HIT
    ├── mlst # results of MLST
    ├── mash # result of mash sketches

```