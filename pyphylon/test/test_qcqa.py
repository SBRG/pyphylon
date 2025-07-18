from pyphylon.qcqa import *
import pandas as pd
import numpy as np
import  pytest
import os

@pytest.fixture
def genome_file():
    from pyphylon.util import load_config
    CONFIG = load_config('pyphylon/test/data/util_test_files/config.example.yml')
    return CONFIG['GENOMES_FILE']

@pytest.fixture
def species_name():
    from pyphylon.util import load_config
    CONFIG = load_config('pyphylon/test/data/util_test_files/config.example.yml')
    return CONFIG['SPECIES_NAME']

def test_filter_by_species(genome_file, species_name) -> None:
    summary_file = pd.read_csv(genome_file, sep='\t', dtype=object)
    species_summary = filter_by_species(summary_file, species_name)
    assert len(species_summary) == 2286


def test_filter_by_genome_quality(genome_file) -> None:
    summary_file = pd.read_csv(genome_file, index_col=0, dtype={'genome_id':str}, sep='\t')
    species_summary, stats = filter_by_genome_quality(summary_file)
    assert len(species_summary) == 257