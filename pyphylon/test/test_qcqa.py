from pyphylon.qcqa import *
import pandas as pd
import numpy as np
import  pytest
import os

def test_filter_by_species() -> None:
    from pyphylon.util import load_config

    CONFIG = load_config('examples/config.example.yml')
    summary_file = pd.read_csv('.' + CONFIG['GENOMES_FILE'], sep='\t', dtype=object)
    species_summary = filter_by_species(summary_file, CONFIG['SPECIES_NAME'])
    assert len(species_summary) == 2286


def test_filter_by_genome_quality() -> None:
    from pyphylon.util import load_config

    CONFIG = load_config('examples/config.example.yml')
    summary_file = pd.read_csv('.' + CONFIG['GENOMES_FILE'], index_col=0, dtype={'genome_id':str}, sep='\t')
    species_summary, stats = filter_by_genome_quality(summary_file)
    assert len(species_summary) == 257