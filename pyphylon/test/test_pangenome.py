import pytest
from pyphylon.pangenome import *
import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path
from pyphylon.pangenome import *

# we are calling CD-HIT in the snakemake pipeline, so it is difficult to test the pangenome construction functions fully as we are not running CD-HIT
# need to discuss with Jon and Sidd how we want to proceed. 

GENOMES_TO_TEST = ['798300.3','530008.3','286636.3']
CD_HIT_STATS = {"GENOMES":3, "CLUSTERS":2144, "ALLELES":5279}

testdata = [(GENOMES_TO_TEST, CD_HIT_STATS)]

@pytest.mark.parametrize("GENOMES,CD_HIT_STATS", testdata)
def test_cds_pangenome_construction(tmp_path, GENOMES,CD_HIT_STATS) -> None:
    PATHS = ['pyphylon/test/data/bakta/' + x + '/' + x + '.faa' for x in GENOMES]

    for item in Path('pyphylon/test/data/cd-hit-results/').iterdir():
        if item.is_file():
            shutil.copyfile(str(item), str(tmp_path / item.name))
            
    df_alleles, df_genes, header_to_allele = build_cds_pangenome(
        genome_faa_paths=PATHS,
        output_dir=str(tmp_path),
        name='test_files',
        cdhit_args={'-n': 5, '-c':0.8, '-aL':0.8, '-T': 0, '-M': 0},
        fastasort_path=None,
        save_csv=False
    )

    assert df_alleles.shape[0] == CD_HIT_STATS['ALLELES']
    assert df_genes.shape[0] == CD_HIT_STATS['CLUSTERS']
    assert df_alleles.shape[1] == len(GENOMES)
    assert df_genes.shape[1] == len(GENOMES)



genomes,core_size,acc_size,rare_size,core_prop,acc_prop,rare_prop = (50,200,500,1000,.99,.7,.05)
np.random.seed(42)
CORE_P = np.random.choice([0, 1], size=(core_size,genomes), p=[1-core_prop, core_prop])
ACC_P = np.random.choice([0, 1], size=(acc_size,genomes), p=[1-acc_prop, acc_prop])
RARE_P = np.random.choice([0, 1], size=(rare_size,genomes), p=[1-rare_prop, rare_prop])

P_MATRIX = pd.DataFrame(np.concatenate((CORE_P,ACC_P,RARE_P), axis=0))

@pytest.mark.parametrize('P_MATRIX',[(P_MATRIX)])
def test_find_pangenome_segments(P_MATRIX) -> None:
    segments, popt, r_squared, mae = find_pangenome_segments(P_MATRIX, threshold=0.1)
    
    df_freq = P_MATRIX.sum(axis=1)

    df_core = P_MATRIX[df_freq > np.floor(segments[0])]
    df_rare = P_MATRIX[df_freq < np.ceil(segments[1])]
    acc_gene_list = list(set(P_MATRIX.index)
                     - set(df_core.index)
                     - set(df_rare.index)
                    )
    df_acc = P_MATRIX.loc[acc_gene_list].copy()
    
    assert 0 < segments[0] < genomes
    assert 0 < segments[1] < genomes
    assert segments[0] > segments[1]
    assert df_core.sum(axis=1).min()  >= df_acc.sum(axis=1).min() >= df_rare.sum(axis=1).min()


@pytest.mark.parametrize('P_MATRIX',[(P_MATRIX)])
def test_heaps_law_functions(P_MATRIX) -> None:
    P_MATRIX = P_MATRIX.astype(pd.SparseDtype("int8", 0))
    df_pan_core = estimate_pan_core_size(P_MATRIX, num_iter=20, log_batch=1) 

    assert df_pan_core.shape[0] == 20
    assert df_pan_core.shape[1] == 4 * genomes

    output = fit_heaps_by_iteration(df_pan_core, section='pan')
    assert output.lambda_.mean() < 1


@pytest.mark.parametrize('P_MATRIX',[(P_MATRIX)])
def test_submatrix_calculation(P_MATRIX) -> None:
    P_submatrices = get_gene_frequency_submatrices(P_MATRIX, breakpoints=[0, 25, 50, 75, 100])

    for key1, item1 in P_submatrices.items():
        prev_item = None
        for key2, item2 in item1.items():
            if key1 >= key2:
                assert item2.shape == (0,0)
            else:
                if prev_item:
                    assert item2.shape[0] > prev_item.shape[0]
                assert item2.shape[1] == P_MATRIX.shape[1]
                assert item2.equals(P_MATRIX.loc[item2.index])