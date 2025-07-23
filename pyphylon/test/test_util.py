from pyphylon.util import *
import pandas as pd
import numpy as np
import  pytest


def test_load_config() -> None:
    CONFIG = load_config('pyphylon/test/data/util_test_files/config.example.yml')
    assert CONFIG['WORKDIR'] == 'data/'
    assert CONFIG['SPECIES_NAME'] == "Streptococcus pyogenes"
    assert CONFIG['DEBUG'] == True
    assert CONFIG['PG_NAME']  =="SPyogenes"
    assert CONFIG['METADATA_FILE']  == "pyphylon/test/data/util_test_files/spyogenes_metadata_summary.tsv"
    assert CONFIG['GENOMES_FILE']  == "pyphylon/test/data/util_test_files/spyogenes_genome_summary.tsv"
    assert CONFIG['FILTERED_GENOMES_FILE']  == "/examples/data/interim/mash_scrubbed_species_metadata_2b.csv"


def test_sparse_conversion() -> None:
    from pyphylon.util import _convert_sparse
    df = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0]
    })
    result = _convert_sparse(df, dtype='int8')
    
    assert all(isinstance(result[col].dtype, pd.SparseDtype) for col in result.columns)
    assert result.dtypes.unique().tolist() == [pd.SparseDtype("int8", 0)]
    assert result.sparse.to_dense().eq(df).all().all()

    # test with floats
    df = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0]
    }, dtype=float)
    result = _convert_sparse(df, dtype='int8')
    
    assert all(isinstance(result[col].dtype, pd.SparseDtype) for col in result.columns)
    assert result.dtypes.unique().tolist() == [pd.SparseDtype("int8", 0)]
    assert result.sparse.to_dense().eq(df).all().all()

    
def test_binary_sparse() -> None:
    from pyphylon.util import _check_and_convert_binary_sparse
    df = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0]
    })

    result = _check_and_convert_binary_sparse(df)
    
    assert all(isinstance(result[col].dtype, pd.SparseDtype) for col in result.columns)
    assert result.sparse.to_dense().eq(df).all().all()

    df = pd.DataFrame({
        'a': [0, 2, 0],
        'b': [1, 0, -1]
    })

    with pytest.raises(Exception):
        result = _check_and_convert_binary_sparse(df)


def test_get_normalization_diagonals() -> None:
    from pyphylon.util import _get_normalization_diagonals
    
    df = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0]
    })

    df2 = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0],
        'c': [1, 0, 1]
    })

    result1 = _get_normalization_diagonals(df)
    result2 = _get_normalization_diagonals(df2)
    
    assert ((result1[0] @ result1[1]) == np.identity(df.shape[1])).all()
    assert ((result2[0] @ result2[1]) == np.identity(df2.shape[1])).all()


def test_validate_matrix_shape() -> None:
    from pyphylon.util import _validate_identical_shapes

    df1 = pd.DataFrame({
        'a': [0, 1, 0],
        'b': [1, 0, 0]
    })

    df2 = pd.DataFrame({
        'a': [1, 1, 0],
        'b': [1, 0, 1]
    })

    _validate_identical_shapes(df1, df2, 'check1', 'check2')

    df3 = pd.DataFrame({
        'a': [1, 1],
        'b': [1, 0]
    })
    
    with pytest.raises(Exception):
        _validate_identical_shapes(df1, df3, 'check1', 'check3')


def test_validate_decomposition_shape() -> None:
    from pyphylon.util import _validate_decomposition_shapes

    df_base = pd.DataFrame(np.zeros((100,100)))
    
    df1 = pd.DataFrame(np.zeros((100,10)))

    df2 = pd.DataFrame(np.zeros((10,100)))

    _validate_decomposition_shapes(df_base, df1, df2, 'base', 'decomp1', 'decomp2')

    df1 = pd.DataFrame(np.zeros((99,10)))
    with pytest.raises(Exception):
        _validate_decomposition_shapes(df_base, df1, df2, 'base', 'decomp1', 'decomp2')

    df1 = pd.DataFrame(np.zeros((100,10)))
    df2 = pd.DataFrame(np.zeros((10,99)))
    with pytest.raises(Exception):
        _validate_decomposition_shapes(df_base, df1, df2, 'base', 'decomp1', 'decomp2')

    df1 = pd.DataFrame(np.zeros((100,9)))
    df2 = pd.DataFrame(np.zeros((10,100)))
    with pytest.raises(Exception):
        _validate_decomposition_shapes(df_base, df1, df2, 'base', 'decomp1', 'decomp2')

    
def test_remove_empty_files() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files
        empty_file_1 = os.path.join(tmpdir, "empty1.txt")
        empty_file_2 = os.path.join(tmpdir, "empty2.txt")
        non_empty_file = os.path.join(tmpdir, "data.txt")

        # Write empty files
        open(empty_file_1, 'w').close()
        open(empty_file_2, 'w').close()

        # Write non-empty file
        with open(non_empty_file, 'w') as f:
            f.write("content")

        # Run the function
        deleted_files = remove_empty_files(tmpdir)

        # Check that only empty files were deleted
        assert sorted(deleted_files) == sorted(["empty1.txt", "empty2.txt"])
        assert not os.path.exists(empty_file_1)
        assert not os.path.exists(empty_file_2)
        assert os.path.exists(non_empty_file)