import pytest
from pyphylon.downloads import *
from unittest.mock import patch, mock_open, MagicMock, call

@pytest.fixture
def taxon_id_and_value():
    # test values of input and expected output for E. coli K12 reference strain
    # API returns exact scaffold N50 in bp (previously 4600000 from rounded "4.6 Mb" Selenium scrape)
    return ('562', 4641652)


def test__get_scaffold_n50_for_species(taxon_id_and_value) -> None:
    n50_value = get_scaffold_n50_for_species(taxon_id_and_value[0])
    assert n50_value == taxon_id_and_value[1]

# prevent file download in function, test correct download functionality
@patch('pyphylon.downloads.download_from_bvbrc')
def test_download_bvbrc_genome_info(mock_download, tmp_path) -> None:
    download_bvbrc_genome_info(output_dir = tmp_path, force = False)

    assert mock_download.call_count == 3
    assert mock_download.call_args_list[0][0][0] == 'RELEASE_NOTES/genome_summary'

# test if file download is skipped if files already exist
@patch("pyphylon.downloads.download_from_bvbrc")
def test_skip_download_if_files_exist(mock_download, tmp_path):
    # Create temp files for summary and metadata
    for fname in ['genome_summary.tsv', 'genome_metadata.tsv', 'PATRIC_genome_AMR.tsv']:
        (tmp_path / fname).write_text("existing content")

    download_bvbrc_genome_info(output_dir=tmp_path, force=False)

    assert mock_download.call_count == 0

@patch("pyphylon.downloads.requests.get")
def test_download_example_bvbrc_genome_info(mock_get, tmp_path) -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"test data"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    download_example_bvbrc_genome_info(output_dir=tmp_path, force=False)

    assert mock_get.call_count == 2
    assert mock_get.call_args_list[1][0][0] == "https://zenodo.org/record/11226678/files/genome_summary_Oct_12_23.tsv?download=1"
    assert mock_get.call_args_list[0][0][0] == "https://zenodo.org/record/11226678/files/genome_metadata_Oct_12_23.tsv?download=1"