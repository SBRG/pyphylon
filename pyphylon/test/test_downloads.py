import pytest
from pyphylon.downloads import *

def test__get_scaffold_n50_for_species():
    taxon_id =  '562'
    n50_value = get_scaffold_n50_for_species('562')
    assert n50_value == 4600000