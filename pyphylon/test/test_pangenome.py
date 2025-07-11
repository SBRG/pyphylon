import pytest
from pyphylon.pangenome import *
import pandas as pd
import numpy as np
import os

@pytest.fixture
def faa_paths():
    path_to_files = './pyphylon/test/data/faa/'
    return [path_to_files + str(x) for x in os.listdir('pyphylon/test/data/faa') if '.faa' in x]



# we are calling CD-HIT in the snakemake pipeline, so it is difficult to test the pangenome construction functions fully as we are not running CD-HIT
# need to discuss with Jon and Sidd how we want to proceed. 