"""
Core functions for the NmfData object.
"""

from typing import Optional, Union
import pandas as pd
from prince import MCA

from .models import NmfModel
from .util import (
    _validate_identical_shapes,
    _validate_decomposition_shapes,
    _check_and_convert_binary_sparse,
    _convert_sparse
)

class NmfData(object):
    """
    Class representation of all related data for phylon analysis.
    """
    
    def __init__(
        self, P: pd.DataFrame, genome_table: Optional[pd.DataFrame],
        gene_table: Optional[pd.DataFrame], phylon_table: Optional[pd.DataFrame],
        L_norm: Optional[pd.DataFrame] = None, L_bin: Optional[pd.DataFrame] = None,
        A_norm: Optional[pd.DataFrame] = None, A_bin: Optional[pd.DataFrame] = None,
        V: Optional[pd.DataFrame] = None, U_norm: Optional[pd.DataFrame] = None,
        U_bin: Optional[pd.DataFrame] = None, F_norm: Optional[pd.DataFrame] = None,
        F_bin: Optional[pd.DataFrame] = None, mca: Optional[MCA] = None,
        nmf: Optional[NmfModel] = None, pvge: Optional[dict] = None, **kwargs
    ):
        """
        Initialize the NmfData object with required and optional dataframes.

        Parameters:
        - P: DataFrame with genes as rows and strains/genomes as columns.
        - genome_table: Optional DataFrame with genome_id as index and additional info like genome_name, mlst, mash_cluster.
        - gene_table: Optional DataFrame with functional annotations for all genes in the pangenome
        - phylon_table: Optional DataFrame with annotations of phylons
        - L_norm: Optional DataFrame for L normalization.
        - L_bin: Optional DataFrame for binary version of L_norm.
        - A_norm: Optional DataFrame for A normalization.
        - A_bin: Optional DataFrame for binary version of A_norm.
        - V: Optional DataFrame for alleles.
        - U_norm: Optional DataFrame for U normalization linked with V.
        - U_bin: Optional DataFrame for binary version of U_norm.
        - F_norm: Optional DataFrame for F normalization linked with V.
        - F_bin: Optional DataFrame for binary version of F_norm.
        - mca: Optional MCA model for optimal rank determination.
        - nmf: Optional NmfModel of results from running optimal rank determination.
        - pvge: Optional dictionary of results + clustering from running PVGE
        - kwargs: Additional keyword arguments like paths to .fna, .faa, .gff files.
        """
        self._P = P
        self._genome_table = genome_table
        self._gene_table = gene_table
        self._phylon_table = phylon_table
        self._L_norm = L_norm
        self._L_bin = L_bin
        self._A_norm = A_norm
        self._A_bin = A_bin
        self._V = V
        self._U_norm = U_norm
        self._U_bin = U_bin
        self._F_norm = F_norm
        self._F_bin = F_bin
        self.kwargs = kwargs
        self.validate_data()
        self.validate_model(model='all')

    
    def validate_data(self):
        """
        Validate the correctness of the inputs based on the provided specifications.
        """
        # Validate P matrix
        self._P = _check_and_convert_binary_sparse(self._P)
        
        # Validate L and A matrices if provided
        if self._L_norm is not None and self._A_norm is not None:
            _validate_decomposition_shapes(self._P, self._L_norm, self._A_norm, 'P', 'L_norm', 'A_norm')
            self._L_norm = _convert_sparse(self._L_norm)
            self._A_norm = _convert_sparse(self._A_norm)

            if self._L_bin:
                _validate_identical_shapes(self._L_norm, self._L_bin, 'L_norm', 'L_bin')
                self._L_bin = _check_and_convert_binary_sparse(self._L_bin)
            
            if self._A_bin:
                _validate_identical_shapes(self._A_norm, self._A_bin, 'A_norm', 'A_bin')
                self._A_bin = _check_and_convert_binary_sparse(self._A_bin)
        
        # Validate V matrix if provided
        if self._V:
            if self._V.shape[1] != self._P.shape[1]:
                raise ValueError("Columns in V must match the columns in P.")
            self._V = _check_and_convert_binary_sparse(self._V)
        
        # Validate U and F matrices if provided
        if self._V is not None and self._U_norm is not None and self._F_norm is not None:
            _validate_decomposition_shapes(self._V, self._U_norm, self._F_norm, 'V', 'U_norm', 'F_norm')
            self._U_norm = _convert_sparse(self._U_norm)
            self._F_norm = _convert_sparse(self._F_norm)
            
            if self._U_bin:
                _validate_identical_shapes(self._U_norm, self._U_bin, 'U_norm', 'U_bin')
                self._U_bin = _check_and_convert_binary_sparse(self._U_bin)
            
            if self._F_bin:
                _validate_identical_shapes(self._F_norm, self._F_bin, 'F_norm', 'F_bin')
                self._F_bin = _check_and_convert_binary_sparse(self._F_bin)
    
    def validate_model(model: Union[MCA, NmfModel, dict]):
        pass
    
    @property
    def P(self):
        """Get the P matrix."""
        return self._P
    
    @property
    def L(self):
        """Get the L matrix."""
        return self._L_norm
    
    @property
    def L_binarized(self):
        """Get the binarized version of the L matrix."""
        return self._L_bin
    
    @property
    def A(self):
        """Get the A matrix."""
        return self._A_norm
    
    @property
    def A_binarized(self):
        """Get the binarized version of the A matrix."""
        return self._A_bin
    
    @property
    def V(self):
        """Get the V matrix."""
        return self._V
    
    @property
    def U(self):
        """Get the U matrix."""
        return self._U_norm
    
    @property
    def U_binarized(self):
        """Get the binarized version of the U matrix."""
        return self._U_bin
    
    @property
    def F(self):
        """Get the F matrix."""
        return self._F_norm
    
    @property
    def F_binarized(self):
        """Get the binarized version of the F matrix."""
        return self._F_bin
    
    @property
    def genome_table(self):
        """Get the genome table."""
        return self._genome_table
    
    @genome_table.setter
    def genome_table(self, table):
        #TODO: implement this setter
        # genome_table = _check_table(table)
        # other code...
        # Set the genome table
        # self._genome_table = genome_table
        pass
    
    @property
    def gene_table(self):
        '''Get the gene annotations.'''
        return self._gene_table
    
    @gene_table.setter
    def gene_table(self, table):
        #TODO: implement this setter
        # gene_table = _check_table(table)
        # other code...
        # Set the gene annotations
        # self._gene_table = gene_table
        pass

    @property
    def phylon_table(self, table):
        '''Get the phylon table.'''
        return self._phylon_table
    
    @phylon_table.setter
    def phylon_table(self, table):
        #TODO: implement this setter
        # genome_table = _check_table(table)
        # other code...
        # set the phylon table
        # self._phylon_table = phylon_table
        pass

    def view_phylon(self, phylon: Union[int, str]):
        """
        View genes in a phylon and show relevant information about each gene.

        Parameters:
        - imodulon (int or str): Name of phylon
        
        Returns:
        - final_rows (pd.DataFrame): Table showing phylon gene information
        """
        # TODO: Only template has been implemented, more needs to be done
        # Find genes in phylon
        inPhylon = abs(self.L[phylon]) > self.thresholds[phylon]

        # Get gene weights information
        gene_weights = self.L.loc[inPhylon, phylon]
        gene_weights.name = "gene_weight"
        gene_rows = self.gene_table.loc[inPhylon]
        final_rows = pd.concat([gene_weights, gene_rows], axis=1)

        return final_rows
    
    def find_mobile_phylons(self):
        """
        Find all small phylons.
        
        These usually consist of fewer than 150 genes, and typically
        include mobile genetic elements such as sex pili, transposases,
        and components of the F plasmid transfer operon, among others.
        """
        # TODO: Implement this
        pass
