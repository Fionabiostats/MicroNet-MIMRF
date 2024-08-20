"""
Created on Jan 11, 2024
@author: FionaFeng

Data transformation: Converting absolute abundance table to relative abundance, and methods for handling biased compositional data transformation
"""

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from MicroNet_MIMRF.DataPreprocess import AbsTaxa


class RelTaxa(AbsTaxa):
    """
    Format of RelReads array should be consistent with format of AbsReads array
    """
    def __init__(self, absTaxa: AbsTaxa, RelReads: np.ndarray = None):
        super().__init__(absTaxa.SampleID, absTaxa.TaxaID, absTaxa.AbsReads)
        self.RelReads = RelReads
        if RelReads is None:
            self.calculate_rel_reads()
        else:
            self.check()
    
    # Check if the user-input relative abundance matrix and absolute count matrix are consistent
    def check(self):
        abs_array = self.AbsReads
        rel_array = self.RelReads
        if (abs_array.shape) == (rel_array.shape):
            self.RelReads = rel_array
        else:
            raise TypeError("The shape of relative array is different from absolute array. Please check again!")
    
    # Calculate relative abundance matrix
    def calculate_rel_reads(self):
        abs_array = self.AbsReads
        sample_total_reads = np.sum(abs_array, axis=0)
        rel_array = abs_array / sample_total_reads[np.newaxis, :]
        self.RelReads = rel_array
        return rel_array
    
    def show_rel(self) -> pd.DataFrame:
        df = pd.DataFrame(self.RelReads, columns=self.SampleID)
        df = df.set_index(pd.Index(self.TaxaID))
        return df
    

"""
Log-ratio transformation method for relative abundance (compositional data): 
- ALR: Log-ratio additive transformation
- CLR: Centered log-ratio transformation
"""
class LRTable(RelTaxa):
    def __init__(self, relMatrix: np.ndarray, method: str='alr', lrMatrix: np.ndarray = None):
        self.relMatrix = relMatrix
        self.lrMatrix = lrMatrix
        if lrMatrix is None:
            if method == 'alr':
                self.alr()
            elif method == 'clr':
                self.clr()
            else:
                raise TypeError("method must be 'alr' or 'clr'!")
    
    ## ALR Method
    def alr(self) -> np.ndarray:
        rel_matrix = self.relMatrix
        if rel_matrix.ndim == 2:
            num_index = list(range(0, rel_matrix.shape[0]))
            del num_index[0]
            ln_matrix = np.log(rel_matrix[num_index, :] / rel_matrix[0, :])
            self.lrMatrix = ln_matrix
        else:
            raise ValueError("The demention of relative matrix must be 2.")

    ## CLR Method
    def clr(self) -> np.ndarray:
        rel_matrix = self.relMatrix
        ln_matrix = np.log(rel_matrix)
        geo_mean = ln_matrix.mean(axis=0)
        result = (ln_matrix - geo_mean)
        self.lrMatrix = result

