"""
Created on Jan 09, 2024
@author: FionaFeng

Data preprocess: rarery data matrix, deal with zeros, filt OTUs/MAGs that keeps low frequency in all samples.
"""

import numpy as np
import pandas as pd
from typing import Union


class AbsTaxa:
    """
    AbsRaeds array format: Using microbial aundance matrix M_{n*p} as input
    n: having n samples, columns in this matrix
    p: having p taxa, rows in this matrix
    The term taxa could be OTUs, ASVs, MAGs, and so on
    """
    def __init__(self, SampleID:list, TaxaID:list, AbsReads:np.ndarray):
        self.SampleID = SampleID
        self.TaxaID = TaxaID
        self.AbsReads = AbsReads
    
    """
    Data format checking and data display
    """
    # Check if the input abundance data format is correct
    def check_array(self):
        myarray = self.AbsReads
        rows = len(self.TaxaID)
        cols = len(self.SampleID)
        if (myarray.shape[0] == rows) and (myarray.shape[1] == cols):
            print("丰度矩阵格式正确！")
            print("Format of abundence array is right!")
        else:
            print("格式有误，请检查！")
            print("Format of the array is wrong, please check it again!")      
    
    def show(self) -> pd.DataFrame:
        df = pd.DataFrame(self.AbsReads, columns=self.SampleID)
        df = df.set_index(pd.Index(self.TaxaID))
        return df
    

    # Flattening of the absolute abundance matrix
    """
    1. Determine the need for flattening
    - If the difference between the total reads of the largest and smallest samples is less than or equal to 10, flattening is not necessary by default

    2. rarefying and diluting the OTU table
    - Rarefaction of the OTU matrix by the minimum sample size (column sum)
    - Dilute the OTU matrix by a specified dilution factor
    """
    ## To determine if rarefaction is needed
    def is_flat(self) -> bool:
        myarray = self.AbsReads
        sample_total_reads = np.sum(myarray, axis=0)
        # Criteria
        if (max(sample_total_reads) - min(sample_total_reads)) < 10:
            print(True)
        else:
            print(False)
    
    ## Function for rarefaction and dilution
    def rarefy(self, rarecount:int) -> 'AbsTaxa':
        myarray = self.AbsReads
        sample_total_reads = np.sum(myarray, axis=0)
        # Calculate the dilution/rarefaction factor
        dilution_factor = rarecount / sample_total_reads[:, np.newaxis]

        # Perform dilution/rarefaction on each sample
        rarefied_array = np.zeros_like(myarray)
        for i, factor in enumerate(dilution_factor):
            rare_counts = np.random.multinomial(rarecount, myarray[:, i] / np.sum(myarray[:, i]))
            rarefied_array[:, i] = rare_counts
        # Create a new AbsTaxa object and return it
        rarefied_taxa = AbsTaxa(self.SampleID, self.TaxaID, rarefied_array)
        return rarefied_taxa
    

    # zero inflation and zero value handling
    ## Calculating the zero inflation rate
    def zi_rate(self, decimal:int = 2) -> float:
        t = self.AbsReads.size
        non_zeros = np.count_nonzero(self.AbsReads)
        zir = (t - non_zeros) / t
        zir = round(zir, decimal)
        return print("Zero inflation rate: ", zir)
    ## Zero value filling and processing
    def zero_fill(self, pseudo: Union[int, float] = 0.01, mod: str = 'all') -> 'AbsTaxa':
        # Determine the types of filled pseudo counts
        if isinstance(pseudo, int):
            myarray = self.AbsReads
        elif isinstance(pseudo, float):
            myarray = self.AbsReads.astype(float)
        else:
            raise TypeError("pseudo must be an integer or float!")
        
        # Fill zero values
        taxas = myarray.shape[0]
        samples = myarray.shape[1]
        if mod == 'all':
            myarray[myarray == 0] = pseudo
        elif mod == 'pair':
            for i in range(samples):
                n_zeros = taxas - np.count_nonzero(myarray[:, i])
                if n_zeros > 1:
                    zero_loc = np.where(myarray[:, i] == 0)[0]
                    sl = n_zeros - 1
                    selected_loc = np.random.choice(zero_loc, sl, replace=False)
                    (myarray[:, i])[selected_loc] = pseudo
        else:
            raise TypeError("mod must be 'all' or 'pair'!")
        
        # Create a new AbsTaxa object and return it
        zerofilled_taxa = AbsTaxa(self.SampleID, self.TaxaID, myarray)
        return zerofilled_taxa
    

    # Frequency of occurrence of a taxon in all samples
    ## Frequency of occurrence calculation
    def taxa_freq(self, pseudo: float = 0.01) -> pd.Series:
        myarray = self.AbsReads
        rows = myarray.shape[0]
        
        freq_list = []
        for i in range(rows):
            total_elements = len(myarray[i])
            non_zero_count = np.count_nonzero(myarray[i] > pseudo)
            freq = non_zero_count / total_elements
            freq_list.append(freq)
        freq_s = pd.Series(freq_list, index=self.TaxaID)
        return freq_s
    
    ## Trim taxa based on the set frequency threshold
    def taxa_freq_cut(self, cutoff: float = 0.0001) -> 'AbsTaxa':
        freq_s = self.taxa_freq()

        cutid_list = []
        for i in range(len(freq_s)):
            if (freq_s.iloc[i] < cutoff) or (freq_s.iloc[i] == cutoff):
                 cut_id = freq_s.index[i]
                 cutid_list.append(cut_id)
        
        if (len(cutid_list) == 0):
            print("No taxa is need to be filted!")
        else:
            df_ori = self.show()
            df_new = df_ori.drop(cutid_list, axis=0, errors='ignore')
            sample_id = df_new.columns.to_list()
            taxa_id = df_new.index.to_list()
            new_array = df_new.values
            # Create a new AbsTaxa object and return it
            fillted_taxa = AbsTaxa(sample_id, taxa_id, new_array)
            return fillted_taxa
       

