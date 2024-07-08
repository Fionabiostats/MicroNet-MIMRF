"""
Created on Mar 13, 2024
@author: FionaFeng

Calculating conditional dependencies between taxon and taxon
"""

import numpy as np
import pandas as pd


# Zero-inflation model fitting
class FitModels:
    """
    Input Dataframe type of data: Each column represents one Taxa (or OTU, ASV, MAG, etc.), and each row represents one sample
    """

    def __init__(self, data: pd.DataFrame, method: str='zip', expected: pd.DataFrame = None):
        self.data = data
        self.taxa = self.data.columns.to_list()
        self.samples = self.data.index.to_list()
        self.counts = self.data.values
        self.expected = expected
        
        if expected is None:
            if method == 'zip':
                self.zip()
            elif method == 'zip2':
                self.zip2()
            else:
                raise TypeError("method must be 'zip' or 'zip2'!")


    # ZIP model by default
    def zip(self) -> pd.DataFrame:
        # Implement zip method
        data = self.data
        samples = self.samples
        taxa = self.taxa
        counts = self.counts
        expected_df = pd.DataFrame(np.zeros_like(counts), index=samples, columns=taxa)

        # model fitting
        from statsmodels.discrete.count_model import ZeroInflatedPoisson
        total_sum = data.sum(axis=1)
        taxa_all_sum = data.sum(axis=0)
        for i in taxa:
            y = data[i]
            taxa_i_sum = taxa_all_sum.iloc[i]
            x_cov = taxa_i_sum / total_sum
            x_cov = np.nan_to_num(x_cov, nan=0, posinf=0.9, neginf=0)
            taxa_i_bar = data[i].mean()
            x_dep = data[i] / taxa_i_bar
            x_dep = np.nan_to_num(x_dep, nan=0, posinf=10, neginf=0)
            x = pd.DataFrame({"coverage":x_cov, "deepth":x_dep})
            result = ZeroInflatedPoisson(y, x).fit()

            # calculate expectation
            expectation = result.predict()
            expected_df[i] = expectation

        # return the expectation dataframe
        self.expected = expected_df

    # ZIP2 model  method='lbfgs'  ‘lbfgs’ for limited-memory BFGS with optional box constraints
    def zip2(self) -> pd.DataFrame:
        # Implement zip method
        data = self.data
        samples = self.samples
        taxa = self.taxa
        counts = self.counts
        expected_df = pd.DataFrame(np.zeros_like(counts), index=samples, columns=taxa)

       # model fitting
        from statsmodels.discrete.count_model import ZeroInflatedPoisson
        total_sum = data.sum(axis=1)
        taxa_all_sum = data.sum(axis=0)
        for i in taxa:
            y = data[i]
            taxa_i_sum = taxa_all_sum.iloc[i]
            x_cov = taxa_i_sum / total_sum
            x_cov = np.nan_to_num(x_cov, nan=0, posinf=0.9, neginf=0)
            taxa_i_bar = data[i].mean()
            x_dep = data[i] / taxa_i_bar
            x_dep = np.nan_to_num(x_dep, nan=0, posinf=10, neginf=0)
            x = pd.DataFrame({"coverage":x_cov, "deepth":x_dep})
            result = ZeroInflatedPoisson(y, x).fit(method='lbfgs')

            # calculate expectation
            expectation = result.predict()
            expected_df[i] = expectation

        # return the expectation dataframe
        self.expected = expected_df


#####################discrete matrix#############################
# Discretize the original count matrix into binary using the expected value as the cutoff
def disbinary(counts: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    samples = counts.index.to_list()
    print("Total samples of your counts table is ", len(samples))
    taxa = counts.columns.to_list()
    print("Total taxa of your counts table is ", len(taxa))

    dis_df = pd.DataFrame(np.zeros_like(counts.to_numpy()), index=samples, columns=taxa)
    for i in samples:
        for j in taxa:
            if counts.loc[i, j] < expected.loc[i, j]:
                dis_df.loc[i, j] = 0
            else:
                dis_df.loc[i, j] = 1
    return dis_df

# Using the expected value as the cutoff, set the value to 1 when the condition is met, otherwise, it remains 0
def disbinary2(expected: pd.DataFrame) -> pd.DataFrame:
    samples = expected.index.to_list()
    print("Total samples of your counts table is ", len(samples))
    taxa = expected.columns.to_list()
    print("Total taxa of your counts table is ", len(taxa))

    dis_df = pd.DataFrame(np.zeros_like(expected.to_numpy()), index=samples, columns=taxa)
    for i in samples:
        for j in taxa:
            if expected.loc[i, j] < np.mean(expected[j]):
                dis_df.loc[i, j] = 0
            else:
                dis_df.loc[i, j] = 1
    return dis_df



####################mutual information matrix##############################
# Calculate the taxon-taxon mutual information matrix based on the discretized binary matrix
def taxa_mi(dis_df) -> pd.DataFrame:
    taxa = dis_df.columns.to_list()
    n = len(taxa)
    dis_mat = dis_df.values

    mi_mat = np.zeros((n, n))
    from sklearn.metrics import mutual_info_score
    for i in range(n):
        x = dis_mat[:, i]
        for j in range(n):
            y = dis_mat[:, j]
            mi = mutual_info_score(x, y)
            mi_mat[i, j] = mi
    mi_df = pd.DataFrame(mi_mat, index=taxa, columns=taxa)

    return mi_df


####################adjacent matrix 1##############################
# Infer the adjacency matrix based on the mutual information matrix
def adj_mat(mi_df, cutmet='mean') -> pd.DataFrame:
    if cutmet == 'mean':
        cutoff = mi_df.mean(axis=1).mean()
    elif cutmet == 'median':
        cutoff = mi_df.median(axis=1).mean()
    
    taxa = mi_df.columns.to_list()
    n = len(taxa)

    adj_mat0 = pd.DataFrame(np.zeros((n,n), dtype=int), index=taxa, columns=taxa)
    for i in range(n):
        for j in range(n):
            if mi_df.iloc[i,j] < cutoff:
                adj_mat0.iloc[i,j] = 0
            else:
                adj_mat0.iloc[i,j] = 1
    return adj_mat0

def adj_spe_mat(mi_df, cutmet='mean') -> pd.DataFrame:
    if cutmet == 'quant':
        cutoff_list = list(mi_df.quantile(0.75, axis=1))
    elif cutmet == 'mean':
        cutoff_list = list(mi_df.mean(axis=1))
    elif cutmet == 'median':
        cutoff_list = list(mi_df.median(axis=1))
    
    taxa = mi_df.columns.to_list()
    n = len(taxa)

    adj_mat0 = pd.DataFrame(np.zeros((n,n), dtype=int), index=taxa, columns=taxa)
    for i in range(n):
        for j in range(n):
            if mi_df.iloc[i,j] < cutoff_list[i]:
                adj_mat0.iloc[i,j] = 0
            else:
                adj_mat0.iloc[i,j] = 1
    return adj_mat0

####################adjacent matrix 2#############################
# Infer the adjacency matrix based on covariance
def cov_cut(cov_matrix, mul=1.5):
    data = np.abs(cov_matrix)
    np.fill_diagonal(data, 0)
    
    std_dev = np.std(data, axis=1)
    std_multiplier = mul
    # Calculate the threshold
    threshold = std_dev * std_multiplier
    
    n = data.shape[0]
    adj_mat = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if data[i, j] > threshold[j]:
                adj_mat[i, j] = 1
    return adj_mat