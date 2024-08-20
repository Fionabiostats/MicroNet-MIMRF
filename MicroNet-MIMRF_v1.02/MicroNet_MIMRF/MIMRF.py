"""
Created on Mar 20, 2024
@author: FionaFeng

(M)utual (I)nformation based (M)arkov (R)andom (F)eild

MIMRF is used to infer a pairwise association between taxon and taxon, then we can estabulish a micrbial network which its vertexes are taxa and its edges are pairwise dependencies. 

"""

import numpy as np
import pandas as pd
from statistics import mean

###################################################################
# Calculate the maximum pairwise mutual information
def max_mi(dis_mat: np.array, col: int):
    """
    dis_df

    {X_0}     {X_j}     {X_p}
    x_01------x_0j------x_0p
    |---------------------|
    |---------------------|
    x_i1------x_ij------x_ip
    |---------------------|
    |---------------------|
    x_n1------x_nj------x_np
      
    """    

    n = dis_mat.shape[0]
    p = dis_mat.shape[1]

    from sklearn.metrics import mutual_info_score
    x = dis_mat[:, col]
    x_mi = [mutual_info_score(x, dis_mat[:, j]) for j in range(p) if j != col]
    
    mi_max = max(x_mi)
    return mi_max

# Calculate the minimum mutual information for non-zero values
def min_mi(dis_mat: np.array, col: int):
    p = dis_mat.shape[1]

    from sklearn.metrics import mutual_info_score
    x = dis_mat[:, col]
    x_mi = [mutual_info_score(x, dis_mat[:, j]) for j in range(p) if j != col]
    xmi_nozeros = np.where(x_mi == 0, np.inf, x_mi)
    mi_min = min(xmi_nozeros)
    return mi_min


# Calculate the zero-inflation rate
def zir_cal(dis_mat: np.array):
    t = dis_mat.size
    non_zeros = np.count_nonzero(dis_mat)
    zir = round(((t-non_zeros)/t), 2)
    return zir

# Calculate the random probability
def random_p():
    p = np.random.rand()
    while p < 0.5:
        p = np.random.rand()
    return p

################################################################
class MRFforMN:
    def __init__(self, states:list, dis_df:pd.DataFrame):
        self.states = states
        self.dis_df = dis_df
        self.mrf = self.dis_df.values
        self.zir = zir_cal(self.mrf)

        # Other properties of state parameters
        self.nstates = len(states)
        self.maxstate = max(states)
        self.minstate = min(states)

        # Other properties of mrf
        self.shmrf = self.mrf.shape
    
    def transState(self, value):
            os = value
            while True:
                ns = np.random.randint(0, self.maxstate+1, size=self.shmrf[0])
                if np.array_equal(ns, os):
                    ns = np.random.randint(0, self.maxstate+1, size=self.shmrf[0])
                else:
                    return ns

    def transState_zir(self, value):
        os = value
        while True:
            ns = np.random.choice([0, 1], size=self.shmrf[0], p=[self.zir, 1-self.zir])
            if np.array_equal(ns, os):
                ns = np.random.randint(0, self.maxstate+1, size=self.shmrf[0])
            else:
                return ns
    

    """
    simAnnealing: Finding optimized paths in microbial networks
    """
    def simAnnealing(self, MMI=max_mi, maxIterations=30, Temp=100, k=0.9):
        omrf = self.mrf.copy()
        final_it = 0
        delta_o_list = []
        delta_n_list = []

        while Temp > 0:
            if Temp < 1:
                Temp = 0
            else:
                Temp = Temp
            # Begin iterative calculation
            for it in range(maxIterations):
                delta_o_list.clear()
                delta_n_list.clear()
                final_it += 1
                for j in range(self.shmrf[1]):  # column
                    maxmi_o = MMI(self.mrf, j)  # original max mutual information
                    minmi_o = min_mi(self.mrf, j)
                    delta_o = maxmi_o - minmi_o
                    delta_o_list.append(delta_o)
                    ostate = self.mrf[:, j]  # original state
                    # new states
                    nstate = self.transState(self.mrf[:, j])  # trans the state of the original value
                    self.mrf[:, j] = nstate
                    maxmi_n = MMI(self.mrf, j)  # transed max mutual information
                    minmi_n = min_mi(self.mrf, j)
                    delta_n = maxmi_n - minmi_n
                    delta_n_list.append(delta_n)

                    # Determine if the maximum mutual information has been reached
                    if maxmi_n > maxmi_o:
                        self.mrf[:, j] = nstate
                    else:
                        d = (maxmi_n - maxmi_o) * 100
                        pt = np.exp(d / Temp)  # probability of keeping the value with lower max mutual information
                        pc = random_p()
                        if pc > pt:
                            self.mrf[:, j] = ostate
                
                if np.all(omrf == self.mrf):  # if there are no change, then finished process.
                    print("Successfully finished, iteration: " + str(final_it))
                    break
                else:
                    omrf = self.mrf.copy()

            # Determine if the output condition has been reached
            if mean(delta_n_list) > mean(delta_o_list):
                print("Successfully finished, iteration: " + str(final_it))
                return self.mrf
            else:
                # Decrease the Temp
                Temp *= k
                # Reset the iteration count, reducing the iteration count as the Temp decreases
                maxIterations = max(5, round(maxIterations * k))
                print("Temperature reduced to:", Temp)

        print("Temperature reached 0. Finished all iterations. Iteration: " + str(final_it))
        return self.mrf

    def simAnnealing_zir(self, MMI=max_mi, maxIterations=30, Temp=100, k=0.9):
        omrf = self.mrf.copy()
        final_it = 0
        delta_o_list = []
        delta_n_list = []

        while Temp > 0:
            for it in range(maxIterations):
                delta_o_list.clear()
                delta_n_list.clear()
                final_it += 1
                for j in range(self.shmrf[1]):  # column
                    maxmi_o = MMI(self.mrf, j)  # original max mutual information
                    minmi_o = min_mi(self.mrf, j)
                    delta_o = maxmi_o - minmi_o
                    delta_o_list.append(delta_o)
                    ostate = self.mrf[:, j]  # original state
                    # new states
                    nstate = self.transState_zir(self.mrf[:, j])  # trans the state of the original value
                    self.mrf[:, j] = nstate
                    maxmi_n = MMI(self.mrf, j)  # transed max mutual information
                    minmi_n = min_mi(self.mrf, j)
                    delta_n = maxmi_n - minmi_n
                    delta_n_list.append(delta_n)

                    # Determine if the maximum mutual information has been reached
                    if maxmi_n > maxmi_o:
                        self.mrf[:, j] = nstate
                    else:
                        d = (maxmi_n - maxmi_o) * 100
                        pt = np.exp(d / Temp)  # probability of keeping the value with lower max mutual information
                        pc = random_p()
                        if pc > pt:
                            self.mrf[:, j] = ostate
                
                if np.all(omrf == self.mrf):  # if there are no change, then finished process.
                    print("Successfully finished, iteration: " + str(final_it))
                    break
                else:
                    omrf = self.mrf.copy()

            # Determine if the maximum mutual information has been reached
            if mean(delta_n_list) > mean(delta_o_list):
                print("Successfully finished, iteration: " + str(final_it))
                return self.mrf
            else:
                # Decrease the Temp
                Temp *= k
                # Reset the iteration count, reducing the iteration count as the Temp decreases
                maxIterations = max(5, round(maxIterations * k))
                print("Temperature reduced to:", Temp)

        print("Temperature reached 0. Finished all iterations. Iteration: " + str(final_it))
        return self.mrf

    ##################### Microbial networks inference ###########################
    def inference(self, MMI = max_mi, opMethod = "SA", maxIterations=30, Temp=100, k=0.9):
        omrf = self.mrf.copy()

        if (opMethod == "SA"):
            return self.simAnnealing(MMI=MMI, maxIterations=maxIterations, Temp=Temp, k=k)
        elif (opMethod == "SA_zir"):
            return self.simAnnealing_zir(MMI=MMI, maxIterations=maxIterations, Temp=Temp, k=k)
        else:
            raise TypeError("opMethod must be 'SA' or 'SA_zir' !")