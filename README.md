# MicroNet-MIMRF  
## A Microbial Network Inference Approach Based on Mutual Information and Markov Random Fields  
version: v1.02  
update: 2024/08/19  

## Installation   
Please put MicroNet_MIMRF relevant files into a conda environment (python 3.7+).  
Dependancies: `numpy`, `pandas`, `scikit-learn`, `statsmodels`.  

## Usage example  
Loading example data.    
```
# import dependancies  
import numpy as np
import pandas as pd

# generate example data
data_mat = np.array([[13,0,27,0,13], 
                     [40,27,9,0,0], 
                     [13,0,13,13,27], 
                     [27,13,0,11,0], 
                     [7,13,0,7,0], 
                     [10,54,0,0,27]])

taxa = ['taxa{}'.format(i) for i in range(1, 6)]
samples = ['S{}'.format(i) for i in range(1, 7)]

# for our tool, the input file should be consist from n rows' samples and p columns' taxa
df = pd.DataFrame(data_mat, index=samples, columns=taxa)
```

Microbial networks inference process.  
```
# import MicroNet_MIMRF tool
from MicroNet_MIMRF.Dependency import FitModels,disbinary,taxa_mi,adj_spe_mat,cov_cut
from MicroNet_MIMRF.MIMRF import MRFforMN,max_mi

# data discretization
fit_result = FitModels(df, method='zip2')
expected_df = fit_result.expected
dis_df = disbinary(df, expected_df)

# dependancy inference
dis_mat = dis_df.values
cov_matrix = np.cov(dis_mat.T)
cov_bool_matrix = cov_cut(cov_matrix)
cov_bool_df = pd.DataFrame(cov_bool_matrix, index=dis_df.columns.to_list(), columns = dis_df.columns.to_list())
s = [0, 1]
mrf = MRFforMN(states=s, dis_df=dis_df)
mrf_sa = mrf.inference(MMI=max_mi, opMethod="SA")
mrf_df = pd.DataFrame(mrf_sa, index=dis_df.index.to_list(), columns=dis_df.columns.to_list())
mi_ori_df = taxa_mi(dis_df=mrf_df)
mi_bool_df = adj_spe_mat(mi_df=mi_ori_df)

# adjacent matrix for a microbial network
adj_final_df = cov_bool_df & mi_bool_df
print(adj_final_df)
```
  
---  

## 基于互信息和马尔科夫随机场的微生物网络推断方法  
版本：v1.02  
更新时间：2024/08/20    

## 安装    
请将 MicroNet_MIMRF 文件夹及其相关文件置于一个conda环境中（python 3.7+）。    
依赖包： `numpy`, `pandas`, `scikit-learn`, `statsmodels`.   

## 用法示例    
导入示例数据  
```
# 导入依赖包    
import numpy as np
import pandas as pd

# 生成示例数据
data_mat = np.array([[13,0,27,0,13], 
                     [40,27,9,0,0], 
                     [13,0,13,13,27], 
                     [27,13,0,11,0], 
                     [7,13,0,7,0], 
                     [10,54,0,0,27]])

taxa = ['taxa{}'.format(i) for i in range(1, 6)]
samples = ['S{}'.format(i) for i in range(1, 7)]

# 输入文件应由n行样本和p列物种分类群组成
df = pd.DataFrame(data_mat, index=samples, columns=taxa)
```

开始微生物网络推断    
```
# 导入 MicroNet_MIMRF 工具
from MicroNet_MIMRF.Dependency import FitModels,disbinary,taxa_mi,adj_spe_mat,cov_cut
from MicroNet_MIMRF.MIMRF import MRFforMN,max_mi

# 数据离散化
fit_result = FitModels(df, method='zip2')
expected_df = fit_result.expected
dis_df = disbinary(df, expected_df)

# 依赖关系推断
dis_mat = dis_df.values
cov_matrix = np.cov(dis_mat.T)
cov_bool_matrix = cov_cut(cov_matrix)
cov_bool_df = pd.DataFrame(cov_bool_matrix, index=dis_df.columns.to_list(), columns = dis_df.columns.to_list())
s = [0, 1]
mrf = MRFforMN(states=s, dis_df=dis_df)
mrf_sa = mrf.inference(MMI=max_mi, opMethod="SA")
mrf_df = pd.DataFrame(mrf_sa, index=dis_df.index.to_list(), columns=dis_df.columns.to_list())
mi_ori_df = taxa_mi(dis_df=mrf_df)
mi_bool_df = adj_spe_mat(mi_df=mi_ori_df)

# 生成微生物网络的邻接矩阵
adj_final_df = cov_bool_df & mi_bool_df
print(adj_final_df)
```
