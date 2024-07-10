# MNBMIMRF  
## 基于互信息和马尔科夫随机场的微生物网络推断方法  
## A Microbial Network Inference Approach Based on Mutual Information and Markov Random Fields  
版本 version: v1.01  
更新时间 update: 2024/07/08  

## 安装 Installation  
请将MNBMIMRF相关文件置于一个conda环境中（python 3.7+）  
Please put MNBMIMRF relevant files into a conda environment (python 3.7+).  
依赖包 Dependancies: `numpy`, `pandas`, `sklearn`.  

## 用法示例 Usage example  
导入示例数据  
```
# load example data
data_mat = np.array([[1, 0, 3, 1, 5],
                 [4, 0, 6, 3, 1],
                 [0, 8, 3, 9, 0],
                 [5, 0, 11, 12, 3],
                 [3, 0, 15, 8, 2]])

taxa = ['taxa{}'.format(i) for i in range(1, 6)]
samples = ['S{}'.format(i) for i in range(1, 6)]
df = pd.DataFrame(data_mat, index=samples, columns=taxa)
```

开始微生物网络推断  
```
from MNBMIMRF.Dependency import FitModels,disbinary,taxa_mi,adj_spe_mat,cov_cut
from MNBMIMRF.MIbMRF import MRFforMN,max_mi

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
