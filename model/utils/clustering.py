# -*- coding: utf-8 -*-
'''
@Time    : 2025/3/18 20:11
@Author  : Linjie Wang
@FileName: clustering.py
@Software: PyCharm
'''
import numpy as np
from sklearn.preprocessing import StandardScaler

def mclust_R(adata, num_cluster, scale=False, modelNames='EEE', used_obsm='STAGATE', key_added='mclust', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    if scale:
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(adata.obsm[used_obsm])
    else:
        norm_data = adata.obsm[used_obsm]
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(norm_data), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[key_added] = mclust_res
    adata.obs[key_added] = adata.obs[key_added].astype('int')
    adata.obs[key_added] = adata.obs[key_added].astype('category')
    return adata