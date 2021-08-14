import sys
import os
import json
from IPython.display import display

import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import time
import pickle

notebook_path = os.path.split(os.path.abspath(os.getcwd()))[0]
os.chdir(notebook_path)

def rmse(predictions,target,axis=0):
    rmse = np.sqrt(np.mean((predictions-target)**2,axis=axis))
    return rmse

def normalize(vec):
    new_vec = [vec[i]/np.sum(vec) for i in range(len(vec))]
    return new_vec

def gen_rnd_sample_2(n_p,n_s,mu,sig2_p,sig2_s,sig2_ps):
    sd_p = np.sqrt(sig2_p)
    sd_s = np.sqrt(sig2_s)
    sd_ps = np.sqrt(sig2_ps)
    z_p = np.random.standard_normal(size=(n_p,1))
    z_s = np.random.standard_normal(size=(1,n_s))
    z_ps = np.random.standard_normal(size=(n_p,n_s))
    results = mu + sd_p*z_p+sd_s*z_s+sd_ps*z_ps
    return results

def gen_rnd_sample_3(n_p,n_s,n_d,mu,sig2_p,sig2_s,sig2_d,sig2_ps,sig2_pd,sig2_ds,sig2_pds):
    sd_p = np.sqrt(sig2_p);sd_s = np.sqrt(sig2_s);sd_d = np.sqrt(sig2_d);sd_ps=np.sqrt(sig2_ps)
    sd_pd=np.sqrt(sig2_pd);sd_ds=np.sqrt(sig2_ds);sd_pds=np.sqrt(sig2_pds)
    z_p = np.random.standard_normal(size=(n_p,1,1))
    z_s = np.random.standard_normal(size=(1,n_s,1))
    z_d = np.random.standard_normal(size=(1,1,n_d))
    z_ps = np.random.standard_normal(size=(n_p,n_s,1))
    z_pd = np.random.standard_normal(size=(n_p,1,n_d))
    z_ds = np.random.standard_normal(size=(1,n_s,n_d))
    z_pds = np.random.standard_normal(size=(n_p,n_s,n_d))
    results = mu + sd_p*z_p+sd_s*z_s+sd_d*z_d+sd_ps*z_ps+sd_pd*z_pd+sd_ds*z_ds+sd_pds*z_pds
    
    return results

def compute_components_2(results):
    n_p = results.shape[0]
    n_s = results.shape[1]
    X_p = np.mean(results,axis=1) # depth mean
    X_s = np.mean(results,axis=0) # seed mean

    X_tot = np.mean(results,axis=(0,1)) # universe mean

    SS_p = n_s*np.sum((X_p-X_tot)**2)
    MS_p = SS_p/(n_p-1)
    SS_s = n_p*np.sum((X_s-X_tot)**2)
    MS_s = SS_s/(n_s-1)
    SS_ps = np.zeros(shape=(n_p,n_s))
    for i in range(n_p):
        for j in range(n_s):
            SS_ps[i,j] = (results[i,j] - X_p[i] - X_s[j] + X_tot)**2

    SS_ps = np.sum(SS_ps,axis=(0,1))
    MS_ps = SS_ps/((n_p-1)*(n_s-1))
    sigma2_p = (MS_p-MS_ps)/n_s
    sigma2_s = (MS_s-MS_ps)/n_p
    sigma2_ps = MS_ps


    df = pd.DataFrame([sigma2_p,sigma2_s,
                    sigma2_ps],)
    df.index=['p','s','ps']

    var_matrix = np.array([sigma2_p,sigma2_s,sigma2_ps])
    var_matrix[var_matrix<0] = 0 # set each negative variance to 0

    df_0 = pd.DataFrame(var_matrix)
    df_0.index=['p','s','ps']

    return df,df_0,var_matrix

def compute_components_3(results):
    n_p = results.shape[0];n_s=results.shape[1];n_d=results.shape[2]
    X_p = np.mean(results,axis=(1,2)) # person mean
    X_d = np.mean(results,axis=(0,1)) # depth mean
    X_s = np.mean(results,axis=(0,2)) # seed mean

    X_pd = np.mean(results,axis=1) # person x depth mean
    X_ps = np.mean(results,axis=2) # person x seed mean
    X_ds = np.mean(results,axis=0) # depth x seed mean

    X_tot = np.mean(results,axis=(0,1,2)).reshape(-1,1) # (observable) universe mean

    SS_p = (n_d*n_s*np.sum((X_p-X_tot)**2,axis=1))[0]
    SS_d = (n_p*n_s*np.sum((X_d-X_tot)**2,axis=1))[0]
    SS_s = (n_p*n_d*np.sum((X_s-X_tot)**2,axis=1))[0]

    SS_pd = np.zeros(shape=(n_p,n_d))
    SS_ps = np.zeros(shape=(n_p,n_s))
    SS_ds = np.zeros(shape=(n_s,n_d))

    SS_pds = np.zeros(shape=(n_p,n_s,n_d))

    for i in range(n_s):
        for j in range(n_d):
            SS_ds[i,j] = (X_ds[i,j].ravel()-X_d[j].ravel()-X_s[i].ravel()+X_tot.ravel())**2
            for k in range(n_p):
                SS_pd[k,j] = (X_pd[k,j].ravel()-X_d[j].ravel()-X_p[k].ravel()+X_tot.ravel())**2
                SS_pds[k,i,j] = (results[k,i,j].ravel()-X_ds[i,j].ravel() - X_pd[k,j].ravel()-X_ps[k,i].ravel()
                                +X_d[j].ravel()+X_s[i].ravel()+X_p[k].ravel()-X_tot.ravel())**2
        for k in range(n_p):
            SS_ps[k,i] = (X_ps[k,i].ravel()-X_s[i].ravel()-X_p[k].ravel()+X_tot.ravel())**2
            
    SS_ds = n_p*np.sum(SS_ds,axis=(0,1))
    SS_pd = n_s*np.sum(SS_pd,axis=(0,1))
    SS_ps = n_d*np.sum(SS_ps,axis=(0,1))
    SS_pds = np.sum(SS_pds,axis=(0,1,2))


    ## Degrees of freedom
    df_d = n_d-1
    df_s = n_s-1
    df_p = n_p-1
    df_ds = df_d*df_s
    df_pd = df_d*df_p
    df_ps = df_s*df_p
    df_pds = df_d*df_s*df_p


    # The mean squares (MS) are simply the sum of squares divided by it's degrees of freedom.
    MS_d = SS_d/df_d
    MS_s = SS_s/df_s
    MS_p = SS_p/df_p
    MS_ds = SS_ds/df_ds
    MS_pd = SS_pd/df_pd
    MS_ps = SS_ps/df_ps
    MS_pds = SS_pds/df_pds


    # The variance is then computed from the mean squares (MS) according to the following manner. It is here explained for a number $j$ of possible facets in order to show that it generalizes easiliy. Consdider $\alpha$ the set of indices which are contained in a facet, $\dot{\alpha}$ (notice the dot) for the set of indices not contained in the facet and $\omega$ the set of possible indices. For example in $\bar{X}_{ds}$, we would have $\alpha = \{d,s\}$, $\dot{\alpha} = \{p\}$ and $\omega = \{d,s,p\}$. The variance is computed as: $\frac{MS(\alpha)-\sum_j MS(\alpha;\dot{\alpha}_j) + \sum_i MS(\alpha;\dot{\alpha}_i) }{ \pi(\alpha)}$. Here, the index $j$ sums over all MS that contain the terms in $\alpha$ and an odd number of terms from $\dot{\alpha}$. On the other hand, $i$ sums over all MS containing the terms in $\alpha$ and an even number of terms from $\dot{\alpha}$. $\pi(\alpha) = \prod_{i \in \dot{\alpha}} n_i$. So for example for the variance for facet $d$ we have $\hat{\sigma}^2(d) = \frac{MS(d)-MS(ds)-MS(pd)+MS(pds)}{n_s n}$. 
    sigma2_d = (MS_d-MS_ds-MS_pd+MS_pds)/(n_s*n_p)
    sigma2_s = (MS_s-MS_ds-MS_ps+MS_pds)/(n_d*n_p)
    sigma2_p = (MS_p-MS_pd-MS_ps+MS_pds)/(n_d*n_s)
    sigma2_ds = (MS_ds-MS_pds)/n_p
    sigma2_pd = (MS_pd-MS_pds)/n_s
    sigma2_ps = (MS_ps-MS_pds)/n_d
    sigma2_pds = MS_pds

    df = pd.DataFrame([sigma2_p,sigma2_s,
                    sigma2_d,sigma2_ps,
                    sigma2_pd,sigma2_ds,
                    sigma2_pds],)
    df.index=['p','s','d','ps','pd','ds','pds']

    # ### Negative variances ###
    # Due to the construction of the variance components from the MS some variance components can turn out to become negative. This has also been discussed by Cronbach, Shavelson and Brennan. The best approach according to the literature is to set the negative estimates to zero due to sampling variability. This will not affect the other variance estimates, but it will influence the generalizability coefficient. The table above shows that the variance which are negative are in fact very small. The table below shows the mean variances for each facet when setting negative variances to 0.
    var_matrix = np.array([sigma2_p,sigma2_s,sigma2_d,sigma2_ps,sigma2_pd,sigma2_ds,sigma2_pds])
    var_matrix[var_matrix<0] = 0 # set each negative variance to 0

    sigma2_p_hat = var_matrix[0]
    sigma2_s_hat = var_matrix[1]
    sigma2_d_hat = var_matrix[2]
    sigma2_ps_hat = var_matrix[3]
    sigma2_pd_hat = var_matrix[4]
    sigma2_ds_hat = var_matrix[5]
    sigma2_pds_hat = var_matrix[6]

    df_0 = pd.DataFrame([sigma2_p_hat,sigma2_s_hat,
                    sigma2_d_hat,sigma2_ps_hat,
                    sigma2_pd_hat,sigma2_ds_hat,
                    sigma2_pds_hat],)
    df_0.index=['p','s','d','ps','pd','ds','pds']

    return df,df_0,var_matrix

def gen_rnd_sample_di_2(n_p,n_s,mu,sig2_p,sig2_s,sig2_ps):
    sd_p = np.sqrt(sig2_p)
    sd_s = np.sqrt(sig2_s)
    sd_ps = np.sqrt(sig2_ps)
    z_p = np.random.standard_normal(size=(n_p,1))
    z_s = np.random.standard_normal(size=(1,n_s))
    z_ps = np.random.standard_normal(size=(n_p,n_s))
    y = mu + sd_p*z_p+sd_s*z_s+sd_ps*z_ps
    results = 1/(1+np.exp(-y))
    return results

def gen_rnd_sample_di_3(n_p,n_s,n_d,mu,sig2_p,sig2_s,sig2_d,sig2_ps,sig2_pd,sig2_ds,sig2_pds):
    sd_p = np.sqrt(sig2_p);sd_s = np.sqrt(sig2_s);sd_d = np.sqrt(sig2_d);sd_ps=np.sqrt(sig2_ps)
    sd_pd=np.sqrt(sig2_pd);sd_ds=np.sqrt(sig2_ds);sd_pds=np.sqrt(sig2_pds)
    z_p = np.random.standard_normal(size=(n_p,1,1))
    z_s = np.random.standard_normal(size=(1,n_s,1))
    z_d = np.random.standard_normal(size=(1,1,n_d))
    z_ps = np.random.standard_normal(size=(n_p,n_s,1))
    z_pd = np.random.standard_normal(size=(n_p,1,n_d))
    z_ds = np.random.standard_normal(size=(1,n_s,n_d))
    z_pds = np.random.standard_normal(size=(n_p,n_s,n_d))
    y = mu + sd_p*z_p+sd_s*z_s+sd_d*z_d+sd_ps*z_ps+sd_pd*z_pd+sd_ds*z_ds+sd_pds*z_pds
    results = 1/(1+np.exp(-y))
    return results

def run_1_sim2(n_p,n_s,sig2):
    mu = sig2[0];sig2_p=sig2[1];sig2_s=sig2[2];sig2_ps=sig2[3]
    results = gen_rnd_sample_2(n_p,n_s,mu,sig2_p,sig2_s,sig2_ps)
    _,_,var_matrix = compute_components_2(results)
    return var_matrix

def run_sims2(N,n_p,n_s=100,sig2 = np.array([0,4,4,2])):
    sim_res = np.zeros((N,3))
    for i in tqdm(range(N)):
        sim_res[i,] = run_1_sim2(n_p,n_s,sig2)
    return sim_res

def run_1_sim(n_p,n_s,n_d,sig2):
    mu = sig2[0];sig2_p=sig2[1];sig2_s=sig2[2];sig2_d=sig2[3]
    sig2_ps =sig2[4];sig2_pd=sig2[5];sig2_ds=sig2[6];sig2_pds=sig2[7]
    results = gen_rnd_sample_3(n_p,n_s,n_d,mu,sig2_p,sig2_s,sig2_d,sig2_ps,sig2_pd,sig2_ds,sig2_pds)
    _,_,var_matrix = compute_components_3(results)
    return var_matrix

def run_sims(N,n_p,n_s=10,n_d=10,sig2=np.array([0,4,8,16,5,5,5,2])):
    sim_res = np.zeros((N,7))
    for i in tqdm(range(N)):
        sim_res[i,] = run_1_sim(n_p,n_s,n_d,sig2)
    return sim_res

def save_results(sim_res):
    results_dir = os.path.join(notebook_path,"projects/results")
    os.chdir(results_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory = "results_"+str(timestr)
    path = os.path.join(results_dir, directory)
    os.mkdir(path)
    os.chdir(path)
    with open("sim_gtheory_"+str(timestr)+".pckl","wb") as file:
        pickle.dump(sim_res,file)  

def run_1_di_sim2(n_p,n_s,sig2):
    mu = sig2[0];sig2_p=sig2[1];sig2_s=sig2[2];sig2_ps=sig2[3]
    results = gen_rnd_sample_di_2(n_p,n_s,mu,sig2_p,sig2_s,sig2_ps)
    _,_,var_matrix = compute_components_2(results)
    
    return var_matrix

def run_1_di_sim3(n_p,n_s,n_d,sig2):
    mu = sig2[0];sig2_p=sig2[1];sig2_s=sig2[2];sig2_d=sig2[3]
    sig2_ps =sig2[4];sig2_pd=sig2[5];sig2_ds=sig2[6];sig2_pds=sig2[7]
    results = gen_rnd_sample_di_3(n_p,n_s,n_d,mu,sig2_p,sig2_s,sig2_d,sig2_ps,sig2_pd,sig2_ds,sig2_pds)
    _,_,var_matrix = compute_components_3(results)
    
    return var_matrix

def run_sims_di_2(N,n_p=1000,n_s=10,sig2 = np.array([0,4,4,2])):
    sig2 = normalize(sig2)
    sim_res = np.zeros((N,3))
    for i in tqdm(range(N)):
        sim_res[i,] = run_1_di_sim2(n_p,n_s,sig2)
    return sim_res

def run_sims_di_3(N,n_p=1000,n_s=10,n_d=10,sig2 = np.array([0,4,8,16,5,5,5,2])):
    sig2 = normalize(sig2)
    sim_res = np.zeros((N,7))
    for i in tqdm(range(N)):
        sim_res[i,] = run_1_di_sim3(n_p,n_s,n_d,sig2)
    return sim_res

def mc_experiment(N,obs_vec = np.array([250,500,1000,2000]),facets=2,type='bi'):
    
    if (facets==2):
        colnames = ['sigma2_p','sigma2_s','sigm2_ps']
        sig2 = np.array([4,4,2])
        if (type=='bi'):
            true = normalize(sig2)
        elif (type=='norm'):
            true=sig2
        data = np.zeros(shape=(len(obs_vec)*4+6,len(sig2)))
        data[0,:] = sig2
        data[1,:] = true
    if (facets==3):
        colnames = ['sigma2_p','sigma2_s','sigm2_d','sigma2_ps','sigma2_pd','sigma2_ds','sigma2_pds']
        sig2 = np.array([4,8,16,5,5,5,2])
        if (type=='bi'):
            true = normalize(sig2)
        elif (type=='norm'):
            true=sig2
        data = np.zeros(shape=(len(obs_vec)*4+6,len(sig2)))
        data[0,:] = sig2
        data[1,:] = true
    
    for i in tqdm(range(len(obs_vec))):
        n_p = obs_vec[i]
        row=i+3
        incr = len(obs_vec)
        if (type=='bi'):
            if (facets==2):
                sim_res = run_sims_di_2(N,n_p=n_p,n_s=n_p)
            elif (facets==3):
                sim_res = run_sims_di_3(N,n_p=n_p,n_s=n_p,n_d=n_p)
            data[row,:] = normalize(np.mean(sim_res,axis=0))
            normed_data = np.array([normalize(sim_res[i,:]) for i in range(sim_res.shape[0])])
            data[row+incr+1,:] = np.mean(np.abs(normed_data-true),axis=0)
            data[row+2*incr+2,:] = np.std(normed_data,axis=0)
            data[row+3*incr+3,:] = rmse(normed_data,true)
        elif(type=='norm'):
            if (facets==2):
                sim_res = run_sims2(N,n_p=n_p,n_s=n_p)
            elif (facets==3):
                sim_res = run_sims(N,n_p=n_p,n_s=n_p,n_d=n_p)
            data[row,:] = np.mean(sim_res,axis=0)
            data[row+incr+1,:] = np.mean(np.abs(sim_res-true),axis=0)
            data[row+2*incr+2,:] = np.std(sim_res,axis=0)
            data[row+3*incr+3,:] = rmse(sim_res,true)
        
    df = pd.DataFrame(data=data)
    df.columns = colnames
    indexnames = ['parameters','normalized parameters','mean']+list(map(str,obs_vec))+['MAE']+list(map(str,obs_vec))+['STD']+list(map(str,obs_vec))+['RMSE']+list(map(str,obs_vec))
    df.index = indexnames
    if (type == 'bi'):
        df.to_excel("output_di_"+str(facets)+".xlsx")  
    elif (type=='norm'):
        df.to_excel("output_norm_"+str(facets)+".xlsx") 

mc_experiment(N=10,facets=3,type='norm')

