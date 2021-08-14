
from IPython.display import display
from numpy.lib.function_base import disp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import t
from tqdm import tqdm
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle

def rmse(predictions,target,axis=0):
    rmse = np.sqrt(np.mean((predictions-target)**2,axis=axis))
    return rmse

def normalize(vec):
    new_vec = [vec[i]/np.sum(vec) for i in range(len(vec))]
    return new_vec

type = 'di'
numsims=[250,500,1000]
names = ['mu','sigma2_p','sigma2_s','sigm2_d','sigma2_ps','sigma2_pd','sigma2_ds','sigma2_pds']
means = np.zeros(shape=(1,len(names)))
stds = np.zeros(shape=(1,len(names)))
mins = np.zeros(shape=(1,len(names)))
maxs = np.zeros(shape=(1,len(names)))
true = np.array([0,4,8,16,5,5,5,2])
normed_true = normalize(true)
results = np.zeros(shape=(16,8))

for j in range(len(numsims)):
    N = numsims[j]
    data = np.zeros(shape=(1000,8))
    
    import os
    
    path = os.getcwd()
    newpath=os.path.join(path,'4_mc1\output_di_'+str(N)+'_1000')
    os.chdir(newpath)
    files = [i for i in os.listdir(newpath) if os.path.isfile(os.path.join(newpath,i)) and 'output_'+type+'_3' in i]
    for i in range(len(files)):
        data[i,:] = pd.read_excel(
            os.path.join(newpath, files[i]),
            engine='openpyxl')
    if (type=='di'):
        normed_data = np.array([normalize(data[i,:]) for i in range(data.shape[0])])
        results[j,:] = np.mean(normed_data,axis=0)
        results[j+3,:] = np.mean(np.abs(normed_data-normed_true),axis=0)
        results[j+6,:] = np.std(normed_data,axis=0)
        results[j+9,:] = rmse(normed_data,normed_true)
    else:
        results[j,:] = np.mean(data,axis=0)
        results[j+3,:] = np.mean(np.abs(data-true),axis=0)
        results[j+6,:] = np.std(data,axis=0)
        results[j+9,:] = rmse(data,true)
    os.chdir(path)
df = pd.DataFrame(data=results)
df.to_excel('sims_'+type+'3facets.xlsx')


# for l in range(len(numsims)):
#     num = numsims[l]
#     with open("Var_reg2_"+str(num)+"_"+str(2)+".pickle","rb") as file:
#         data = pickle.load(file)
#         current_path = os.path.abspath(os.getcwd())
#     for i in range(data.shape[1]):
#         means[l,i]=np.mean(data[:,i])
#         stds[l,i] = np.std(data[:,i])
#         mins[l,i]= np.min(data[:,i])
#         maxs[l,i] = np.max(data[:,i])
# pd.DataFrame(means).to_excel("means2.xlsx")
# pd.DataFrame(stds).to_excel("stds2.xlsx")
# pd.DataFrame(mins).to_excel("mins2.xlsx")
# pd.DataFrame(maxs).to_excel("maxs2.xlsx")


