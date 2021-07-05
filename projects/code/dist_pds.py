from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import sys
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
sys.path.append('C:/Users/janwi/OneDrive/Documents/stage/')
from code import init_dist_fns

seed_list = np.linspace(100,1000,10,dtype=int)

#depth_list = np.linspace(100,1000,10,dtype=int) ### USE FOR RANDOM FOREST ###
#depth_list = np.linspace(30,50,10,dtype=int) ### USE FOR DECISION TREE ###
depth_list = np.linspace(2,30,10,dtype=int) ### USE FOR DECISION TREE ###

num_obs = [250,500,1000,2000]

for i in tqdm(range(len(num_obs))):
    facets =2 
    N = num_obs[i]
    dist_MS,dist_sig = init_dist_fns.data_sim(1000,N,depth_list,seed_list,regtree=True)
    init_dist_fns.create_plots(N,dist_MS,type="MS",savefig=True,filename="MS_reg_"+str(N))
    init_dist_fns.create_plots(N,dist_sig,savefig=True,filename="Var_reg_"+str(N))
    with open("MS_reg2_"+str(N)+"_"+str(facets)+".pickle","wb") as file:
        pickle.dump(dist_MS,file)
    with open("Var_reg2_"+str(N)+"_"+str(facets)+".pickle","wb") as file:
        pickle.dump(dist_sig,file)
