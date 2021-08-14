import sys
import os
import json
from IPython.display import display
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from notebooks import training_settings
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
import inquirer
from scipy.stats import t
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import time
import pickle
import seaborn as sns






facet=[3]
exp = 2
types = ['depth','trees']
nums = ['all']

for i in range(6):
    exp = i+1
    m=0
    facets=3
    for k in types:
        type2 = k
        
        filename = 'cors_exp'+str(exp)+'_3_facets'+str(type2)
        my_file = '{}.pdf'.format(filename)
        pp = PdfPages(my_file)
        with open('cors_exp'+str(exp)+'_'+str(facets)+'_facets'+str(type2)+'.pickle','rb') as file:
            data = pickle.load(file)
            n_s,n_d = data.shape
        if (type2=='depth'):
            heatmap = sns.heatmap(data,linewidths=0.5,cmap='hot',cbar_kws={'label':'overlap'},
            xticklabels=np.linspace(5,15,n_d,dtype=int),yticklabels=np.linspace(100,1000,n_s,dtype=int))
        elif (type2=='trees'):
            heatmap = sns.heatmap(data,linewidths=0.5,cmap='hot',cbar_kws={'label':'overlap'},
            xticklabels=np.linspace(300,900,n_d,dtype=int),yticklabels=np.linspace(100,1000,n_s,dtype=int))
        plt.xlabel(type2)
        plt.ylabel('seed')
        pp.savefig()
        pp.close()
        plt.close()



