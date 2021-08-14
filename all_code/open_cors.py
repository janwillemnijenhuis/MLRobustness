import os
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import seaborn as sns


path = os.getcwd()
os.chdir(os.path.join(path,'5_application\cors'))



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



