import pickle
import os
from IPython.display import display
from numpy.core.fromnumeric import var
from numpy.lib.function_base import disp
import pandas as pd
import numpy as np

path0 = os.getcwd()
os.chdir(os.path.join(path0,'5_application/results_OMH'))

facet=[2,3]
exp = 2
types = ['depth','trees']
nums = ['all']

for i in range(6):
    exp = i+1
    m=0
    for j in facet:
        facets=j
        for k in types:
            type2=k
            if (facets==2):
                type2='seed'
            for n in nums:
                numobs = n
                if (facets==2):
                    pathname = 'exp'+str(exp)+'_results_'+str(facets)+'_seed_'+numobs
                else:
                    pathname = 'exp'+str(exp)+'_results_'+str(facets)+'_seed_'+type2+'_'+numobs

                
                path = os.getcwd()
                newpath = os.path.join(path,pathname)
                os.chdir(newpath)
                display(os.getcwd())
                files = os.listdir(newpath)
                for p in files:
                    if('omh_var_components' in p):
                        with open(p,'rb') as file:
                            data = pickle.load(file)
                        if (facets==2):
                            n_p,n_s = data.shape
                            outcome = np.zeros(shape=(1000,n_s))
                            cors = np.zeros(shape=(n_s))
                            for i in range(n_s):
                                thresh = -np.sort(-data[:,i])[1000]
                                idx = np.where((data[:,i]>thresh)==True)
                                outcome[:,i] = idx[0]
                                cors[i] = np.sum((idx[0]<=1000))/1000
                            display(np.mean(cors))
                        elif (facets==3):
                            n_p,n_s,n_d = data.shape
                            outcome = np.zeros(shape=(1000,n_s,n_d))
                            cors = np.zeros(shape=(n_s,n_d))
                            for i in range(n_s):
                                for j in range(n_d):
                                    thresh = -np.sort(-data[:,i,j])[1000]
                                    idx = np.where((data[:,i,j]>thresh)==True)
                                    outcome[:,i,j] = idx[0]
                                    cors[i,j] = np.sum((idx[0]<=1000))/1000
                            display(np.mean(cors))

                m+=1
                os.chdir(path)

            df = pd.DataFrame(data=cors)
            df.to_excel('results_cors_exp'+str(exp)+'_'+str(facets)+'_facets_'+str(type2)+'.xlsx')
            with open('cors_exp'+str(exp)+'_'+str(facets)+'_facets'+str(type2)+'.pickle','wb') as file:
                pickle.dump(cors,file)




