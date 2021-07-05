from sklearn import datasets
import sys
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from IPython import display
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
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

# def create_plots(N,dist_matrix,type="variance",savefig=False,filename="name"):
#     names = ['person','depth','seed','person x depth', 'person x seed', 'depth x seed', 'residual']
#     current_path = os.path.abspath(os.getcwd())
#     my_file = 'results\{}.pdf'.format(filename)
#     if (savefig==True): pp = PdfPages(os.path.join(current_path, my_file))
#     for i in range(7):
#         fig = plt.figure(figsize=(8,4))
#         ax = fig.add_subplot()
#         sns.distplot(dist_matrix[:,i], hist=True, kde=True, 
#                 bins=int(180/5), color = 'darkblue', 
#                 hist_kws={'edgecolor':'black'},
#                 kde_kws={'linewidth': 4},ax=ax)
#         plt.axvline(np.mean(dist_matrix[:,i]),color='red')
#         ax.set_title(names[i]+" "+type+" "+"distribution (N="+str(N)+")")
#         ax.set_ylabel('probability density')
#         ax.set_xlabel(fr'$\sigma^2({names[i]})$')
#         if (savefig==True): 
#             pp.savefig()
#     if (savefig==True): pp.close()
# numsims=[250,500,1000,2000]
# names = ['person','depth','seed','person x depth', 'person x seed', 'depth x seed', 'person x depth x seed']

numsims=[250,500,1000,2000]
# names = ['person','depth','seed','person x depth', 'person x seed', 'depth x seed', 'person x depth x seed']
names = ['person','seed','person x seed']
# for l in range(len(numsims)):
#     num = numsims[l]
#     with open("Var_reg2_"+str(num)+"_"+str(2)+".pickle","rb") as file:
#         data = pickle.load(file)
#         current_path = os.path.abspath(os.getcwd())
#     filename = str(num)+"_var_results"+str(2)
#     my_file = 'results\{}.pdf'.format(filename)
#     pp = PdfPages(os.path.join(current_path, my_file))
#     fig = plt.figure(figsize=(16,4))
#     axs = fig.subplots(nrows=1,ncols=4)
#     fig.set_tight_layout(True)
#     fig.suptitle("Variance distribution (N="+str(num)+")", x=0.9,y=0.5)
#     fig.subplots_adjust(top=0.95)
#     for i in range(data.shape[1]):
#         k=0;j=0
#         if (i>3):k=1;j=i-4
#         else:j=i
#         sns.distplot(data[:,i], hist=True, kde=True, 
#                     bins=int(180/5), color = 'darkblue', 
#                     hist_kws={'edgecolor':'black'},
#                     kde_kws={'linewidth': 4},ax=axs[i])
#         axs[i].axvline(np.mean(data[:,i]),color='red')
#         plt.setp(axs[i].get_xticklabels(), rotation=30, horizontalalignment='right')
#         axs[i].set_ylabel('probability density')
#         axs[i].set_xlabel(fr'$\sigma^2({names[i]})$')
#     fig.delaxes(axs[3])
#     pp.savefig()
#     pp.close()

# for l in range(len(numsims)):
#     num = numsims[l]
#     with open("Var_reg2_"+str(num)+".pickle","rb") as file:
#         data = pickle.load(file)
#         current_path = os.path.abspath(os.getcwd())
#     filename = str(num)+"_var_results"
#     my_file = 'results\{}.pdf'.format(filename)
#     pp = PdfPages(os.path.join(current_path, my_file))
#     fig = plt.figure(figsize=(16,8))
#     axs = fig.subplots(nrows=2,ncols=4)
#     fig.set_tight_layout(True)
#     fig.suptitle("Variance distribution (N="+str(num)+")", x=0.9,y=0.25)
#     fig.subplots_adjust(top=0.95)
#     for i in range(data.shape[1]):
#         k=0;j=0
#         if (i>3):k=1;j=i-4
#         else:j=i
#         sns.distplot(data[:,i], hist=True, kde=True, 
#                     bins=int(180/5), color = 'darkblue', 
#                     hist_kws={'edgecolor':'black'},
#                     kde_kws={'linewidth': 4},ax=axs[k,j])
#         axs[k,j].axvline(np.mean(data[:,i]),color='red')
#         plt.setp(axs[k,j].get_xticklabels(), rotation=30, horizontalalignment='right')
#         axs[k,j].set_ylabel('probability density')
#         axs[k,j].set_xlabel(fr'$\sigma^2({names[i]})$')
#     fig.delaxes(axs[1][3])
#     pp.savefig()
#     pp.close()

means = np.zeros(shape=(len(numsims),len(names)))
stds = np.zeros(shape=(len(numsims),len(names)))
mins = np.zeros(shape=(len(numsims),len(names)))
maxs = np.zeros(shape=(len(numsims),len(names)))
for l in range(len(numsims)):
    num = numsims[l]
    with open("Var_reg2_"+str(num)+"_"+str(2)+".pickle","rb") as file:
        data = pickle.load(file)
        current_path = os.path.abspath(os.getcwd())
    for i in range(data.shape[1]):
        means[l,i]=np.mean(data[:,i])
        stds[l,i] = np.std(data[:,i])
        mins[l,i]= np.min(data[:,i])
        maxs[l,i] = np.max(data[:,i])
pd.DataFrame(means).to_excel("means2.xlsx")
pd.DataFrame(stds).to_excel("stds2.xlsx")
pd.DataFrame(mins).to_excel("mins2.xlsx")
pd.DataFrame(maxs).to_excel("maxs2.xlsx")


