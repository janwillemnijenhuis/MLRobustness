import os
from IPython.display import display
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import inquirer
import pickle

def create_plots(N,dist_matrix,testval,facets=2,type="variance",savefig=False,filename="name"):
    if facets==3:
        names = ['person variance','depth variance','seed variance','person x depth variance', 'person x seed variance', 'depth x seed variance', 'person x depth x seed variance','population error variance','prediction error']
        ind = ['p','d','s','pd','ps','ds','pds','\eta','\epsilon']
        cols = 9
        fcol = cols-1
    elif facets==2:
        cols =5
        fcol=cols-1
        names = ['person variance','seed variance','person x seed variance','population error variance','prediction error']
        ind = ['p','s','ps','\eta','\epsilon']
    current_path = os.path.abspath(os.getcwd())
    my_file = '{}.pdf'.format(filename)
    if (savefig==True): pp = PdfPages(os.path.join(current_path, my_file))
    
    for i in range(cols):
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot()
        sns.histplot(dist_matrix[:,i],bins=50,ax=ax)
        plt.axvline(np.mean(dist_matrix[:,i]),color='red')
        ax.set_title(names[i]+" distribution (N="+str(N)+") for "+fr"$\delta = $"+str(testval))
        ax.set_ylabel('count')
        if (i==fcol):
            ax.set_xlabel(fr'${ind[i]}$')
        else:
            ax.set_xlabel(fr'$\sigma^2({ind[i]})$')
        if (savefig==True): 
            pp.savefig()
    if (savefig==True): pp.close()

notebook_path = os.path.split(os.path.abspath(os.getcwd()))[0]
display(notebook_path)
os.chdir(os.path.join(os.getcwd(),'4_mc2\datafiles'))
display(os.getcwd())
facets = [2,3]
result='plots'
obs = [250,500,1000]
testvals = [-1.0,0.0,1.0]
for j in range(len(facets)):
    facet = facets[j]
    if (facet==2):
        dataf = np.zeros(shape=(20,5))
    else:
        dataf = np.zeros(shape=(20,9))
    for k in range(len(testvals)):
        testval = testvals[k]
        for i in range(len(obs)):
            n = obs[i]
            with open('Var_reg2_1000_'+str(n)+'_'+str(facet)+'_'+str(testval)+'.pickle',"rb") as file:
                data = pickle.load(file)
            if (result=='plots'):
                create_plots(n,data,testval=testval,facets=facet,type="variance",savefig=True,filename='Var_reg2_1000_'+str(n)+'_'+str(facet)+'_'+str(testval))
            else:
                dataf[i,:] = np.mean(data,0)
                dataf[i+4,:] = np.std(data,0)
                dataf[i+8,:] = np.var(data,axis=0,ddof=0)
                dataf[i+12,:] = np.min(data,0)
                dataf[i+16,:] = np.max(data,0)
        if(result=='stats'):
            df = pd.DataFrame(data=dataf)
            df.to_excel('sim2_'+str(facet)+'facets_'+str(n)+'obs_'+str(testval)+'.xlsx')

def open_results():
    results_dir = os.path.join(notebook_path,"results\data")
    
    os.chdir(results_dir)
    list_files = [x for x in os.listdir(results_dir)]
    questions=[
        inquirer.List('file',
                    message="Which file do you need us to open?",
                    choices=list_files,
                ),
    ]
    answers = inquirer.prompt(questions)
    filename = answers['file']
    with open(filename,"rb") as file:
        data = pickle.load(file)
    return data

def rmse(predictions,target,axis=0):
    rmse = np.sqrt(np.mean((predictions-target)**2,axis=axis))
    return rmse

def normalize(vec):
    new_vec = [vec[i]/np.sum(vec) for i in range(len(vec))]
    return new_vec

