
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
os.chdir(os.path.join(os.getcwd(),'results'))
display(os.getcwd())
facets = [2]
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
                create_plots(n,data,testval=testval,facets=facet,type="variance",savefig=True,filename='Var_reg2_1000_'+str(n)+'_'+str(facet)+'_'+str(testval)+'pi')
            else:
                dataf[i,:] = np.mean(data,0)
                dataf[i+4,:] = np.std(data,0)
                dataf[i+8,:] = np.var(data,axis=0,ddof=0)
                dataf[i+12,:] = np.min(data,0)
                dataf[i+16,:] = np.max(data,0)
        if(result=='stats'):
            df = pd.DataFrame(data=dataf)
            df.to_excel('sim2_'+str(facet)+'facets_'+str(n)+'obs_'+str(testval)+'.xlsx')
#os.chdir(os.path.join(os.getcwd(),'results'))
# with open('Var_reg2_2000_3.pickle','rb') as file:
#     dist_matrix = pickle.load(file)
# N = 2000
# create_plots(N,dist_matrix,facets=3,type="variance",savefig=True,filename="name")

def open_results():
    results_dir = os.path.join(notebook_path,"results\data")
    # list_dirs = [x[0] for x in os.walk(results_dir)]
    # questions = [
    # inquirer.List('folder',
    #                 message="Which folder do you need us to open?",
    #                 choices=list_dirs,
    #             ),
    # ]
    # answers = inquirer.prompt(questions)
    # path = answers['folder']
    # os.chdir(path)
    # print(os.getcwd())
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

# facets =2
# if (facets==2):
#     sig2 = np.array([5,15,25])
# elif (facets==3):
#     sig2 = np.array([5,10,25,5,5,5,2])
# sig2 = np.array([4,8,16,5,5,5,2])
# data = open_results()
# display(normalize(sig2))
# display(normalize(np.mean(data,axis=0)))
# normed_data = np.array([normalize(data[i,:]) for i in range(data.shape[0])])
# display(np.mean(np.abs(normed_data-normalize(sig2)),axis=0))
# display(np.std(normed_data,axis=0))
# display(rmse(normed_data,normalize(sig2)))