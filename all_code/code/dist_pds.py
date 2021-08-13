import sys
import numpy as np
from IPython.display import display
from tqdm import tqdm
import sys
import pickle
sys.path.append('C:/Users/janwi/OneDrive/Documents/stage/projects')
import init_dist_fns


seed_list = np.linspace(100,1000,10,dtype=int)

#depth_list = np.linspace(100,1000,10,dtype=int) ### USE FOR RANDOM FOREST ###
#depth_list = np.linspace(30,50,10,dtype=int) ### USE FOR DECISION TREE ###
depth_list = np.linspace(2,30,10,dtype=int) ### USE FOR DECISION TREE ###
display('test')
num_obs = [250,500,1000,2000]
for i in tqdm(range(len(num_obs))):
    N = num_obs[i]
    facets=2
    dist_MS,dist_sig = init_dist_fns.yi_sim(1000,num_obs,depth_list,seed_list,facets,regtree=True)
    init_dist_fns.create_plots(N,dist_MS,facets,type="MS",savefig=True,filename="MS_reg_"+str(N))
    init_dist_fns.create_plots(N,dist_sig,facets,savefig=True,filename="Var_reg_"+str(N))
    display(dist_sig)
    display(np.mean(dist_sig,axis=0))
    display(np.std(dist_sig,axis=0))
    display(np.min(dist_sig,axis=0))
    display(np.max(dist_sig,axis=0))
    
    with open("MS_reg2_"+str(N)+"_"+str(facets)+".pickle","wb") as file:
        pickle.dump(dist_MS,file)
    with open("Var_reg2_"+str(N)+"_"+str(facets)+".pickle","wb") as file:
        pickle.dump(dist_sig,file)
