import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from mpl_toolkits import mplot3d
from matplotlib import cm


import os
import numpy as np
path = os.getcwd()
os.chdir(os.path.join(path,'cors'))

data = np.array(pd.read_excel('results_cors_exp2_3_facets_depth.xlsx'))
n_s = data.shape[0]
n_d = data.shape[1]-1
Z = np.array(data)[:n_s,1:n_d+1]
X = np.linspace(100,1000,n_s,dtype=int)
Y = np.linspace(5,15,n_d,dtype=int)
Y,X = np.meshgrid(Y,X)
fig = plt.figure(figsize=(16,16))
ax = plt.axes(projection='3d')
ax.set_xlabel('seed')
ax.set_ylabel('depth')
ax.set_zlabel('overlap')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.view_init(20, 45)
plt.savefig('exp2')
