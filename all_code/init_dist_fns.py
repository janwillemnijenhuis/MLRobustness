#!/usr/bin/env python
# coding: utf-8

# In[2]:



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
import os
from tqdm import tqdm
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def data_sim(B,N,depth_list,seed_list,facets=2,regtree=False):

    #len_data = np.array([250,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000])
    #B = 1000 # number of replications
    if facets == 3:
        dist_matrix0 = np.zeros((B,7))
        dist_matrix1 = np.zeros((B,7)) # matrix with resulting variance estimates
    elif facets ==2:
        dist_matrix0 = np.zeros((B,3))
        dist_matrix1 = np.zeros((B,3))
    #N = 250 # number of predictions
    alpha = 4 # 
    beta =2
    gamma=1

    # generate test data
    X = np.sort(10*np.random.rand(N,1),axis=0)
    y = (alpha*np.sin(X)+beta) # sine function
    #     y = alpha*X + beta # linear function
    #     y = alpha*X**2 + beta # quadratic function
    y += gamma*np.random.normal(size=(N,1))
    y = y.ravel()

    #X,y = load_boston(return_X_y=True)
    #X,y = load_iris(return_X_y=True)
        
    X_train1,X_test,y_train1,_ = train_test_split(X,y,test_size=0.3)

    for l in tqdm(range(B)):
        X = np.sort(10*np.random.rand(N,1),axis=0)
        y = (alpha*np.sin(X)+beta) # sine function
    #     y = alpha*X + beta # linear function
    #     y = alpha*X**2 + beta # quadratic function
        y += gamma*np.random.normal(size=(N,1))
        y = y.ravel()

        #X,y = load_boston(return_X_y=True)
        #X,y = load_iris(return_X_y=True)
        
        X_train,_,y_train,_ = train_test_split(X_train1,y_train1,test_size=0.5)
        n = len(X_test)
        n_d = len(depth_list)
        n_s = len(seed_list)

        df_d = n_d-1
        df_s = n_s-1
        df_p = n-1
        df_ds = df_d*df_s
        df_pd = df_d*df_p
        df_ps = df_s*df_p
        df_pds = df_d*df_s*df_p
        results = np.zeros(shape=(n,n_d,n_s))
        
        if facets ==3:
            for i in range(n_d):
                for j in range(n_s):   

                    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state = seed_list[j])
                    if (regtree==True): 
                        reg = RandomForestRegressor(max_depth=int(depth_list[i]), random_state=int(seed_list[j]),n_jobs=-1)
                    else: 
                        reg = DecisionTreeRegressor(max_depth=int(depth_list[i]),random_state=seed_list[j])

                    reg.fit(X_train,y_train)
                    results[:,i,j] = reg.predict(X_test)
        elif facets==2:
            results = np.zeros(shape=(n,n_s))
            for i in range(n_s):   

                #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state = seed_list[j])
                if (regtree==True): 
                    reg = RandomForestRegressor(random_state=int(seed_list[i]),n_jobs=-1)
                else: 
                    reg = DecisionTreeRegressor(max_depth=int(depth_list[i]),random_state=seed_list[j])

                reg.fit(X_train,y_train)
                results[:,i] = reg.predict(X_test)
        if facets==3:        
            X_p = np.mean(results,axis=(1,2)) # person mean
            X_d = np.mean(results,axis=(0,2)) # depth mean
            X_s = np.mean(results,axis=(0,1)) # seed mean

            X_pd = np.mean(results,axis=2) # person x depth mean
            X_ps = np.mean(results,axis=1) # person x seed mean
            X_ds = np.mean(results,axis=0) # depth x seed mean

            X_tot = np.mean(results,axis=(0,1,2)).reshape(-1,1) # (observable) universe mean
            
            SS_p = (n_d*n_s*np.sum((X_p-X_tot)**2,axis=1))[0]
            SS_d = (n*n_s*np.sum((X_d-X_tot)**2,axis=1))[0]
            SS_s = (n*n_d*np.sum((X_s-X_tot)**2,axis=1))[0]

            SS_pd = np.zeros(shape=(n,n_d))
            SS_ps = np.zeros(shape=(n,n_s))
            SS_ds = np.zeros(shape=(n_d,n_s))

            SS_pds = np.zeros(shape=(n,n_d,n_s))

            for i in range(n_d):
                for j in range(n_s):
                    SS_ds[i,j] = (X_ds[i,j].ravel()-X_d[i].ravel()-X_s[j].ravel()+X_tot.ravel())**2
                    for k in range(n):
                        SS_ps[k,j] = (X_ps[k,j].ravel()-X_s[j].ravel()-X_p[k].ravel()+X_tot.ravel())**2
                        SS_pds[k,i,j] = (results[k,i,j].ravel()-X_ds[i,j].ravel() - X_pd[k,i].ravel()-X_ps[k,j].ravel()
                                        +X_d[i].ravel()+X_s[j].ravel()+X_p[k].ravel()-X_tot.ravel())**2
                for k in range(n):
                    SS_pd[k,i] = (X_pd[k,i].ravel()-X_d[i].ravel()-X_p[k].ravel()+X_tot.ravel())**2

            SS_ds = n*np.sum(SS_ds,axis=(0,1))
            SS_pd = n_s*np.sum(SS_pd,axis=(0,1))
            SS_ps = n_d*np.sum(SS_ps,axis=(0,1))
            SS_pds = np.sum(SS_pds,axis=(0,1,2))
            
            MS_d = SS_d/df_d
            MS_s = SS_s/df_s
            MS_p = SS_p/df_p
            MS_ds = SS_ds/df_ds
            MS_pd = SS_pd/df_pd
            MS_ps = SS_ps/df_ps
            MS_pds = SS_pds/df_pds
            
            sigma2_d = (MS_d-MS_ds-MS_pd+MS_pds)/(n_s*n)
            sigma2_s = (MS_s-MS_ds-MS_ps+MS_pds)/(n_d*n)
            sigma2_p = (MS_p-MS_pd-MS_ps+MS_pds)/(n_d*n_s)
            sigma2_ds = (MS_ds-MS_pds)/n
            sigma2_pd = (MS_pd-MS_pds)/n_s
            sigma2_ps = (MS_ps-MS_pds)/n_d
            sigma2_pds = MS_pds
            
            dist_matrix0[l,:] = [MS_p,MS_d,MS_s,MS_pd,MS_ps,MS_ds,MS_pds]
            dist_matrix1[l,:] = [sigma2_p,sigma2_d,sigma2_s,sigma2_pd,sigma2_ps,sigma2_ds,sigma2_pds]
        elif facets==2:
            X_p = np.mean(results,axis=1)
            X_s = np.mean(results,axis=0)
            X_tot = np.mean(results)
            SS_p = n_s*np.sum((X_p-X_tot)**2)
            SS_s = n*np.sum((X_s-X_tot)**2)
            SS_ps = np.zeros(shape=(n,n_s))
            for i in range(n):
                for j in range(n_s):
                    SS_ps[i,j] = (results[i,j]-X_p[i]-X_s[j]+X_tot)**2
            SS_ps = np.sum(SS_ps)
            MS_p = SS_p/(n-1)
            MS_s = SS_s/(n_s-1)
            MS_ps = SS_ps/((n-1)*(n_s-1))
            sigma2_p = (MS_p-MS_ps)/n_s
            sigma2_s = (MS_s-MS_ps)/N
            sigma2_ps = MS_ps

            dist_matrix0[l,:] = [MS_p,MS_s,MS_ps]
            dist_matrix1[l,:] = [sigma2_p,sigma2_s,sigma2_ps]

    return dist_matrix0, dist_matrix1

def create_plots(N,dist_matrix,facets=2,type="variance",savefig=False,filename="name"):
    if facets==3:
        names = ['person','depth','seed','person x depth', 'person x seed', 'depth x seed', 'residual']
        cols = 7
    elif facets==2:
        cols =3
        names = ['person','seed','person x seed']
    current_path = os.path.abspath(os.getcwd())
    my_file = '{}.pdf'.format(filename)
    if (savefig==True): pp = PdfPages(os.path.join(current_path, my_file))
    
    for i in range(cols):
        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot()
        sns.distplot(dist_matrix[:,i], hist=True, kde=True, 
                bins=int(180/5), color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4},ax=ax)
        plt.axvline(np.mean(dist_matrix[:,i]),color='red')
        ax.set_title(names[i]+" "+type+" "+"distribution (N="+str(N)+")")
        ax.set_ylabel('probability density')
        ax.set_xlabel(fr'$\sigma^2({names[i]})$')
        if (savefig==True): 
            pp.savefig()
    if (savefig==True): pp.close()

def yi_sim(B,num_obs,depth_list,seed_list,test_val,facets=2,regtree=False):

    #len_data = np.array([250,500,1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000])
    #B = 1000 # number of replications
    if facets == 3:
        dist_matrix0 = np.zeros((B,8))
        dist_matrix1 = np.zeros((B,8)) # matrix with resulting variance estimates
        err = np.zeros(shape=(B,len(depth_list),len(seed_list)))
    elif facets ==2:
        dist_matrix0 = np.zeros((B,4))
        dist_matrix1 = np.zeros((B,4))
        err = np.zeros(shape=(B,len(seed_list)))
    
    X_true = test_val*np.array([1])
    V_true = test_val*np.array([1])
    Z_true = test_val*np.array([1])
    W_true = test_val*np.array([1])
    test_true = np.hstack([V_true, W_true, X_true, Z_true])
    f_test = 2*V_true-W_true**2+(X_true-1)**2-3*Z_true+2
    
    for i in tqdm(range(len(num_obs))):
        N=num_obs[i]
        
        
        for l in tqdm(range(B)):
            V = 5*np.random.normal(0,1,size=(N,1))
            W = 5*np.random.normal(0,1,size=(N,1))
            X = 5*np.random.normal(0,1,size=(N,1))
            Z = 5*np.random.normal(0,1,size=(N,1))


            train  = np.hstack([V, W, X, Z])
            
            f_x = (2*V-W**2+(X-1)**2-3*Z+2)

            y = (f_x + 1*np.random.normal(loc=0,scale=1,size=(N,1)))
            
            eta = y-f_x
            sig2_eta = np.var(eta) 
            y=y.ravel()
            
            if (i == 0):

                V1 = np.array([-1,0,1])
                W1 = np.array([-1,0,1])
                X1 = np.array([-1,0,1])
                Z1 = np.array([-1,0,1])
                y1 = 2*V1-W1**2+(X1-1)**2-3*Z1+2

                vrange = np.linspace(np.min(V),np.max(V),1000)
                wrange = np.linspace(np.min(W),np.max(W),1000)
                xrange = np.linspace(np.min(X),np.max(X),1000)
                zrange = np.linspace(np.min(Z),np.max(Z),1000)

                fig = plt.figure(figsize=(16,16))
                fig.suptitle('N = '+str(N))

                ax1 = fig.add_subplot(221)
                ax1.scatter(V,y)
                ax1.plot(vrange,2*vrange,c='r')
                ax1.scatter(V1,y1,c=['green','yellow','purple'],marker='s',s=75)
                ax1.title.set_text('y=2V')

                ax2 = fig.add_subplot(222)
                ax2.scatter(W,y)
                ax2.plot(wrange,-(wrange**2),c='r')
                ax2.title.set_text('y=-W^2')
                ax2.scatter(W1,y1,c=['green','yellow','purple'],marker='s',s=75)

                ax3 = fig.add_subplot(223)
                ax3.scatter(X,y)
                ax3.plot(xrange,(xrange-1)**2,c='r')
                ax3.title.set_text('y=(X-1)^2')
                ax3.scatter(X1,y1,c=['green','yellow','purple'],marker='s',s=75)

                ax4 = fig.add_subplot(224)
                ax4.scatter(Z,y)
                ax4.plot(zrange,-3*zrange,c='r')
                ax4.title.set_text('y=-3Z')
                ax4.scatter(Z1,y1,c=['green','yellow','purple'],marker='s',s=75)

                plt.savefig('dist_'+str(N))
               

            
            n = test_true.shape[0]
            n_d = len(depth_list)
            n_s = len(seed_list)

            df_d = n_d-1
            df_s = n_s-1
            df_p = n
            df_ds = df_d*df_s
            df_pd = df_d*df_p
            df_ps = df_s*df_p
            df_pds = df_d*df_s*df_p
            results = np.zeros(shape=(n,n_d,n_s))
            
            if facets ==3:
                for i in range(n_d):
                    for j in range(n_s):   
                        if (regtree==True): 
                            reg = RandomForestRegressor(max_depth=int(depth_list[i]), random_state=int(seed_list[j]),n_jobs=-1)
                        # else: 
                        #     reg = DecisionTreeRegressor(max_depth=int(depth_list[i]),random_state=seed_list[j])

                        reg.fit(train,y)
                        
                        y_pred = reg.predict(test_true.reshape(1,-1)) 
                        
                        results[:,i,j] = y_pred
                        
                        err[l,i,j] = y_pred-f_test
            elif facets==2:

                results = np.zeros(shape=(n,n_s))
                for i in range(n_s):   
                    
                    if (regtree==True): 
                        reg = RandomForestRegressor(n_estimators=100,random_state=int(seed_list[i]),n_jobs=-1)
                    #else: 
                        #reg = DecisionTreeRegressor(max_depth=int(depth_list[i]),random_state=seed_list[j])

                    reg.fit(train,y)

                    y_pred = reg.predict(test_true.reshape(1,-1)) 
                    
                    results[:,i] = y_pred
                    
                    err[l,i] = y_pred-f_test
                    
            if facets==3:        
                X_p = np.mean(results,axis=(1,2)) # person mean
                X_d = np.mean(results,axis=(0,2)) # depth mean
                X_s = np.mean(results,axis=(0,1)) # seed mean

                X_pd = np.mean(results,axis=2) # person x depth mean
                X_ps = np.mean(results,axis=1) # person x seed mean
                X_ds = np.mean(results,axis=0) # depth x seed mean

                X_tot = np.mean(results,axis=(0,1,2)).reshape(-1,1) # (observable) universe mean
                SS_p = (n_d*n_s*np.sum((X_p-X_tot)**2,axis=1))[0]
                SS_d = (n*n_s*np.sum((X_d-X_tot)**2,axis=1))[0]
                SS_s = (n*n_d*np.sum((X_s-X_tot)**2,axis=1))[0]

                SS_pd = np.zeros(shape=(n,n_d))
                SS_ps = np.zeros(shape=(n,n_s))
                SS_ds = np.zeros(shape=(n_d,n_s))

                SS_pds = np.zeros(shape=(n,n_d,n_s))

                for i in range(n_d):
                    for j in range(n_s):
                        SS_ds[i,j] = (X_ds[i,j].ravel()-X_d[i].ravel()-X_s[j].ravel()+X_tot.ravel())**2
                        for k in range(n):
                            SS_ps[k,j] = (X_ps[k,j].ravel()-X_s[j].ravel()-X_p[k].ravel()+X_tot.ravel())**2
                            SS_pds[k,i,j] = (results[k,i,j].ravel()-X_ds[i,j].ravel() - X_pd[k,i].ravel()-X_ps[k,j].ravel()
                                            +X_d[i].ravel()+X_s[j].ravel()+X_p[k].ravel()-X_tot.ravel())**2
                    for k in range(n):
                        SS_pd[k,i] = (X_pd[k,i].ravel()-X_d[i].ravel()-X_p[k].ravel()+X_tot.ravel())**2

                SS_ds = n*np.sum(SS_ds,axis=(0,1))
                SS_pd = n_s*np.sum(SS_pd,axis=(0,1))
                SS_ps = n_d*np.sum(SS_ps,axis=(0,1))
                SS_pds = np.sum(SS_pds,axis=(0,1,2))
                
                MS_d = SS_d/df_d
                MS_s = SS_s/df_s
                MS_p = SS_p/df_p
                MS_ds = SS_ds/df_ds
                MS_pd = SS_pd/df_pd
                MS_ps = SS_ps/df_ps
                MS_pds = SS_pds/df_pds
                
                sigma2_d = (MS_d-MS_ds-MS_pd+MS_pds)/(n_s*n)
                sigma2_s = (MS_s-MS_ds-MS_ps+MS_pds)/(n_d*n)
                sigma2_p = (MS_p-MS_pd-MS_ps+MS_pds)/(n_d*n_s)
                sigma2_ds = (MS_ds-MS_pds)/n
                sigma2_pd = (MS_pd-MS_pds)/n_s
                sigma2_ps = (MS_ps-MS_pds)/n_d
                sigma2_pds = MS_pds
                
                dist_matrix0[l,:] = [MS_p,MS_d,MS_s,MS_pd,MS_ps,MS_ds,MS_pds,sig2_eta]
                dist_matrix1[l,:] = [sigma2_p,sigma2_d,sigma2_s,sigma2_pd,sigma2_ps,sigma2_ds,sigma2_pds,sig2_eta]
            elif facets==2:
                X_p = np.mean(results,axis=1)
                X_s = np.mean(results,axis=0)
                X_tot = np.mean(results)

                SS_p = n_s*np.sum((X_p-X_tot)**2)
                SS_s = n*np.sum((X_s-X_tot)**2)
                SS_ps = np.zeros(shape=(n,n_s))
                for i in range(n):
                    for j in range(n_s):
                        SS_ps[i,j] = (results[i,j]-X_p[i]-X_s[j]+X_tot)**2
                SS_ps = np.sum(SS_ps)
                MS_p = SS_p/df_p
                MS_s = SS_s/df_s
                MS_ps = SS_ps/(df_p*df_s)
                sigma2_p = (MS_p-MS_ps)/n_s
                sigma2_s = (MS_s-MS_ps)/n
                sigma2_ps = MS_ps

                dist_matrix1[l,:] = [sigma2_p,sigma2_s,sigma2_ps,sig2_eta]
        if (facets==2):
            error = np.mean(err,axis=1)
        elif (facets==3):
            error = np.mean(err,axis=(1,2))
        err_var = np.var(error)
        data = np.hstack((dist_matrix1,error.reshape(-1,1)))
        with open("Var_reg2_"+str(B)+"_"+str(N)+"_"+str(facets)+"_"+str(test_val)+".pickle","wb") as file:
            pickle.dump(data,file)
        display(np.mean(data,axis=0))
        display(err_var)

seed_list = np.linspace(100,1000,10,dtype=int)
facets=3
#depth_list = np.linspace(100,1000,10,dtype=int) ### USE FOR RANDOM FOREST ###
#depth_list = np.linspace(30,50,10,dtype=int) ### USE FOR DECISION TREE ###
depth_list = np.linspace(5,50,10,dtype=int) ### USE FOR DECISION TREE ###       
num_obs = [250,500,1000]
test_vals = [-1.0,0.0,1.0]
for val in tqdm(test_vals):
    yi_sim(1000,num_obs,depth_list,seed_list,val,facets,regtree=True)