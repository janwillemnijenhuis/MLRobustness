# standard python imports #
import os
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from scipy.stats import t
from tqdm import tqdm
import time
import pickle

# locally created imports #
notebook_path = os.path.split(os.path.abspath(os.getcwd()))[0]
os.chdir(notebook_path)

dvb_path = os.path.split(os.path.abspath(notebook_path))[0]
os.chdir(dvb_path)
from dvb.bdp.model.oversluit_model.tools import utils_eda_specific as eda

os.chdir(notebook_path)

with open("saved_df_pilot.pckl","rb") as file:
    data_df = pickle.load(file)

np.random.seed(seed=123)

def gen_train_test_val_split(data_df,train_size=0.34,test_size=0.33,random_state=123):

    val_size = 1-train_size-test_size
    thresh1 = train_size/(train_size+test_size+val_size)
    np.random.seed(random_state); idx = np.random.rand(len(data_df))<thresh1
    data_df_train = data_df[idx]
    data_df_test_val = data_df[~idx]
    thresh2 = test_size/(test_size+val_size)
    np.random.seed(random_state); idx2 = np.random.rand(len(data_df_test_val))<thresh2
    data_df_test = data_df_test_val[idx2]
    data_df_val = data_df_test_val[~idx2]

    return data_df_train,data_df_test,data_df_val

features = ['proxy_rente_incentive_10y',
 'proxy_rente_incentive_10y_pos_part',
 'proxy_rente_incentive_20y',
 'ind_kisg_container_oversluiten_lb3',
 'weighted_duur_tot_einde_rvp',
 'proxy_rente_incentive_20y_pos_part',
 'duur_tot_einde_rvp_min',
 'ind_kisg_boete_of_aflosnota_lb3',
 'duur_tot_einde_rvp_max',
 'ind_kisg_oversluiten_lb3',
 'AantalVerkochteWoningen_MA',
 'avg_rente_peers_20y',
 'cred_looptijd_verstr_oudst_numc',
 'cred_mnd_hyp_klant_numi',
 'hh_loanAge_numi',
 'cred_looptijd_verstr_numc',
 'min_rente_peers_20y',
 'cred_gem_rente_numc',
 'weighted_delta5Yklantrente_MA',
 'cred_eam_bg_numc',
 'term_gecorr_maandtermijn_nota_bg_numc',
 'avg_rente_peers_10y',
 'hh_leeftijd_min_hfd_numi',
 'cred_pd_berekend_totaal_numc',
 'cred_ltv_m_ix_numc',
 'max_delta5Yklantrente',
 'weighted_delta5Yklantrente',
 'min_rente_peers_10y',
 'sal_gem_saldo_6_maand_numc',
 'min_delta5Yklantrente']

name_target = "ind_change_bank_3_to_9"

eval_time_data = True
# trained_model = eda.fit_model(model, data_df_train, features, name_target)
# predictions = trained_model.predict_proba(data_df_test[features])[:, 1]
# display(predictions)
njobs = 16 # set the number of jobs in order to not kill everyone else's work. Set to -1 if everyone is home (and you're still going strong!)
if (eval_time_data==False):
    data_df_facets,data_df_test,_ = gen_train_test_val_split(data_df,train_size=0.67,test_size=0.33) #split into data used for facet research and data for testing (the latter needs to be the same all the time)
elif (eval_time_data==True):
    idx = (data_df['jm_nr']==202005)
    data_df_facets = data_df[~idx]
    data_df_test = data_df[idx]
display(len(data_df_facets))
display(len(data_df_test))
data_df_train_true,_,_ = gen_train_test_val_split(data_df_facets,train_size=0.5,test_size=0) # split the facet data in a train and validation part
model = RandomForestClassifier(class_weight={0: 1, 1: 5}, max_depth=5, n_estimators=500,
                       n_jobs=njobs, random_state=123) #basis "true" model
trained_model = eda.fit_model(model, data_df_train_true, features, name_target) #train the model
true_predictions = trained_model.predict_proba(data_df_test[features])[:,1] #make predictions
top_1000_idx = sorted(range(len(true_predictions)), reverse=True, key=lambda k: true_predictions[k])[:1000] #select indices for top 1000 probabilities

seed_list = np.linspace(100,1000,25,dtype=int)
#depth_list = np.linspace(5,15,5,dtype=int)
depth_list = np.linspace(300,900,5,dtype=int)
display(depth_list)

n_p = len(top_1000_idx) #len(data_df_test1)
n_s = len(seed_list)
n_d = len(depth_list)

facets = 2
if (facets==2):
    results = np.zeros(shape=(n_p,n_s))
    for i in tqdm(range(n_s)):
        data_df_train_facets,_,_ = gen_train_test_val_split(data_df_facets,train_size=0.5,test_size=0,random_state=seed_list[i])
        model = RandomForestClassifier(class_weight={0: 1, 1: 5}, max_depth=5, n_estimators=500,
                        n_jobs=njobs, random_state=seed_list[i])
        trained_model = eda.fit_model(model, data_df_train_facets, features, name_target)
        results[:,i] = (trained_model.predict_proba(data_df_test[features])[:,1])[top_1000_idx] #predictions for class==1
        #top_1000_idx = sorted(range(len(true_predictions)), reverse=True, key=lambda k: true_predictions[k])#[:1000] #select indices for top 1000 probabilities

elif (facets==3):
    results = np.zeros(shape=(n_p,n_s,n_d))
    for i in tqdm(range(n_s)):
        for j in tqdm(range(n_d)):
            data_df_train_facets,_,_ = gen_train_test_val_split(data_df_facets,train_size=0.5,test_size=0,random_state=seed_list[i])
            # model = RandomForestClassifier(class_weight={0: 1, 1: 5}, max_depth=depth_list[j], n_estimators=500,
            #                n_jobs=njobs, random_state=seed_list[i])
            model = RandomForestClassifier(class_weight={0: 1, 1: 5}, max_depth=5, n_estimators=depth_list[j],
                            n_jobs=njobs, random_state=seed_list[i])
            trained_model = eda.fit_model(model, data_df_train_facets, features, name_target)
            results[:,i,j] = (trained_model.predict_proba(data_df_test[features])[:,1])[top_1000_idx] #predictions for class==1

def compute_components_2(results):
    X_p = np.mean(results,axis=1) # depth mean
    X_s = np.mean(results,axis=0) # seed mean

    X_tot = np.mean(results,axis=(0,1)) # universe mean

    SS_p = n_s*np.sum((X_p-X_tot)**2)
    MS_p = SS_p/(n_p-1)
    SS_s = n_p*np.sum((X_s-X_tot)**2)
    MS_s = SS_s/(n_s-1)
    SS_ps = np.zeros(shape=(n_p,n_s))
    for i in range(n_p):
        for j in range(n_s):
            SS_ps[i,j] = (results[i,j] - X_p[i] - X_s[j] + X_tot)**2

    SS_ps = np.sum(SS_ps,axis=(0,1))
    MS_ps = SS_ps/((n_p-1)*(n_s-1))
    sigma2_p = (MS_p-MS_ps)/n_s
    sigma2_s = (MS_s-MS_ps)/n_p
    sigma2_ps = MS_ps


    df = pd.DataFrame([sigma2_p,sigma2_s,
                    sigma2_ps],)
    df.index=['p','s','ps']

    var_matrix = np.array([sigma2_p,sigma2_s,sigma2_ps])
    var_matrix[var_matrix<0] = 0 # set each negative variance to 0

    df_0 = pd.DataFrame(var_matrix)
    df_0.index=['p','s','ps']

    return df,df_0,var_matrix

def compute_components_3(results):
    X_p = np.mean(results,axis=(1,2)) # person mean
    X_d = np.mean(results,axis=(0,1)) # depth mean
    X_s = np.mean(results,axis=(0,2)) # seed mean

    X_pd = np.mean(results,axis=1) # person x depth mean
    X_ps = np.mean(results,axis=2) # person x seed mean
    X_ds = np.mean(results,axis=0) # depth x seed mean

    X_tot = np.mean(results,axis=(0,1,2)).reshape(-1,1) # (observable) universe mean

    SS_p = (n_d*n_s*np.sum((X_p-X_tot)**2,axis=1))[0]
    SS_d = (n_p*n_s*np.sum((X_d-X_tot)**2,axis=1))[0]
    SS_s = (n_p*n_d*np.sum((X_s-X_tot)**2,axis=1))[0]

    SS_pd = np.zeros(shape=(n_p,n_d))
    SS_ps = np.zeros(shape=(n_p,n_s))
    SS_ds = np.zeros(shape=(n_s,n_d))

    SS_pds = np.zeros(shape=(n_p,n_s,n_d))

    for i in range(n_s):
        for j in range(n_d):
            SS_ds[i,j] = (X_ds[i,j].ravel()-X_d[j].ravel()-X_s[i].ravel()+X_tot.ravel())**2
            for k in range(n_p):
                SS_pd[k,j] = (X_pd[k,j].ravel()-X_d[j].ravel()-X_p[k].ravel()+X_tot.ravel())**2
                SS_pds[k,i,j] = (results[k,i,j].ravel()-X_ds[i,j].ravel() - X_pd[k,j].ravel()-X_ps[k,i].ravel()
                                +X_d[j].ravel()+X_s[i].ravel()+X_p[k].ravel()-X_tot.ravel())**2
        for k in range(n_p):
            SS_ps[k,i] = (X_ps[k,i].ravel()-X_s[i].ravel()-X_p[k].ravel()+X_tot.ravel())**2
            
    SS_ds = n_p*np.sum(SS_ds,axis=(0,1))
    SS_pd = n_s*np.sum(SS_pd,axis=(0,1))
    SS_ps = n_d*np.sum(SS_ps,axis=(0,1))
    SS_pds = np.sum(SS_pds,axis=(0,1,2))


    ## Degrees of freedom
    df_d = n_d-1
    df_s = n_s-1
    df_p = n_p-1
    df_ds = df_d*df_s
    df_pd = df_d*df_p
    df_ps = df_s*df_p
    df_pds = df_d*df_s*df_p


    # The mean squares (MS) are simply the sum of squares divided by it's degrees of freedom.
    MS_d = SS_d/df_d
    MS_s = SS_s/df_s
    MS_p = SS_p/df_p
    MS_ds = SS_ds/df_ds
    MS_pd = SS_pd/df_pd
    MS_ps = SS_ps/df_ps
    MS_pds = SS_pds/df_pds


    # The variance is then computed from the mean squares (MS) according to the following manner. It is here explained for a number $j$ of possible facets in order to show that it generalizes easiliy. Consdider $\alpha$ the set of indices which are contained in a facet, $\dot{\alpha}$ (notice the dot) for the set of indices not contained in the facet and $\omega$ the set of possible indices. For example in $\bar{X}_{ds}$, we would have $\alpha = \{d,s\}$, $\dot{\alpha} = \{p\}$ and $\omega = \{d,s,p\}$. The variance is computed as: $\frac{MS(\alpha)-\sum_j MS(\alpha;\dot{\alpha}_j) + \sum_i MS(\alpha;\dot{\alpha}_i) }{ \pi(\alpha)}$. Here, the index $j$ sums over all MS that contain the terms in $\alpha$ and an odd number of terms from $\dot{\alpha}$. On the other hand, $i$ sums over all MS containing the terms in $\alpha$ and an even number of terms from $\dot{\alpha}$. $\pi(\alpha) = \prod_{i \in \dot{\alpha}} n_i$. So for example for the variance for facet $d$ we have $\hat{\sigma}^2(d) = \frac{MS(d)-MS(ds)-MS(pd)+MS(pds)}{n_s n}$. 
    sigma2_d = (MS_d-MS_ds-MS_pd+MS_pds)/(n_s*n_p)
    sigma2_s = (MS_s-MS_ds-MS_ps+MS_pds)/(n_d*n_p)
    sigma2_p = (MS_p-MS_pd-MS_ps+MS_pds)/(n_d*n_s)
    sigma2_ds = (MS_ds-MS_pds)/n_p
    sigma2_pd = (MS_pd-MS_pds)/n_s
    sigma2_ps = (MS_ps-MS_pds)/n_d
    sigma2_pds = MS_pds

    df = pd.DataFrame([sigma2_p,sigma2_s,
                    sigma2_d,sigma2_ps,
                    sigma2_pd,sigma2_ds,
                    sigma2_pds],)
    df.index=['p','s','d','ps','pd','ds','pds']

    # ### Negative variances ###
    # Due to the construction of the variance components from the MS some variance components can turn out to become negative. This has also been discussed by Cronbach, Shavelson and Brennan. The best approach according to the literature is to set the negative estimates to zero due to sampling variability. This will not affect the other variance estimates, but it will influence the generalizability coefficient. The table above shows that the variance which are negative are in fact very small. The table below shows the mean variances for each facet when setting negative variances to 0.
    var_matrix = np.array([sigma2_p,sigma2_s,sigma2_d,sigma2_ps,sigma2_pd,sigma2_ds,sigma2_pds])
    var_matrix[var_matrix<0] = 0 # set each negative variance to 0

    sigma2_p_hat = var_matrix[0]
    sigma2_s_hat = var_matrix[1]
    sigma2_d_hat = var_matrix[2]
    sigma2_ps_hat = var_matrix[3]
    sigma2_pd_hat = var_matrix[4]
    sigma2_ds_hat = var_matrix[5]
    sigma2_pds_hat = var_matrix[6]

    df_0 = pd.DataFrame([sigma2_p_hat,sigma2_s_hat,
                    sigma2_d_hat,sigma2_ps_hat,
                    sigma2_pd_hat,sigma2_ds_hat,
                    sigma2_pds_hat],)
    df_0.index=['p','s','d','ps','pd','ds','pds']

    return df,df_0,var_matrix

def compute_g_coefficients_2(var_matrix,n_s):
    sigma2_p_hat = var_matrix[0]
    sigma2_s_hat = var_matrix[1]
    sigma2_ps_hat = var_matrix[2]

    del_p = sigma2_ps_hat/n_s
    Del_p = sigma2_s_hat/n_s + sigma2_ps_hat/n_s 
    rho2_p = sigma2_p_hat/(sigma2_p_hat+del_p)
    phi_p = sigma2_p_hat/(sigma2_p_hat+Del_p)

    df = pd.DataFrame([rho2_p, phi_p])
    return df

def compute_g_coefficients_3(var_matrix,n_s,n_d):
    sigma2_p_hat = var_matrix[0]
    sigma2_s_hat = var_matrix[1]
    sigma2_d_hat = var_matrix[2]
    sigma2_ps_hat = var_matrix[3]
    sigma2_pd_hat = var_matrix[4]
    sigma2_ds_hat = var_matrix[5]
    sigma2_pds_hat = var_matrix[6]

    del_p = sigma2_pd_hat/n_d + sigma2_ps_hat/n_s + sigma2_pds_hat/(n_d*n_s)
    Del_p = sigma2_s_hat/n_s + sigma2_d_hat/n_d + sigma2_pd_hat/n_d + sigma2_ps_hat/n_s + sigma2_pds_hat/(n_d*n_s)
    rho2_p = sigma2_p_hat/(sigma2_p_hat+del_p)
    phi_p = sigma2_p_hat/(sigma2_p_hat+Del_p)

    df = pd.DataFrame([rho2_p, phi_p])
    df.index = ['rho','phi']
    return df

if (facets==2):
    df,df_0,var_matrix = compute_components_2(results)
    df_g = compute_g_coefficients_2(var_matrix,n_s)
elif (facets==3):
    df,df_0,var_matrix = compute_components_3(results)
    df_g = compute_g_coefficients_3(var_matrix,n_s,n_d)

print("Estimated variance components:")
display(df.round(6))
print("After setting negative to 0:")
display(df_0.round(6))
print("G-coefficients")
display(df_g.round(6))

#file_num = len(os.listdir(results_dir))+1 #if you want to save with number of files in directory in name
def save_results(df_0,df_g,results):
    results_dir = os.path.join(notebook_path,"jwn/results")
    os.chdir(results_dir)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    directory = "results_"+str(facets)+"_"+str(n_p)+"_"+str(timestr)
    path = os.path.join(results_dir, directory)
    os.mkdir(path)
    os.chdir(path)
    with open("omh_var_components_"+str(timestr)+".pckl","wb") as file:
        pickle.dump(df_0,file)
    with open("omh_g_cofficients_"+str(timestr)+".pckl","wb") as file:
        pickle.dump(df_g,file)
    with open("omh_var_results_"+str(timestr)+".pckl","wb") as file:
        pickle.dump(results,file)

save_results(df_0,df_g,results)