from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from tqdm import tqdm

obs = [250,500,1000]
seed_list = np.linspace(100,1000,10,dtype=int)
depth_list = np.linspace(100,1000,10,dtype=int)
B = 10
V1 = np.array([-1,0,1])
W1 = np.array([-1,0,1])
X1 = np.array([-1,0,1])
Z1 = np.array([-1,0,1])
y1 = 2*V1-W1**2+(X1-1)**2-3*Z1+2

X_true = np.array([1])
V_true = np.array([1])
Z_true = np.array([1])
W_true = np.array([1])
test_true = np.hstack([V_true, W_true, X_true, Z_true])
y_true = 2*V_true-W_true**2+(X_true-1)**2-3*Z_true+2

K = len(seed_list)
L = len(depth_list)
y_pred = np.zeros(shape=(B,K,L))
err = np.zeros(shape=(B,K,L))
vars = np.zeros(3)
std=np.zeros(3)
eta = np.zeros(3)
for j in range(len(obs)):
    N = obs[j]
    for i in tqdm(range(B)):
        V = 5*np.random.normal(0,1,size=(N,1))
        W = 5*np.random.normal(0,1,size=(N,1))
        X = 5*np.random.normal(0,1,size=(N,1))
        Z = 5*np.random.normal(0,1,size=(N,1))

        y = (2*V-W**2+(X-1)**2-3*Z+2 + 1.0*np.random.normal(loc=0,scale=1,size=(N,1))).ravel()

        train  = np.hstack([V, W, X, Z])

        if (i == 0):

            vrange = np.linspace(np.min(V),np.max(V),1000)
            wrange = np.linspace(np.min(W),np.max(W),1000)
            xrange = np.linspace(np.min(X),np.max(X),1000)
            zrange = np.linspace(np.min(Z),np.max(Z),1000)

            fig = plt.figure(figsize=(16,16))
            fig.suptitle('Data distribution (N = '+str(N)+')')

            ax1 = fig.add_subplot(221)
            ax1.scatter(V,y)
            ax1.plot(vrange,2*vrange,c='r')
            ax1.scatter(V1,y1,c=['green','yellow','purple'],marker='s',s=75)
            ax1.set_xlabel(fr'$x_1$')
            ax1.set_ylabel('y')
            ax1.title.set_text(fr'$y=2x_1$')

            ax2 = fig.add_subplot(222)
            ax2.scatter(W,y)
            ax2.plot(wrange,-(wrange**2),c='r')
            ax2.set_xlabel(fr'$x_2$')
            ax2.set_ylabel('y')
            ax2.title.set_text(fr'$y=-(x_2)^2$')
            ax2.scatter(W1,y1,c=['green','yellow','purple'],marker='s',s=75)

            ax3 = fig.add_subplot(223)
            ax3.scatter(X,y)
            ax3.plot(xrange,(xrange-1)**2,c='r')
            ax3.set_xlabel(fr'$x_3$')
            ax3.set_ylabel('y')
            ax3.title.set_text(fr'$y=(x_3-1)^2$')
            ax3.scatter(X1,y1,c=['green','yellow','purple'],marker='s',s=75)

            ax4 = fig.add_subplot(224)
            ax4.scatter(Z,y)
            ax4.plot(zrange,-3*zrange,c='r')
            ax4.title.set_text('y=-3Z')
            ax4.set_xlabel(fr'$x_4$')
            ax4.set_ylabel('y')
            ax4.title.set_text(fr'$y=-3x_4$')
            ax4.scatter(Z1,y1,c=['green','yellow','purple'],marker='s',s=75)

            plt.savefig('dist_'+str(N))
        for k in range(K):
            for l in range(L):
                reg = RandomForestRegressor(n_estimators=depth_list[l],random_state=seed_list[k],n_jobs=-1)
                reg.fit(train,y)
                y_pred[i,k,l] = reg.predict(test_true.reshape(1,-1))
                err[i,k,l] = y_pred[i,k,l] - y_true
    error = np.mean(err,axis=(1,2))
    vars[j] = np.var(error)
    std[j] = np.std(error)
    eta[j] = np.mean(error)
display(vars)
display(eta)

