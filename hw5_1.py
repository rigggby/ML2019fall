import sys
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import struct
from numpy.linalg import inv
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize

def visualize(mean, cov, X, X_train, Y_train):
    mean = mean.ravel()
    X = X.ravel()
    intv = 1.96 * np.sqrt(np.diag(cov))#95% interval
    plt.fill_between(X, mean + intv, mean - intv, alpha=0.1,facecolor = "orange")
    plt.plot(X,mean,label='mean')
    plt.scatter(X_train, Y_train)
    plt.legend()
    plt.show()

def rq_kernel(X1, X2, l=1.0, sigma_f=1.0,alpha=10):
    d = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * (1+1/ (2*alpha*l**2) * d)**(-alpha)

def posterior_predictive(data_point, X_train, Y_train, l=1.0, sigma_f=1.0, alpha=10,sigma_y=0.5):
    K=rq_kernel(X_train, X_train, l, sigma_f,alpha) + sigma_y**2*np.eye(len(X_train))
    K_tes=rq_kernel(X_train, data_point, l, sigma_f,alpha)
    K_cov=rq_kernel(data_point, data_point, l, sigma_f,alpha) + sigma_y* np.eye(len(data_point))
    mean=K_tes.T.dot(inv(K)).dot(Y_train)
    cov=K_cov - K_tes.T.dot(inv(K)).dot(K_tes)
    return mean, cov

def nll(X_train, Y_train,noise):
    def func(theta):
        K = rq_kernel(X_train, X_train, l=theta[0], sigma_f=theta[1],alpha=theta[2]) + noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + 0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + 0.5 * len(X_train) * np.log(2*np.pi)
    return func

if __name__ == "__main__":
    A=[]
    B=[]
    X = np.arange(-60,60, 0.2).reshape(-1, 1)
    minimeanm=10000000
    maximeanm=-10000000
    fp = open("input.data", "r")
    line = fp.readline()
    while line:
        sp=line.split(" ")
        #x.append(float(sp[0]))
        #y.append(float(sp[1]))
        #print(sp)
        if float(sp[0])<minimeanm:
            minimeanm=float(sp[0])
        if float(sp[0])>maximeanm:
            maximeanm=float(sp[0])
        A.append([float(sp[0])])
        B.append([float(sp[1])])
        line = fp.readline()
    fp.close()
    l=sigma_f=1
    alpha=10
    res = minimize(nll(np.array(A), np.array(B),0.5), [1, 1,10],bounds=((1e-5, None), (1e-5, None),(1e-5, None)),
        method='L-BFGS-B')
    print(res.x)
    l,sigma_f,alpha=res.x
    mean, cov = posterior_predictive(X, np.array(A),np.array(B),l=l,sigma_f=sigma_f,alpha=alpha)
    visualize(mean, cov, X, X_train=np.array(A), Y_train=np.array(B))

    """plt.scatter(A, B)
    plt.title('Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()"""