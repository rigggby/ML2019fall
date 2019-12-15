from libsvm.python.svmutil import *
from libsvm.python.svm import *
import numpy as np

if __name__ == "__main__":
    #1
    X_train=np.loadtxt("X_train.csv",dtype=np.float,delimiter=',')
    X_test=np.loadtxt("X_test.csv",dtype=np.float,delimiter=',')
    Y_train=np.loadtxt("Y_train.csv",dtype=np.int,delimiter=',')
    Y_test=np.loadtxt("Y_test.csv",dtype=np.int,delimiter=',')
    #print(Y_test.shape)
    m = svm_train(Y_train, X_train, '-t 1')#rbf
    #res = svm_predict(Y_test, X_test, m)
    print(svm_predict(Y_test, X_test, m))

    #2
    X_train=np.loadtxt("X_train.csv",dtype=np.float,delimiter=',')
    X_test=np.loadtxt("X_test.csv",dtype=np.float,delimiter=',')
    Y_train=np.loadtxt("Y_train.csv",dtype=np.int,delimiter=',')
    Y_test=np.loadtxt("Y_test.csv",dtype=np.int,delimiter=',')
    #print(Y_test.shape)
    bac=0
    b_log2c=0
    b_coef=0
    b_log2g=0
    b_d=0
    """print("=============Linear===============")
    for log2c in range(-5,10):
      res = svm_train(Y_train, X_train, "-v 5 -t 0 "  + "-c " + str(2**log2c))
      print("log2c={},acc={}\n".format(log2c,res))
      if res > bac:
        bac=res
        b_log2c=log2c
    print("best result:{},best c:{}".format(bac,2**b_log2c))
    """
    """
    print("=============Polynomial===============")
    for log2c in range(-3,2):
      for log2g in range(-2,1):
        for d in range(4):
          for coeff in range(4):
            res = svm_train(Y_train, X_train, "-v 3 -t 1 "  + "-c " + str(2**log2c)+" -g "+str(2**log2g)+" -d "+str(d)+" -r "+str(coeff))
            print("log2c={},log2g={},d={},coeff={},acc={}\n".format(log2c,log2g,d,coeff,res))
            if res > bac:
              bac=res
              b_log2c=log2c
              b_coeff=coeff
              b_log2g=log2g
              b_d=d
    print("best result:{},c:{},gamma:{},d:{},coeff:{}".format(bac,2**b_log2c,2**b_log2g,b_gamma,b_d,b_coeff))
    """
    print("=============RBF===============")
    for log2c in range(-5,6):
      for log2g in range(-3,3):
            res = svm_train(Y_train, X_train, "-v 3 -t 1 "  + "-c " + str(2**log2c)+" -g "+str(2**log2g))
            print("log2c={},log2g={},acc={}\n".format(log2c,log2g,res))
            if res > bac:
              bac=res
              b_log2c=log2c
              b_log2g=log2g
    print("best result:{},c:{},gamma:{}".format(bac,2**b_log2c,2**b_log2g))

    #3
    def kernel(X1, X2, l=1.0,gamma=0.25):
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma* sqdist)

    X_train=np.loadtxt("X_train.csv",dtype=np.float,delimiter=',')
    X_test=np.loadtxt("X_test.csv",dtype=np.float,delimiter=',')
    Y_train=np.loadtxt("Y_train.csv",dtype=np.int,delimiter=',')
    Y_test=np.loadtxt("Y_test.csv",dtype=np.int,delimiter=',')
    #print(Y_test.shape)
    K_train=np.zeros((len(X_train),len(X_train)+1))
    K_train[:,1:]=0.5*np.dot(X_train,X_train.T)+0.5*kernel(X_train, X_train)
    K_train[:,:1]=np.arange(len(X_train))[:,np.newaxis]+1
    m = svm_train(Y_train,[list(row) for row in K_train], '-t 4')#user defined

    K_test=np.zeros( (len(X_test),len(X_train)+1))
    K_test[:,1:]=0.5*np.dot(X_test,X_train.T)+0.5*kernel(X_test, X_train)
    K_test[:,:1]=np.arange(len(X_test))[:,np.newaxis]+1
    #res = svm_predict(Y_test, X_test, m)
    print(svm_predict(Y_test, [list(row) for row in K_test], m))





