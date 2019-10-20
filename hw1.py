import math
import numpy as np
import sys
import matplotlib.pyplot as plt

def read_point(filename,n):
    A=[]
    B=[]
    minimum=10000000
    maximum=-10000000
    fp = open(filename, "r")
    line = fp.readline()
    while line:
        sp=line.split(",")
        #x.append(float(sp[0]))
        #y.append(float(sp[1]))
        if float(sp[0])<minimum:
            minimum=float(sp[0])
        if float(sp[0])>maximum:
            maximum=float(sp[0])
        A.append([float(sp[0])**i for i in reversed(range(n))])
        B.append([float(sp[1])])
        line = fp.readline()
    fp.close()
    return A,B,minimum,maximum

def lu_decomposition(A):
    n = len(A)
    L= [[0.0 for i in range(n)] for j in range(n)]
    U= [[0.0 for i in range(n)] for j in range(n)]
    """for i in range(n):
        L[i][0]=A[i][0]
    for j in range(1,n):
        U[0][j]=A[0][j]/L[0][0]
    for j in range(1,n-1):
        for i in range(j,n):
            s1 = sum(U[k][j] * L[i][k] for k in range(j-1))
            L[i][j] = A[i][j] - s1
        for k in range(j+1, n):
            s2 = sum(U[i][k] * L[j][i] for i in range(j-1))
            L[j][k] = (A[j][k] - s2) / L[j][j]
    L[n-1][n-1]=A[n-1][n-1]-sum(U[k][n-1] * L[n-1][k] for k in range(n-1))"""
    for j in range(n):
        L[j][j] = 1.0
        for i in range(j+1):
            U[i][j] = A[i][j] - sum(U[k][j] * L[i][k] for k in range(i))
        for i in range(j, n):
            L[i][j] = (A[i][j] - sum(U[k][j] * L[i][k] for k in range(j))) / U[j][j]
    return L,U

def find_inverse(L,U):
    n=len(L)
    identity=[[float(i==j) for i in range(n)] for j in range(n)]
    d=[0.0 for i in range(len(identity))]
    result=[[0.0 for i in range(len(identity[0]))] for j in range(n)]
    for i in range(len(identity[0])):
        d=[0.0 for k in range(len(identity))]
        for j in range(n):
            if j==0:
                d[0]=identity[j][i]/L[0][0]
            else:
                d[j]=(identity[j][i]-sum(L[j][k] * d[k] for k in range(j)))/L[j][j]
        for j in reversed(range(n)):
            if j==n-1:
                result[j][i]=d[j]/U[j][j]
            else:
                result[j][i]=(d[j]-sum(U[j][k] * result[k][i] for k in range(j+1,n)))/U[j][j]
    return result

def newton_method(n,x,y):
    theta=[[0.0] for i in range(n)]
    #print(theta)
    f=np.matmul(np.transpose(np.matmul(x,theta)-y),np.matmul(x,theta)-y)
    delta=2*np.matmul(np.transpose(x),np.matmul(x,theta))-2*np.matmul(np.transpose(x),y)
    #print(delta)
    hessian=2*np.matmul(np.transpose(x),x)
    #print(hessian)
    L,U=lu_decomposition(hessian)
    inv=find_inverse(L,U)
    theta=theta-np.matmul(inv,delta)
    f=np.matmul(np.transpose(np.matmul(x,theta)-y),np.matmul(x,theta)-y)
    #print(theta)
    return theta,f


if __name__ == "__main__":
    n=int(sys.argv[2])
    lamda=int(sys.argv[3])
    A,B,minimum,maximum=read_point(sys.argv[1],n)
    theta,n_err=newton_method(n,A,B)
    lamda_matrix=[[lamda*float(i ==j) for i in range(n)] for j in range(n)]
    L,U=lu_decomposition(np.matmul(np.transpose(A),A)+lamda_matrix)
    inv=find_inverse(L,U)
    coeff=np.matmul(np.matmul(inv,np.transpose(A)),B)
    print("LSE:")
    print("Fitting line:",end=' ')
    for i in range(len(coeff)):
        if coeff[i][0]==0:
            continue
        if i!=len(coeff)-1:
            print(str(round(coeff[i][0],11))+"x^"+str(len(coeff)-1-i),end=' ')
            if coeff[i+1][0]>=0:
                print("+",end=" ")
        else:
            print(str(round(coeff[i][0],11)))
    #print(coeff)
    err=np.matmul(np.transpose(np.matmul(A,coeff)-B),np.matmul(A,coeff)-B)
    print("Total error:",round(err[0][0],10))
    x=np.linspace(minimum-1,maximum+1,100)
    y=x-x
    for i in range(len(coeff)):
        y+=coeff[i][0]*(x**(len(coeff)-1-i))
    plt.subplot(2,1,1)
    plt.title("LSE")
    plt.plot(x,y)
    plt.plot(np.array(A)[:,-2],np.reshape(np.array(B),(len(B),-1)),'ro')

    print("Newton's method:")
    print("Fitting line:",end=' ')
    for i in range(len(theta)):
        if theta[i][0]==0:
            continue
        if i!=len(theta)-1:
            print(str(round(theta[i][0],11))+"x^"+str(len(theta)-1-i),end=' ')
            if theta[i+1][0]>=0:
                print("+",end=" ")
        else:
            print(str(round(theta[i][0],11)))
    print("Total error:",round(n_err[0][0],10))
    plt.subplot(2,1,2)
    plt.title("Newton's method")
    nn=x-x
    for i in range(len(theta)):
        nn+=theta[i][0]*(x**(len(theta)-1-i))
    plt.plot(x,nn)
    plt.plot(np.array(A)[:,-2],np.reshape(np.array(B),(len(B),-1)),'ro')
    plt.show()


