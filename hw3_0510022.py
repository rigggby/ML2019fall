import sys
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def gen_gau(u, v):
    upper, V = np.random.uniform(0, 1, 2)
    return u+math.sqrt(v)*math.sqrt(-2 * math.log(upper)) * math.cos(2*math.pi*V)

def gen_poly(a, w):
    e=gen_gau(0,a)
    while True:
        x = random.uniform(-1,1)
        y = sum([w[i]*(x**i) for i in range(len(w))]) + e
        e=gen_gau(0,a)
        yield x, y

def SequentialEstimator(m, s):
    print('Data point source function: N~({:.2f}, {:.2f})'.format(m, s))
    x=gen_gau(m,s)
    n=1
    p_mean=x
    var=M2=0
    while n <1000000:
        x=gen_gau(m,s)
        n+=1
        mean = p_mean + (x - p_mean)/n
        M2 = M2+(x-p_mean)*(x-mean)
        var = M2/(n-1)
        print('Add data point: {} \nMean={} Variance={}'.format(x,mean, var))
        if abs(p_mean-mean) < 0.00001:
            break
        p_mean=mean

def Bayesian(b,n,a, w):
    gen=gen_poly(a, w)
    s0 = b * np.eye(n)
    m0 = np.zeros(n).reshape(-1, 1)
    x_ = []
    y_ = []
    s_ = []
    m_ = []
    pred = []
    posterior = []
    max_iter = 10000
    num = 0
    while num < max_iter:
        num+=1
        x,y=next(gen)
        x_.append(x)
        y_.append(y)
        X=np.array([[x**i for i in range(n)]])
        s1=(1/a) * X.T.dot(X)+s0
        m1=np.linalg.inv(s1).dot((1/a)*y* X.T+s0.dot(m0))
        if num==10 or num==50 or num==max_iter:
            m_.append(m1)
            s_.append(s1)
        if num<=5 or num>max_iter-5:
            y_mean=X.dot(m1)[0,0]
            y_var=(a+X.dot(np.linalg.inv(s1)).dot(X.T))[0, 0]
            posterior.append((m1, s1))
            pred.append((y_mean, y_var))
        m0=m1
        s0=s1
    return s_, m_, posterior, pred,x_, y_

def show_result(b, n,a, w):
    s_, m_, posterior, pred,x_, y_ = Bayesian(b,n,a,w)
    I = np.eye(n)
    X = np.linspace(-2, 2, num=100)
    Y = []
    M = []
    upper = []
    lower = []
    for jj in range(4):
        for x in X:
            d = np.array([[x**i for i in range(n)]])
            b = np.eye(n)
            if jj == 0:
                M.append(0)
                v = (a + d.dot(np.linalg.inv(s_[-1])).dot(d.T))[0, 0]
                y = sum([w[i] * x**i for i in range(n)])
                Y.append(y)
                upper.append(y+0.5*v)
                lower.append(y-0.5*v)
            else:
                v = (a + d.dot(np.linalg.inv(s_[jj-1])).dot(d.T))[0, 0]
                m = (d.dot(m_[jj-1]))[0, 0]
                M.append(m)
                upper.append(m+0.5*v)
                lower.append(m-0.5*v)
    for i, (post, pred) in enumerate(zip(posterior, pred)):
        print('Add data point ({}, {}):\n'.format(x_[i], y_[i]))
        print('Posterior mean:')
        for data in post[0]:
            print(data[0])
        print('\nPosterior variance:')
        for data in np.linalg.inv(post[1]):
            for kk in data:
                print('{:.10f}'.format(kk),end=' ')
            print('\n')
        print('\nPredictive distribution N~({}, {})\n\n'.format(pred[0], pred[1]))
        print('-'*50)
        if i == 4:
            print('\n\nomitted\n\n')
            print('-'*50,'\n\n')



    plt.figure(figsize=(8,8))
    plt.subplot(2, 2, 1)
    plt.gca().set_title('ground truth')
    plt.plot(X, upper[:100], c="r")
    plt.plot(X, lower[:100], c="r")
    plt.plot(X, Y, c="k")
    plt.xlim((-2, 2))
    plt.ylim((-20, 20))

    plt.subplot(2, 2, 2)
    plt.gca().set_title('predict result')
    plt.plot(X, upper[300:], c="r")
    plt.plot(X, lower[300:], c="r")
    plt.plot(X, M[300:], c="k")
    plt.scatter(x_, y_)
    plt.xlim((-2, 2))
    plt.ylim((-20, 20))

    plt.subplot(2, 2, 3)
    plt.gca().set_title('after 10 incomes')
    plt.plot(X, upper[100:200], c="r")
    plt.plot(X, lower[100:200], c="r")
    plt.plot(X, M[100:200], c="k")
    plt.scatter(x_[:10], y_[:10])
    plt.xlim((-2, 2))
    plt.ylim((-20, 20))

    plt.subplot(2, 2, 4)
    plt.gca().set_title('after 50 incomes')
    plt.plot(X, upper[200:300], c="r")
    plt.plot(X, lower[200:300], c="r")
    plt.plot(X, M[200:300], c="k")
    plt.scatter(x_[:50], y_[:50])
    plt.xlim((-2, 2))
    plt.ylim((-20, 20))
    plt.show()

if __name__ == "__main__":
    mode=int(sys.argv[1])
    if mode==1:
        show_result(1,3,3, [1, 2, 3])
    else:
        SequentialEstimator(3,5)



