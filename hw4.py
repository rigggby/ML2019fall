import sys
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import struct

train_data_path = 'train-images.idx3-ubyte'
tr_y_path = 'train-labels.idx1-ubyte'
test_data_path = 't10k-images.idx3-ubyte'
test_label_path = 't10k-labels.idx1-ubyte'

def mnist_pre():
    # tr_x
    with open(train_data_path, 'rb') as f:
        data = f.read(16)
        des,img_nums,row,col = struct.unpack_from('>IIII', data, 0)
        train_x = np.zeros((img_nums, row*col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                train_x[index,:] = np.array(struct.unpack_from('>784B', data, 0)).reshape(1,784)
        f.close()
    # train label
    with open(tr_y_path, 'rb') as f:
        data = f.read(8)
        des, label_nums = struct.unpack_from('>II', data, 0)
        train_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            train_y[index, :] = np.array(struct.unpack_from('>B', data, 0)).reshape(1, 1)
        f.close()
    # test_img
    with open(test_data_path, 'rb') as f:
        data = f.read(16)
        des, img_nums, row, col = struct.unpack_from('>IIII', data, 0)
        test_x = np.zeros((img_nums, row * col))
        for index in range(img_nums):
            data = f.read(784)
            if len(data) == 784:
                test_x[index, :] = np.array(struct.unpack_from('>784B', data, 0)).reshape(1, 784)
        f.close()
    # test label
    with open(test_label_path, 'rb') as f:
        data = f.read(8)
        des, label_nums = struct.unpack_from('>II', data, 0)
        test_y = np.zeros((label_nums, 1))
        for index in range(label_nums):
            data = f.read(1)
            test_y[index, :] = np.array(struct.unpack_from('>B', data, 0)).reshape(1, 1)
        f.close()
    return train_x, train_y,test_x, test_y

def gen_gau(u, v):
    upper, V = np.random.uniform(0, 1, 2)
    return u+math.sqrt(v)*math.sqrt(-2 * math.log(upper)) * math.cos(2*math.pi*V)

def gen_poly(a, w):
    e=gen_gau(0,a)
    while True:
        x = random.uniform(-1,1)
        y = sum([w[i]*(x**i) for i in range(len(w))]) + e
        e=gen_gau(0,a)
        yield x,y

def logestic(x):
    return 1.0 / (1 + np.exp(-x))

def GradientDescent(N, pts):
    X = np.hstack((np.ones(2*N).reshape(-1, 1), pts))
    #print(X)
    w = np.zeros(3)
    y = np.hstack((np.zeros(N), np.ones(N)))
    #print(y)
    count = 0
    epsilon = 0.001
    iterMax = 10000
    while count < iterMax:
        gradient = X.T.dot(logestic(X.dot(w)) - y)
        w -= gradient
        if np.all(np.absolute(gradient) < epsilon):
            break
        count+=1
    p_result = logestic(X.dot(w))
    p_result[:] += 0.5
    p_result = p_result.astype(np.int8)
    #print(p_result)
    #print(y)
    class0 = (p_result[:N] == y[:N])
    class1 = (p_result[N:] == y[N:])
    TP = np.count_nonzero(class0)
    TN = np.count_nonzero(class1)
    FN = class0.shape[0] - TP
    FP = class1.shape[0] - TN
    return w, (TP, FN, FP, TN)

def Newton(N, pts):
    X = np.hstack((np.ones(2*N).reshape(-1, 1), pts))
    w = np.zeros(3)
    y = np.hstack((np.zeros(N), np.ones(N)))
    count = 0
    epsilon =0.001
    iterMax = 10000
    while count < iterMax:
        D = np.diag(logestic(X.dot(w))).dot(np.eye(2*N)-np.diag(logestic(X.dot(w))))
        H = X.T.dot(D).dot(X)
        try:
            delta = np.linalg.inv(H).dot(X.T.dot(logestic(X.dot(w)) - y))
        except:
            delta = X.T.dot(logestic(X.dot(w)) - y)
        w -= delta
        if np.all(np.absolute(delta) < epsilon):
            break
        count += 1
    p_result = logestic(X.dot(w))
    p_result[:] += 0.5
    p_result = p_result.astype(np.int8)
    class0 = (p_result[:N] == y[:N])
    class1 = (p_result[N:] == y[N:])
    TP = np.count_nonzero(class0)
    TN = np.count_nonzero(class1)
    FN = class0.shape[0] - TP
    FP = class1.shape[0] - TN
    return w, (TP, FN, FP, TN)

def Estep(images, P, r):
    w = np.array([r]*N)
    logw = LOG(w) + images.dot(LOG(P.T)) + (1-images).dot(LOG(1-P.T))
    logw = (logw.T - np.max(logw, axis=1)).T
    w = np.exp(logw)
    w = (w.T / np.sum(w, axis=1)).T
    return w

def Mstep(images, w):
    r = np.sum(w, axis=0) / N
    for d in range(D):
        P[:, d] = w.T.dot(images[:, d]) / np.sum(w, axis=0)
    return r, P

def LOG(m):
    return np.log(np.where(m > 10 ** -10, m, 10 ** -10))

def show_result(P):
    result = P.reshape((10,28,28))
    for i in range(10):
        print('class'+' {}:'.format(i))
        for p in range(28):
            for j in range(28):
                print(int(result[i][p][j]>=0.5),end=' ')
            print(' ')

if __name__ == "__main__":
    number=int(sys.argv[1])
    if number==0:
        N =int(sys.argv[10])
        mx1, vx1 = int(sys.argv[2]),int(sys.argv[3])
        my1, vy1 = int(sys.argv[4]),int(sys.argv[5])
        mx2, vx2 = int(sys.argv[6]),int(sys.argv[7])
        my2, vy2 = int(sys.argv[8]),int(sys.argv[9])
        pts1 = np.array([[gen_gau(mx1,vx1), gen_gau(my1,vy1)] for _ in range(N)])
        pts2 = np.array([[gen_gau(mx2,vx2), gen_gau(my2,vy2)] for _ in range(N)])
        pts = np.vstack((pts1, pts2))
        w_G, confusion_G = GradientDescent(N, pts)
        w_N, confusion_N = Newton(N, pts)
        X = np.hstack((np.ones(2*N).reshape(-1, 1), pts)) # 2N * 3
        x_min = np.min(np.minimum(pts1[:,0], pts2[:,0]))-5
        x_max = np.max(np.maximum(pts1[:,0], pts2[:,0]))+5
        y_min = np.min(np.minimum(pts1[:,1], pts2[:,1]))-5
        y_max = np.max(np.maximum(pts1[:,1], pts2[:,1]))+5
        print('Gradient descent:\nweight:')
        for i in range(3):
            print(w_G[i])
        print('\nConfusion Matrix:')
        print('               ', 'Predict cluster 1 ', 'Predict cluster 2 ')
        print('Is cluster 1       ', confusion_G[0],'        ',confusion_G[1])
        print('Is cluster 2       ', confusion_G[2],'        ',confusion_G[3])
        print('Sensitivity (Successfully predict cluster 1):', confusion_G[0] / (confusion_G[0] + confusion_G[1]))
        print('Specificity (Successfully predict cluster 2):', confusion_G[3] / (confusion_G[2] + confusion_G[3]))
        print('\n\nNewton\'s method:\nweight:')
        for i in range(3):
            print(w_G[i])
        print('\nConfusion Matrix:')
        print('               ', 'Predict cluster 1 ', 'Predict cluster 2 ')
        print('Is cluster 1       ', confusion_N[0],'        ',confusion_N[1])
        print('Is cluster 2       ', confusion_N[2],'        ',confusion_N[3])
        print('Sensitivity (Successfully predict cluster 1):', confusion_N[0] / (confusion_N[0] + confusion_N[1]))
        print('Specificity (Successfully predict cluster 2):', confusion_N[3] / (confusion_N[2] + confusion_N[3]))
        plt.subplot(131)
        plt.title('Ground truth')
        plt.scatter(pts1[:, 0], pts1[:, 1], color = 'r')
        plt.scatter(pts2[:, 0], pts2[:, 1], color = 'b')
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.subplot(132)
        plt.title('Gradient descent')
        p_result1 = logestic(X.dot(w_G))
        p_result1[:] += 0.5
        p_result1 = p_result1.astype(np.int8)
        Gpts1x = []
        Gpts1y = []
        Gpts2x = []
        Gpts2y = []
        for i in range(2*N):
            if not p_result1[i]:
                Gpts1x.append(pts[i, 0])
                Gpts1y.append(pts[i, 1])
            else:
                Gpts2x.append(pts[i, 0])
                Gpts2y.append(pts[i, 1])
        plt.scatter(Gpts1x, Gpts1y, color = 'r')
        plt.scatter(Gpts2x, Gpts2y, color = 'b')
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.subplot(133)
        plt.title('Newton\'s method')
        p_result2 = logestic(X.dot(w_N))
        p_result2[:] += 0.5
        p_result2 = p_result2.astype(np.int8)
        N1x = []
        N1y = []
        N2x = []
        N2y = []
        for i in range(2*N):
            if not p_result2[i]:
                N1x.append(pts[i, 0])
                N1y.append(pts[i, 1])
            else:
                N2x.append(pts[i, 0])
                N2y.append(pts[i, 1])
        plt.scatter(N1x, N1y, color = 'r')
        plt.scatter(N2x, N2y, color = 'b')
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.show()
    else:
        tr_x, tr_y, te_x, te_y =mnist_pre()
        D = 28 * 28
        K = 10
        N = len(tr_x)
        images =tr_x[:][:]/128
        threshold = 0.01
        c = 0
        P = np.random.uniform(0.45, 0.55, (K, D))
        r = np.full(10, 0.1)
        while c < 10:
            w = Estep(images, P, r)
            r, P = Mstep(images, w)
            c += 1
        show_result(P)
        result = np.zeros((10,10))
        w_max = np.argmax(w, axis=1)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = np.count_nonzero(tr_y[w_max == i] == j)
        class_digit, class_count = np.unique(tr_y, return_counts=True)
        label_predict = np.argmax(result, axis=0)
        tt_t=tt_f=0
        for label, predict in enumerate(label_predict):
            TP = np.count_nonzero( tr_y[w_max == predict] == label )
            FP = np.count_nonzero( tr_y[w_max == predict] != label )
            FN = np.count_nonzero( tr_y[w_max != predict] == label )
            TN = np.count_nonzero( tr_y[w_max != predict] != label )
            tt_t=tt_t+TP
            tt_f=tt_f+FN
            print('\nConfusion Matrix:')
            print(' '*20, ' {:<15}{:>2} {:<15}{:>2} '.format('Predict number', label, 'Predict not number', label))
            print(' {:>15} {:>2}{:15}{:20}'.format('Is number', label, TP, FN))
            print(' {:>15} {:>2}{:15}{:20}'.format('Isn\'t number', label, FP, TN), '\n')
            print('Sensitivity (Successfully predict number {})    : {}'.format(label, TP / (TP + FN)))
            print('Specificity (Successfully predict not number {}): {}'.format(label, TN / (FP + TN)))
        err_rate=tt_f/(tt_t+tt_f)
        print("Total error rate:",err_rate)





