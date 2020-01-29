import numpy as np
import cv2
import sys
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from PIL import Image
import random

def computeSSE(data, centers, clusterID):
    #inputs are all numpy arrays
    sse = 0
    nData = len(data)
    for i in range(nData):
        c = clusterID[i]
        sse += np.sum(np.square(data[i]-centers[c]))
    return sse

def updateClusterID(data, centers):
    #all inputs are numpy arrays
    nData = len(data)
    nCenters = len(centers)
    clusterID = [0] * nData
    dis_Centers = [0] * nCenters
    for i in range(nData):
        for c in range(nCenters):
            dis_Centers[c] = np.sum(np.square(data[i]-centers[c]))
        clusterID[i] = dis_Centers.index(min(dis_Centers))
    return clusterID

def updateCenters(data, clusterID, K):
    nDim = len(data[0])
    centers = [[0] * nDim for i in range(K)]
    ids = sorted(set(clusterID))
    #print("ids:",ids)
    for id in ids:
        # get the index from clusterID where data points belong to the same cluster
        indices = [i for i, j in enumerate(clusterID) if j == id]
        cluster = [data[i] for i in indices]
        if len(cluster) == 0:
            centers[id] = [0] * nDim
        else:
            centers[id] = [float(sum(col))/len(col) for col in zip(*cluster)]
    #print("centers:",np.array(centers))
    #print("centers shape",np.array(centers).shape)
    return np.array(centers)

def kpp_init(data,K):
    #return centers
    l=len(data)
    a=random.randint(0,l)
    centers=data[a].reshape(1,l)
    print(a)
    #print(centers)
    for i in range(K-1):
        nData = len(data)
        nCenters = i+1
        #print(nData,nCenters)
        d=[0]*nData
        s=0
        dis_Centers = [0] * nCenters
        for i in range(nData):
            for c in range(nCenters):
                dis_Centers[c] = np.sum(np.square(data[i]-centers[c]))
            d[i]=min(dis_Centers)
            s+=d[i]
        seed=random.uniform(0, 2)/10*s
        #print("here")
        for j,dis in enumerate(sorted(d,reverse=True)):
            seed-=dis
            if seed>0:
                continue
            else:
                print(d.index(dis))
                centers=np.concatenate((centers,data[d.index(dis)].reshape(1,l)))
                break
    #print(centers.shape)
    return centers





def kmeans(data, centers, maxIter = 30, tol = 0.00001):
    print("original centers:",centers)
    nData = len(data)
    if nData == 0:
        return []
    K = len(centers)
    clusterID = [0] * nData
    if K >= nData:
        for i in range(nData):
            clusterID[i] = i
        return clusterID
    nDim = len(data[0])
    lastDistance = 0
    for iter in range(maxIter):
        clusterID = updateClusterID(data, centers)
        unique, counts = np.unique(clusterID, return_counts=True)
        print("clusterID:",dict(zip(unique, counts)))
        centers = updateCenters(data, clusterID, K)
        curDistance = computeSSE(data, centers, clusterID)
        rescaled = (255.0 // (K-1) * (np.reshape(clusterID,(100,100)))).astype(np.uint8)
        im = Image.fromarray(rescaled)
        im.save(str(iter)+'.jpg')
        if iter==0:
            lastDistance = curDistance
            print("# of iterations:", iter)
            print("SSE = ", curDistance)
            continue
        elif abs(lastDistance - curDistance)/lastDistance < tol:
            print("# of iterations:", iter)
            print("SSE = ", curDistance)
            return clusterID
        else:
            print("# of iterations:", iter)
            print("SSE = ", curDistance)
        lastDistance = curDistance
    print("# of iterations:", iter)
    print("SSE = ", curDistance)
    return clusterID

def kernel(img, gamma1,gamma2):
    l=img.shape[0]*img.shape[1]
    s_img=np.array([[i//img.shape[0],i%img.shape[0]] for i in range(l)])
    #print(s_img)
    #print(gram.shape)
    s_gram=np.square(squareform(pdist(s_img,'euclidean')))
    #print(img.reshape(-1,img.shape[2]))
    c_gram=np.square(squareform(pdist(img.reshape(-1,img.shape[2]),'euclidean')))
    gram=np.exp(-gamma1*s_gram-gamma2*c_gram)
    #print(gram.shape)
    return gram

if __name__=="__main__":
    img=cv2.imread("image1.png")
    gram=kernel(img,0.0005,0.002)#0.001 0.0005
    #print(gram)
    nDim = len(gram[0])
    K = 2  # cluster number
    print('K=',K)
    centers = []
    for i in range(K):
        centers.append(gram[i*5000])
    centers=np.array(centers)
    #print(centers.shape)
    #centers=kpp_init(gram,K)
    results = kmeans(gram, centers)



