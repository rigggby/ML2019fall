import numpy as np
import math
import struct
import matplotlib.pyplot as plt
import sys

train_data_path = 'train-images.idx3-ubyte'
train_label_path = 'train-labels.idx1-ubyte'
test_data_path = 't10k-images.idx3-ubyte'
test_label_path = 't10k-labels.idx1-ubyte'

def square(list):
    return [i ** 2 for i in list]

def square_root(list):
    return [math.sqrt(i) for i in list]

def fl(list):
    return [float(i) for i in list]

def dev(list1,list2):
    result=[]
    for i in range(len(list1)):
        result.append(list1[i]/list2[i])
    return result

class NaiveBayes(object):
    def log_gaussian(x,u,v):
        return ((-1*(((x - u)**2) / (2*v) ) ) -0.5*math.log(2*pi*v))

    def fit(self, X, Y,class_number):
        self.prior = np.zeros(class_number)
        self.mean=np.zeros((class_number,len(X[0])))
        self.var=np.zeros((class_number,len(X[0])))
        for c in Y:
            self.prior[int(c[0])]+=1
        self.prior= self.prior/ len(Y)
        #print(self.prior)
        index=0
        for i in X:
            self.mean[int(Y[index][0])]+=fl(i)
            self.var[int(Y[index][0])]+=square(fl(i))
            index+=1
        index=0
        for i in self.prior:
            self.mean[index]=self.mean[index]/(len(Y)*i)
            self.var[index]=self.var[index]/(len(Y)*i)
            self.var[index]-=square(self.mean[index])
            self.var[index]+=0.01
            #for i in range(len(self.var[index])):
            #    if self.var[index][i]<0.01:
            #        self.var[index][i]=0.01
            #print(self.var[index])
            index+=1

    def plot(self):
        print("Imagination of numbers in Bayesian classifier:")
        index=0
        for i in self.mean:
            print(index,":")
            j=0
            for p in range(28):
                for q in range(28):
                    print(int(self.mean[index][j]>=128),end=" ")
                    j+=1
                print(" ")
            index+=1

    def predict(self, X,Y):
        index=0
        error=0
        for c in X:
            print("Postirior (in log scale):")
            cl=0
            post=np.zeros(10)
            for i in self.prior:
                for pixel in range(28*28):
                    #post[cl]+=self.log_gaussian(c[pixel],self.mean[cl][pixel],self.var[cl][pixel])
                    post[cl]+=((-1*(((c[pixel] - self.mean[cl][pixel])**2) / (2*self.var[cl][pixel]) ) ) -0.5*math.log(2*math.pi*self.var[cl][pixel]))
                post[cl]+=math.log(float(i))
                #print(post[cl])
                cl+=1
            post=post/np.sum(post)
            for k in range(10):
                print(k,":",end=" ")
                print(post[k])
            predict = np.argmin(post)
            print("predict:",predict,end=" ")
            print("   ans:",int(Y[index][0]))
            if predict != int(Y[index][0]):
                error += 1
            index+=1
        #print(error)
        print("error rate:",error/len(Y))


def mnist_pre():
    # train_img
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
    with open(train_label_path, 'rb') as f:
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


if __name__ == '__main__':
    mode=int(sys.argv[1])
    tr_x, tr_y, te_x, te_y =mnist_pre()
    if mode==1:
        model=NaiveBayes()
        model.fit(tr_x,tr_y,10)
        model.predict(te_x,te_y)
        model.plot()
    else:
        classCount = np.zeros(10)
        pixelCount = np.ones([10,784,32])
        for idex, data in enumerate(tr_x):
            classCount[int(tr_y[idex][0])] += 1
            for pixel in range(784):
                pixelCount[int(tr_y[idex][0])][pixel][(int(data[pixel])//8)] += 1
        error=0
        posterior= np.zeros([len(te_y), 10])
        for idx, data in enumerate(te_x):
            print("Postirior (in log scale):")
            for num in range(10):
                for pixel in range(784):
                    posterior[idx][num] += math.log(pixelCount[num][pixel][int(data[pixel])//8]/classCount[num])
                posterior[idx][num] += math.log(classCount[num]/len(tr_y))
            posterior[idx] = posterior[idx] / np.sum(posterior[idx])
            for k in range(10):
                print(k,":",end=" ")
                print(posterior[idx][k])
            predict = np.argmin(posterior[idx])
            print("predict:",predict,end=" ")
            print("   ans:",int(te_y[idx][0]))
            if predict != int(te_y[idx][0]):
                error += 1
        for k in range(10):
            print(k,":")
            idx=0
            for p in range(28):
                for q in range(28):
                    result=(np.sum(pixelCount[k][idx][:16])<np.sum(pixelCount[k][idx][16:]))
                    #print(int(np.argmax(pixelCount[k][idx])>=16),end=" ")
                    print(int(result),end=" ")
                    idx+=1
                print(" ")
            print(" ")
        print("error rate:",error/len(te_y))
