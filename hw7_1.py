import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
#import cv2

def load_faces(split="Training", root="./Yale_Face_Database"):
    num_file = len(files) # 135
    print("total image #:{}".format(num_file))
    label = np.zeros(num_file, dtype=int)
    faces = np.zeros((num_file, 22194))
    for idx, f in enumerate(files):
        path = os.path.join(root, f)
        im = Image.open(path) # 162*137=22194
        #im=im.convert('L')
        im=im.resize((137,162))
        im = np.asarray(im)
        row, col = im.shape
        im = im.reshape(1,-1)
        faces[idx,:] = im
        label[idx] = int(files[idx][7:9])#total 15 kinds of labels
    return faces, num_file,files, label # 231, 195
w=162
h=137
def eigenface(vec):#show first 25 eigenfaces
    for i in range(5):
        for j in range(5):
            plt.subplot(5,5,i*5+j+1)
            vec_=vec[:,i*5+j].reshape(w,h)#reshape vector to image size
            plt.imshow(vec_,cmap='gray')
    plt.show()
    plt.close()

def PCA_reconstruct(faces,vec):#show reconstruction
    k = np.random.randint(0,135,10)#choose random 10 faces
    new=np.matmul(faces[k,:],vec)
    r=np.matmul(new,vec.T)#reconstruct by multiply original picture with eigen vector and its transpose consecutively
    for i in range(2):
        for j in range(5):
            plt.subplot(2,5,i*5+j+1)
            vec_=r[i*5+j,:].reshape(w,h)#reshape to image size
            plt.imshow(vec_,cmap='gray')
    plt.show()

def decompose_faces(eigenfaces, faces, num):
    coeff = np.zeros((num, 25))
    for i in range(num):
        for eig in range(25):
            coeff[i,eig] = ( faces[i,:] ).dot( eigenfaces[:,eig].reshape(-1,1) )
    return coeff

def classify(test_coeff, test_filename, test_num, test_label, train_coeff, train_filename, train_num, train_label):
    k = 5# k nearest
    predict = np.zeros(test_num, dtype=int)
    acc= 0
    dist = np.zeros(train_num)
    for i in range(test_num):
        for j in range(train_num):
            dist[j] = np.linalg.norm(test_[i]-train_[j])# calculate distance between test case with all train case
        min_dist = np.argsort(dist)[:k]# find k nearest
        k_predict = train_label[min_dist]#obtain k training labels
        predict[i] = np.argmax(np.bincount(k_predict))# find the most common one
        if test_label[i]==predict[i]:
            acc += 1
    print("accuracy rate:",acc/test_num)

if __name__ == "__main__":
    faces, num_file,train_filename, train_label = load_faces() #(45045,num_file=135)
    test_faces, test_num_file, test_filename, test_label= load_faces(split="Testing")
    mean_face = (np.sum(faces,axis=0)/num_file).reshape(1,-1)#calculate average value of all face
    diff_faces=np.zeros(faces.shape)#
    for i in range (135):
      diff_faces[i,:]=faces[i,:]-mean_face#calculate the diffence between each face and mean face
    cov=diff_faces.T.dot(diff_faces)#calculate covariance matrix
    eigen_val,eigen_vec = np.linalg.eigh(cov)#obtain eigen value and vector from covariance matrix
    #np.save("value.npy",eigen_value)
    #np.save("vector.npy",eigen_vector)
    eigen_val=np.load("value.npy")
    eigen_vect=np.load("vector.npy")
    print("here")
    idx = eigen_value.argsort()[::-1]
    vec=eigen_vector[:,idx][:,:25]
    """eigenface(vec)
    PCA_res(faces,vec)
    train_coeff = decompose_faces(vec, faces, num_file)
    test_coeff = decompose_faces(vec, test_faces, test_num_file)
    classify(test_coeff, test_filename, test_num_file, test_label, train_coeff, train_filename, num_file, train_label)
    """
