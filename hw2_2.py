import numpy as np
from math import factorial
import sys

def read_file(filename):
    data = []
    fp=open(filename,"r")
    line=fp.readline()
    while line:
        line = line.strip()
        data.append(np.array(line))
        line=fp.readline()
    return data

def likelihood(p, m, N):
    return (factorial(N)/(factorial(m)*factorial(N-m))) * (p**m) * ((1-p)**(N-m))

if __name__ == "__main__":
    data = read_file(sys.argv[1])
    count0 = np.char.count(data, '0')
    count1 = np.char.count(data, '1')
    pre_a = int(sys.argv[2])
    pre_b = int(sys.argv[3])
    for idex, (i, j) in enumerate(zip(count0, count1)):
        l = likelihood(i/(i+j) , i, i+j)
        print("case {}: {}".format(idex+1, data[idex]))
        print("Likelihood: ",end=" ")
        print(l)
        print("Beta prior:a={} b={}".format(pre_a, pre_b))
        pre_a += j
        pre_b += i
        print("Beta posterior:a={} b={}".format(pre_a, pre_b))
