import tensorflow as tf
import numpy as np
np.random.seed(1337)  # for reproducibility

import os
# import sys
# sys.path.append("../models")
# sys.path.append("../base")
filename = os.path.basename(__file__)

from dbn import DBN
# from cnn import CNN
from base_func import run_sess
from tensorflow.examples.tutorials.mnist import input_data

fr = open('data.txt', 'r')
content = fr.readlines()
datas = []
for x in content[1:]:
    k=content[1:].index(x)
    x = x.strip().split(' ')
    datas.append([float(i) for i in x[:-2]])
    if x[-1]=='危险': datas[k].append(0)
    elif x[-1]=='可疑': datas[k].append(1)
    else: datas[k].append(2)
datas=np.array(datas).astype('float32')

def norm(data):
    m,n=data.shape
    maxMat=np.max(data,0)
    minMat=np.min(data,0)
    diffMat=maxMat-minMat
    for j in range(n-1):
        data[:,j]=(data[:,j]-minMat[j])/diffMat[j]
    return data

def one_hot(data):
    m = data.shape[0]
    ret_dat=np.zeros((m,3))
    for i in range(m):
        if data[i]==0.:
            ret_dat[i]=np.array([1,0,0])
        elif data[i]==1:
            ret_dat[i]=np.array([0,1,0])
        else:
            ret_dat[i]=np.array([0,0,1])
    return ret_dat

traind= norm(datas[:700,:-1].astype('float32'))
trainl= one_hot(datas[:700,-1].astype('float64'))
# print(traind[0]);print(trainl[0])

testd = norm(datas[700:,:-1].astype('float32'))
testl = one_hot(datas[700:,-1].astype('float64'))

dataset=[traind,trainl,testd,testl]

classifier = DBN(
                 hidden_act_func='sigmoid',
                 output_act_func='softmax',
                 loss_func='cross_entropy', # gauss 激活函数会自动转换为 mse 损失函数
                 struct=[26, 11, 6, 11, 11,3],
                 lr=1e-3,
                 momentum=0.8,
                 use_for='classification',
                 bp_algorithm='rmsp',
                 epochs=100,
                 batch_size=20,
                 dropout=0.12,
                 units_type=['gauss','bin'],
                 rbm_lr=1e-3,
                 rbm_epochs=0,
                 cd_k=1,
                 pre_train=True)

run_sess(classifier,dataset,filename,load_saver='')
label_distribution = classifier.label_distribution