#!/usr/bin/env python
# coding: utf-8

# import libraries

from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax


# import data

d1 = r'C:/Users/Paras/Desktop/Pattern Recognition/DS1/training_validation/'
d2 = r'C:/Users/Paras/Desktop/Pattern Recognition/DS1/test/'
d = [d1,d2]
class0 = []
class1 = []
class2 = []
class3 = []
class4 = []
class5 = []
class6 = []
class7 = []
class8 = []
class9 = []
def fetch_file(dn,fn,lis):
    f1 = os.path.join(dn,fn)
    with open(f1) as f:
        lis.append(f.readlines())

for i in d: 
    for filename in os.listdir(i):
        if filename.startswith('class_0'):
            fetch_file(i,filename,class0)
        elif filename.startswith('class_1'):
            fetch_file(i,filename,class1)
        elif filename.startswith('class_2'):
            fetch_file(i,filename,class2)
        elif filename.startswith('class_3'):
            fetch_file(i,filename,class3)
        elif filename.startswith('class_4'):
            fetch_file(i,filename,class4)
        elif filename.startswith('class_5'):
            fetch_file(i,filename,class5)
        elif filename.startswith('class_6'):
            fetch_file(i,filename,class6)
        elif filename.startswith('class_7'):
            fetch_file(i,filename,class7)
        elif filename.startswith('class_8'):
            fetch_file(i,filename,class8)
        elif filename.startswith('class_9'):
            fetch_file(i,filename,class9)
        else:
            print("Class doesn't exist")


# function to preprocess data

def preprocessdata(cl):
    lenc0 = len(cl)
    dim = len(cl[0])
    for  i  in range(lenc0):
        for j in range(dim):
            l0 = cl[i][j]
            cl[i][j] = l0[:-1]

    for i in range(len(cl)):
        cl[i] = list(chain.from_iterable(cl[i]))

    for i in range(len(cl)):
        cl[i] = list(map(int,cl[i]))
    
    return cl


# function to find mean of data

def meanval(cl):
    sumval = list(np.zeros(1024))
    meanval = []
    for i in range(len(cl)):
        for j in range(len(cl[i])):    
            sumval[j] += cl[i][j]

    for i in range(len(sumval)):
        meanval.append(sumval[i]/len(cl))
    return meanval


#preprocessing data and creating dataset

classes = [class0, class1, class2, class3, class4, class5, class6, class7, class8, class9]
for c in classes:
    c = preprocessdata(c)
dataset = []
for c in classes:
    for i in c:
        dataset.append(i)


# calculating mean of dataset

mean_val_dataset = meanval(dataset)


#calculating covariance matrix

cov_mat = np.dot((np.array(dataset) - np.array(mean_val_dataset)).T, (np.array(dataset) - np.array(mean_val_dataset)))
for i in range(len(cov_mat)):
    for j in range(len(cov_mat[i])):
        cov_mat[i][j] = cov_mat[i][j]/(len(dataset)-1)
        

#eigen value and eigen vector of covariance matrix

eig_val,eig_vec = np.linalg.eigh(cov_mat)


#sort eigen vale and eigen vector in descending order

index_maxtomin = np.argsort(eig_val)[::-1]
up_eig_val = eig_val[index_maxtomin] 
up_eig_vec = eig_vec[:,index_maxtomin]


#capturing actual target values in a list

target = []
val = 0
for c in classes:
    for i in c:
        target.append(val)
    val +=1


# plotting with 3 principal components

num_prin_comp = 3
prin_comp = up_eig_vec[:,0:num_prin_comp]

new_data = (np.dot(prin_comp.T, (np.array(dataset) - np.array(mean_val_dataset)).T)).T

pc_data = pd.DataFrame(new_data, columns = ['PC1', 'PC2', 'PC3'])

fig = plt.figure(figsize=(12,12))

axis = fig.add_subplot(111, projection='3d')

axis.scatter(xs = pc_data['PC1'], ys = pc_data['PC2'], zs = pc_data['PC3'], c=target)

axis.set_xlabel("PC1")

axis.set_ylabel("PC2")

axis.set_zlabel("PC3")


# Performing Logistic Regression
# function to find graient

oh_enc = OneHotEncoder(sparse=False)
def grad(data, lab, Weig):
    
    out = np.dot((-1 * data),Weig)
    
    #applying softmax
    
    max_row = np.max(out,axis=1,keepdims=True) 
    out1 = np.exp(out - max_row) 
    sum_row = np.sum(out1,axis=1,keepdims=True) 
    f_x = out1 / sum_row 
    P_val = f_x
    
    size = data.shape[0]
    grad = 1/size * (np.dot(data.T,(lab - P_val))) 

    return grad


# function performing gradient descent

def grad_desc(data, lab, iterations=1000, lr=0.1):
    
    c = 0
    Y_oh = oh_enc.fit_transform(lab)
    Weight = np.zeros((data.shape[1], Y_oh.shape[1]))
 
    while c < iterations:
        c += 1
        Weight -= lr * grad(data, Y_oh, Weight)
    return Weight


#class which perform logistic regression 

class LogReg:
    def train(self, data, lab):
        self.Weight = grad_desc(data, lab)

    def analyse(self, data):
        output = np.dot((-1 * data),self.Weight)
        P_value = softmax(output, axis=1)
        return np.argmax(P_value, axis=1)


# function which return accuracy over the data passed performing Logistic regression over data and return accuracy

def PCA_MultiNomialLogisticRegression(comp, dataset, mean_val_dataset, target):
    
    mod = LogReg()
    new_data1 = (np.dot(comp.T, (np.array(dataset) - np.array(mean_val_dataset)).T)).T
    pc_data1 = pd.DataFrame(new_data1)
    pc_target = pd.DataFrame(target)
    mod.train(pc_data1, pc_target)
    val = mod.analyse(pc_data1)
    
    mis_classified = 0
    for i in range(len(val)):
        if target[i] != val[i]:
            mis_classified +=1

    accuracy = 1 - mis_classified/len(target) 
    return accuracy


# accuracy with data in original data space

acc = PCA_MultiNomialLogisticRegression(up_eig_vec,dataset, mean_val_dataset, target)
print("Accuracy with data in original data space : ",acc)


#calculating 95% of variance

tot_var = 0
for i in up_eig_val:
    tot_var +=i

var_val = []
for i in up_eig_val:
    var_val.append((i/tot_var)*100)

count = 0
sum1 = 0
for i in var_val:
    if sum1 < 95:
        sum1 += i
        count +=1

# data with 95% variance of original data space

num_prin_comp1 = count
prin_comp1 = up_eig_vec[:,0:num_prin_comp1]


# accuracy with data in 95% variance of original data space

acc = PCA_MultiNomialLogisticRegression(prin_comp1,dataset, mean_val_dataset, target)
print("Accuracy with data in reduced data space : ",acc)

