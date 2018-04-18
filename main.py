import numpy as np
import random as rm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn import mixture
import time

##Generating data from a Gaussian Mixture Model

n_samples = 300
np.random.seed(rm.randint(0,10000))

## Spherical Data

# generate spherical data for class1
C1 = np.array([[1, 0], [0, 1]])  ## This matrix can be changed to generated different GMMs
shifted_gaussian = np.dot(np.random.randn(n_samples, 2), C1)+ np.array([1,1])

# generate spherical data for class2
C2 = np.array([[2, 0], [0, 2]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C2)+ np.array([5, 5])

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train) 

Data,label=clf.sample((300))
X=Data
Y=label
class1_x=X[Y==0,0]
class1_y=X[Y==0,1]
class2_x=X[Y==1,0]
class2_y=X[Y==1,1]
plt.figure(1)
plt.scatter(class1_x,class1_y,c='blue',s=50)
plt.scatter(class2_x,class2_y,c='green',marker='*',s=50)
plt.legend(['Cluster1','Cluster2'],loc='upper left')
plt.title('Data Generated with Spherical Covariance Matrix')
plt.show()

alpha=np.zeros(2)
class1_resp=np.zeros((len(Data),1))
class2_resp=np.zeros((len(Data),1))

for i in range(0,len(Data)):
    a=rm.uniform(0,1)
    class1_resp[i]=a
    class2_resp[i]=1-a

alpha0=np.sum(class1_resp)/len(Data)
alpha1=np.sum(class2_resp)/len(Data)
  

for j in range(0,300):
    
    ##Maximization Step
    f1=np.empty((len(Data),2))
    f2=np.empty((len(Data),2))
    for i in range(len(Data)):
        f1[i,:]=(class1_resp[i]*Data[i,:])/np.sum(class1_resp);
        f2[i,:]=(class2_resp[i]*Data[i,:])/np.sum(class2_resp);
    start_mean1=np.sum(f1,axis=0);
    start_mean2=np.sum(f2,axis=0);
    std1=np.zeros((2,2))
    sum_n=np.zeros((2,2))
    sum_d=0
    for i in range(len(Data)):
        sum_d = sum_d + class1_resp[i];
        sum_n = sum_n + class1_resp[i]*(np.transpose((Data[i,:]-start_mean1.reshape(2))))*(Data[i,:].reshape(2,1)-start_mean1)
    std1=sum_n/sum_d;
    std2=np.zeros((2,2))
    sum_n=np.zeros((2,2))
    sum_d=0
    for i in range(len(Data)):
        sum_d = sum_d + class2_resp[i];
        sum_n = sum_n + class2_resp[i]*(np.transpose((Data[i,:]-start_mean2.reshape(2))))*(Data[i,:].reshape(2,1)-start_mean2)
    std2=sum_n/sum_d;
    
    ##Expectation Step
    y1=alpha0*mvn(mean=start_mean1.reshape(2),cov=std1).pdf(Data)
    y2=alpha1*mvn(mean=start_mean2.reshape(2),cov=std2).pdf(Data)
    total=y1+y2
    
    for i in range(len(Data)):
        class1_resp[i]=y1[i]/total[i]
        class2_resp[i]=y2[i]/total[i]
    alpha0=np.sum(class1_resp)/len(Data)
    alpha1=np.sum(class2_resp)/len(Data)
    

label_EM=np.zeros((len(Data),1))

for i in range(0,len(Data)):
    if(class1_resp[i]>0.5):
        label_EM[i]=0
    else:
        label_EM[i]=1

plt.figure(2)
plt.scatter(Data[:,0],Data[:,1],c=label_EM.reshape(len(Data)))
plt.show()
