import numpy as np
import random as rd
import matplotlib.pyplot as plt


def dist(a,b):
    #computes the euclidean distance between complex numbers a and b
    return np.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))


#we can interpret n as the number of bits in each sequence of bits
n = 6


#the numpy array 'data' below is commented because we are using the embedding of equally spaced
#input data (which consists of all sequences of length n comprised of 0's and 1's)
#on the quarter unit circle. This embedding can be given by considering each sequence
#in the input data as a number written in binary and then mapping the n-th largest binary 
#number to the point on the quarter unit circle corresponding to the n-th rotation.


#If we wish to choose a different embedding, expand upon the the numpy array 'data' below

#get matrix whose columns are all sequences of length n consisting of 0's and 1's
#data = np.asarray(list(map(list, itertools.product([0, 1], repeat=n))))




#get array of points equally spaced on quarter unit circle
embedding_vec = np.zeros((pow(2,n),2))

temp=np.linspace(0,1,pow(2,n))
for i in range(0,pow(2,n)):
    x=np.cos(np.pi/2*temp[i])
    y=np.sin(np.pi/2*temp[i])
    embedding_vec[i,0]=x
    embedding_vec[i,1]=y

#step=1/pow(2,n)
#rotation_matrix=np.asarray([[np.cos(step),np.sin(step)],[-1*np.sin(step),np.cos(step)]])

#for i in range(0,int(pow(2,n)/2-1)):
#    embedding_vec[i+1,:]=np.matmul(rotation_matrix,embedding_vec[i,:])


#define the two clusters comprising the input data
m1=5 #number of data in first cluster
m2=5 #number of data in second cluster

cluster_1=np.zeros((m1,2))
cluster_2=np.zeros((m2,2))




#get two vectors (one for each cluster of input data) whose entries 
#are indices of embedding_vec such that embedding_vec evaluated at those
#indices are the entries of the vectors comprising the input data on 
#the quarter unit circle
mean1=rd.randint(0,pow(2,n))
sd1=rd.uniform(0,4)
idx1_vec=np.zeros(m1)
for i in range(0,m1):
    temp=int(np.random.normal(mean1,sd1,size=1))
    if temp >= 0 and temp < pow(2,n):
        idx1_vec[i]=temp
    if temp < 0:
        idx1_vec[i]=0
    if temp >= pow(2,n):
        idx1_vec[i]=pow(2,n)-1


mean2=rd.randint(0,pow(2,n))
sd2=rd.uniform(0,4)
idx2_vec=np.zeros(m2)
for i in range(0,m2):
    temp=int(np.random.normal(mean2,sd2,size=1))
    if temp >= 0 and temp < pow(2,n):
        idx2_vec[i]=temp
    if temp < 0:
        idx2_vec[i]=0
    if temp >= pow(2,n):
        idx2_vec[i]=pow(2,n)-1




#define the two clusters of input data
for i in range(0,m1):
    cluster_1[i,0]=embedding_vec[int(idx1_vec[i]),0]
    cluster_1[i,1]=embedding_vec[int(idx1_vec[i]),1]

for i in range(0,m2):
    cluster_2[i,0]=embedding_vec[int(idx2_vec[i]),0]
    cluster_2[i,1]=embedding_vec[int(idx2_vec[i]),1]




#compute coordinates of the 'average' point in each data cluster
mag_1=np.sqrt(pow(np.mean(cluster_1[:,0]),2)+pow(np.mean(cluster_1[:,1]),2))
cluster_1_avg=np.asarray([np.mean(cluster_1[:,0])/mag_1,np.mean(cluster_1[:,1])/mag_1])

mag_2=np.sqrt(pow(np.mean(cluster_2[:,0]),2)+pow(np.mean(cluster_2[:,1]),2))
cluster_2_avg=np.asarray([np.mean(cluster_2[:,0])/mag_2,np.mean(cluster_2[:,1])/mag_2])




#initialize the weight vectors
h=1/pow(2,n)

theta=np.pi/4
theta1=np.pi/4-h
theta2=np.pi/4+h
weight=np.asarray([np.cos(theta),np.sin(theta)])
weight1=np.asarray([np.cos(theta1),np.sin(theta1)])
weight2=np.asarray([np.cos(theta2),np.sin(theta2)])



#make plots
fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(embedding_vec[:,0],embedding_vec[:,1],color='black')
plt.scatter(cluster_1[:,0],cluster_1[:,1],color='orange',label='cluster 1')
plt.scatter(cluster_2[:,0],cluster_2[:,1],color='green',label='cluster 2')

plt.scatter(cluster_1_avg[0],cluster_1_avg[1],color='red',label='cluster 1 average')
plt.scatter(cluster_2_avg[0],cluster_2_avg[1],color='blue',label='cluster 2 average')

plt.scatter(weight[0],weight[1],color='purple',label='initial weight')
plt.scatter(weight1[0],weight1[1],color='pink',label='seed weight 1')
plt.scatter(weight2[0],weight2[1],color='yellow',label='seed weight 2')





count=0
epsilon = 0.01
while count<1001:
    idx=rd.randint(0,m1+m2-1)
    if idx < m1:#case when the sampled input is in cluster 1
        target = cluster_1[idx,:]
        dot1 = pow(np.dot(cluster_1[idx,:],weight1),2)
        dot2 = pow(np.dot(cluster_1[idx,:],weight2),2)
        if dot1 > dot2 and dot1 < 1-epsilon:
            phi = -h
            rotation_matrix=[[np.cos(phi),-1*np.sin(phi)],[np.sin(phi),np.cos(phi)]]
            weight = np.asarray(np.matmul(rotation_matrix,weight))
            weight1 = np.asarray(np.matmul(rotation_matrix,weight1))
            weight2 = np.asarray(np.matmul(rotation_matrix,weight2))
            plt.scatter(weight[0],weight[1],color='purple')
            ax.text(weight[0],weight[1]+0.01,str(count),horizontalalignment='center',verticalalignment='center')
            print('misclassified input at count ',count)

        if dot1 < dot2 and dot2 < 1-epsilon:
            phi = h
            rotation_matrix=[[np.cos(phi),-1*np.sin(phi)],[np.sin(phi),np.cos(phi)]]
            weight = np.asarray(np.matmul(rotation_matrix,weight))
            weight1 = np.asarray(np.matmul(rotation_matrix,weight1))
            weight2 = np.asarray(np.matmul(rotation_matrix,weight2))
            plt.scatter(weight[0],weight[1],color='purple')
            ax.text(weight[0],weight[1]+0.01,str(count),horizontalalignment='center',verticalalignment='center')
            print('misclassified input at count ',count)
                    


    if idx >= m1:#case when the sampled input is in cluster 2
        target = cluster_2[idx-m1,:]
        dot1 = pow(np.dot(cluster_2[idx-m1,:],weight1),2)
        dot2 = pow(np.dot(cluster_2[idx-m1,:],weight2),2)
        if dot1 > dot2 and dot1 > 1-epsilon:
            phi = h
            rotation_matrix=[[np.cos(phi),-1*np.sin(phi)],[np.sin(phi),np.cos(phi)]]
            weight = np.asarray(np.matmul(rotation_matrix,weight))
            weight1 = np.asarray(np.matmul(rotation_matrix,weight1))
            weight2 = np.asarray(np.matmul(rotation_matrix,weight2))
            plt.scatter(weight[0],weight[1],color='purple')
            ax.text(weight[0],weight[1]+0.01,str(count),horizontalalignment='center',verticalalignment='center')
            print('misclassified input at count ',count)
        
        if dot1 < dot2 and dot2 > 1-epsilon:
            phi = -h
            rotation_matrix=[[np.cos(phi),-1*np.sin(phi)],[np.sin(phi),np.cos(phi)]]
            weight = np.asarray(np.matmul(rotation_matrix,weight))
            weight1 = np.asarray(np.matmul(rotation_matrix,weight1))
            weight2 = np.asarray(np.matmul(rotation_matrix,weight2))
            plt.scatter(weight[0],weight[1],color='purple')
            ax.text(weight[0],weight[1]+0.01,str(count),horizontalalignment='center',verticalalignment='center')
            print('misclassified input at count ',count)
        
    count += 1 
    
    if count==1000:
        print('the count has reached =',count)
        flag=1

#end while loop



plt.legend()
plt.show()
       



