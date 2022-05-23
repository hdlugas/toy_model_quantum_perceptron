import numpy as np
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dist(a,b):
    #computes the euclidean distance between complex numbers a and b
    return np.sqrt(pow(a[0]-b[0],2)+pow(a[1]-b[1],2))


def mag(v):
    #get the L^2 norm of the 1-d array v
    temp=0
    for i in range(0,len(v)):
        temp += pow(v[i],2)
    return np.sqrt(temp)


def line(pt2,ax):
    #draws a line from the origin to 'point'
    pt1 = [0,0,0]
    xvals = [pt1[0],pt2[0]]
    yvals = [pt1[1],pt2[1]]
    zvals = [pt1[2],pt2[2]]
    return ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='purple')


def rotate_theta(w,theta):
    #rotates the vector w by the polar angle theta
    A = [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]
    return np.matmul(A,w)


def rotate_phi(w,phi):
    #rotates the vector w by the azimuthal angle phi
    t=1-np.cos(phi)
    c=np.cos(phi)
    s=np.sin(phi)
    theta=-np.arccos(w[0])
    ux=-np.sin(theta)
    uy=np.cos(theta)
    a11=t*ux*ux+c
    a12=t*ux*uy
    a13=s*uy
    a21=t*ux*uy
    a22=t*uy*uy+c
    a23=-s*ux
    a31=-s*uy
    a32=s*ux
    a33=c

    A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    return np.matmul(A,w)


def get_cost2(weight,cluster):
    #evaluate the cost function at the given weight
    cost1=0
    cost2=0
    for i in range(0,len(cluster[:,0])):
        cost1 = cost1 + pow(np.dot(weight,cluster[i,:]),2)
        cost2 = cost2 + np.dot(weight,cluster[i,:])
    return cost2




def rotate_x(w,theta):
    #rotates the vector w in R^3 by theta wrt the x-axis
    mat = [[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-1*np.sin(theta),np.cos(theta)]]
    return np.matmul(mat,w)


def rotate_y(w,theta):
    #rotates the vector w in R^3 by theta wrt the y-axis
    mat = [[np.cos(theta),0,-1*np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]]
    return np.matmul(mat,w)

def rotate_z(w,theta):
    #rotates the vector w in R^3 by theta wrt the z-axis
    mat = [[np.cos(theta), np.sin(theta),0],[-1*np.sin(theta),np.cos(theta),0],[0,0,1]]
    return np.matmul(mat,w)






#theta is the polar angle, phi the azimuthal
n_theta = 50
n_phi = 25
theta_vec = np.linspace(0,2*np.pi,n_theta)
phi_vec = np.linspace(0,np.pi,n_phi)

x = np.zeros((n_theta*n_phi))
y = np.zeros((n_theta*n_phi))
z = np.zeros((n_theta*n_phi))

for i in range(0,len(theta_vec)):
    k=0
    for j in range(0,len(phi_vec)):
        x[i*len(phi_vec)+j] = np.cos(theta_vec[i]) * np.sin(phi_vec[j])
        y[i*len(phi_vec)+j] = np.sin(theta_vec[i]) * np.sin(phi_vec[j])
        z[i*len(phi_vec)+j] = np.cos(phi_vec[j])
        
embedding = np.zeros((n_theta*n_phi,3))
embedding[:,0]=x
embedding[:,1]=y
embedding[:,2]=z






#define the two clusters comprising the input data
m1=5 #number of data in first cluster
m2=5 #number of data in second cluster

cluster_1=np.zeros((m1,3))
cluster_2=np.zeros((m2,3))


mean_x_1=rd.randint(0,n_theta*n_phi)
sd_x_1=rd.uniform(0,2)
mean_y_1=rd.randint(0,n_theta*n_phi)
sd_y_1=rd.uniform(0,2)
mean_z_1=rd.randint(0,n_theta*n_phi)
sd_z_1=rd.uniform(0,2)
idx_x_vec_1=np.zeros(m1)
idx_y_vec_1=np.zeros(m1)
idx_z_vec_1=np.zeros(m1)

mean_x_2=rd.randint(0,n_theta*n_phi)
sd_x_2=rd.uniform(0,2)
mean_y_2=rd.randint(0,n_theta*n_phi)
sd_y_2=rd.uniform(0,2)
mean_z_2=rd.randint(0,n_theta*n_phi)
sd_z_2=rd.uniform(0,2)
idx_x_vec_2=np.zeros(m2)
idx_y_vec_2=np.zeros(m2)
idx_z_vec_2=np.zeros(m2)


for i in range(0,m1):
    temp_x=int(np.random.normal(mean_x_1,sd_x_1,size=1))
    temp_y=int(np.random.normal(mean_y_1,sd_y_1,size=1))
    temp_z=int(np.random.normal(mean_z_1,sd_z_1,size=1))
    idx_x_vec_1[i] = temp_x
    idx_y_vec_1[i] = temp_y
    idx_z_vec_1[i] = temp_z

for i in range(0,m2):
    temp_x=int(np.random.normal(mean_x_2,sd_x_2,size=1))
    temp_y=int(np.random.normal(mean_y_2,sd_y_2,size=1))
    temp_z=int(np.random.normal(mean_z_2,sd_z_2,size=1))
    idx_x_vec_2[i] = temp_x
    idx_y_vec_2[i] = temp_y
    idx_z_vec_2[i] = temp_z


#define the two clusters of input data
for i in range(0,m1):
    cluster_1[i,0] = embedding[int(idx_x_vec_1[i]),0]
    cluster_1[i,1] = embedding[int(idx_y_vec_1[i]),1]
    cluster_1[i,2] = embedding[int(idx_z_vec_1[i]),2] 

for i in range(0,m2):
    cluster_2[i,0] = embedding[int(idx_x_vec_2[i]),0]
    cluster_2[i,1] = embedding[int(idx_y_vec_2[i]),1]
    cluster_2[i,2] = embedding[int(idx_z_vec_2[i]),2] 


#compute coordinates of the 'average' point in each data cluster
mag_1=np.sqrt(pow(np.mean(cluster_1[:,0]),2)+pow(np.mean(cluster_1[:,1]),2)+pow(np.mean(cluster_1[:,2]),2))
cluster_1_avg=np.asarray([np.mean(cluster_1[:,0])/mag_1,np.mean(cluster_1[:,1])/mag_1,np.mean(cluster_1[:,2])/mag_1])
target=cluster_1_avg

mag_2=np.sqrt(pow(np.mean(cluster_2[:,0]),2)+pow(np.mean(cluster_2[:,1]),2)+pow(np.mean(cluster_2[:,2]),2))
cluster_2_avg=np.asarray([np.mean(cluster_2[:,0])/mag_2,np.mean(cluster_2[:,1])/mag_2,np.mean(cluster_2[:,2])/mag_2])


#initialize the five seed weight vectors
h=0.1 #seed step size
seed_theta_1 = np.pi/4
seed_phi_1 = np.pi/4
seed_weight_1 = np.asarray([np.sin(seed_phi_1)*np.cos(seed_theta_1),np.sin(seed_phi_1)*np.sin(seed_theta_1),np.cos(seed_phi_1)])

seed_theta_2 = np.pi/4+h
seed_phi_2 = np.pi/4
seed_weight_2 = np.asarray([np.sin(seed_phi_2)*np.cos(seed_theta_2),np.sin(seed_phi_2)*np.sin(seed_theta_2),np.cos(seed_phi_2)])

seed_theta_3 = np.pi/4-h
seed_phi_3 = np.pi/4
seed_weight_3 = np.asarray([np.sin(seed_phi_3)*np.cos(seed_theta_3),np.sin(seed_phi_3)*np.sin(seed_theta_3),np.cos(seed_phi_3)])

seed_theta_4 = np.pi/4
seed_phi_4 = np.pi/4+h
seed_weight_4 = np.asarray([np.sin(seed_phi_4)*np.cos(seed_theta_4),np.sin(seed_phi_4)*np.sin(seed_theta_4),np.cos(seed_phi_4)])

seed_theta_5 = np.pi/4
seed_phi_5 = np.pi/4-h
seed_weight_5 = np.asarray([np.sin(seed_phi_5)*np.cos(seed_theta_5),np.sin(seed_phi_5)*np.sin(seed_theta_5),np.cos(seed_phi_5)])



#create plot
fig = plt.figure()
ax = Axes3D(fig)

#plot the embedded points in R^3
plot = ax.scatter(embedding[:,0],embedding[:,1],embedding[:,2], color='black')


#plot cluster 1 and cluster 2
pt1 = [0,0,0]
for i in range(0,m1):
    pt2 = cluster_1[i,:]/mag(cluster_1[i,:])
    xvals = [pt1[0],pt2[0]]
    yvals = [pt1[1],pt2[1]]
    zvals = [pt1[2],pt2[2]]
    plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='orange')

pt2 = cluster_1[0,:]/mag(cluster_1[0,:])
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='orange',label='cluster 1')


for i in range(0,m2):
    pt2 = cluster_2[i,:]/mag(cluster_2[i,:])
    xvals = [pt1[0],pt2[0]]
    yvals = [pt1[1],pt2[1]]
    zvals = [pt1[2],pt2[2]]
    plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='green')

pt2 = cluster_2[0,:]/mag(cluster_2[0,:])
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='green',label='cluster 2')


#plot average point of cluster 1
pt2 = cluster_1_avg
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='red',label='cluster 1 average')

#plot average point of cluster 2
pt2 = cluster_2_avg
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='blue',label='cluster 2 average')



#plot initial seed weights
pt2 = seed_weight_2
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='purple',label='seed weights')

plot = line(seed_weight_3,ax)
plot = line(seed_weight_4,ax)
plot = line(seed_weight_5,ax)




#check to see if any seed weight vectors are near cluster 1
error = 0.08

#now adjust weight vector
    
#get initial cost function values corresponding to each seed weight
cost_vec=np.zeros((4))
    
weight = seed_weight_1
count=0
number_iterations = 1000

while count <= number_iterations:
    #get random input vector 'target'
    rand=rd.randint(0,m1+m2)
    if rand < m1:
        target = cluster_1[rand,:]
    if rand >= m1:
        target = cluster_2[m1-rand,:]
        
    if count==0:
        cost_vec[0] = pow(np.dot(seed_weight_2,target),2)
        cost_vec[1] = pow(np.dot(seed_weight_3,target),2)
        cost_vec[2] = pow(np.dot(seed_weight_4,target),2)
        cost_vec[3] = pow(np.dot(seed_weight_5,target),2)

    else:
        cost_vec[0] = pow(np.dot(weight_1,target),2)
        cost_vec[1] = pow(np.dot(weight_2,target),2)
        cost_vec[2] = pow(np.dot(weight_3,target),2)
        cost_vec[3] = pow(np.dot(weight_4,target),2)

    for i in range(0,len(cost_vec)):
        weight_1=rotate_theta(weight,h)
        weight_2=rotate_theta(weight,-h)
        weight_3=rotate_phi(weight,h)
        weight_4=rotate_phi(weight,-h)
        if (rand < m1 #if the sampled input is in cluster 1
        and cost_vec[i] == np.amax(cost_vec) #get direction of the dot product with target vector squared
        and np.abs(cost_vec[i]) < 0.5): #check to see if we have a false positive
            print('misclassified input at count',count)
            if i==0:
                weight=rotate_theta(weight,h)
                plot = line(weight,ax)
            if i==1:
                weight=rotate_theta(weight,-h)
                plot = line(weight,ax)
            if i==2:
                weight=rotate_phi(weight,h)
                plot = line(weight,ax)
            if i==3:
                weight=rotate_phi(weight,-h)
                plot = line(weight,ax)
            #if count == number_iterations:
                #plot = line(weight,ax)

        if cost_vec[i] == np.amax(cost_vec) and rand >= m1 and np.abs(cost_vec[i]) > 0.5:
            print('misclassified input at count',count)
            if i==0:
                weight=rotate_theta(weight,-h)
                plot = line(weight,ax)
            if i==1:
                weight=rotate_theta(weight,h)
                plot = line(weight,ax)
            if i==2:
                weight=rotate_phi(weight,-h)
                plot = line(weight,ax)
            if i==3:
                weight=rotate_phi(weight,h)
                plot = line(weight,ax)
            #if count == number_iterations:
            #    plot = line(weight,ax)
        

    if count==number_iterations-1:
        print('count has reached ',number_iterations)
    count += 1



'''
#the block below can be used to verify rotation functions do as they're supposed to
pt2 = seed_weight_1
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='red',label='initial')


pt2=rotate_theta(seed_weight_1,np.pi/4)
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='red',label='final')
'''


pt2 = seed_weight_1
xvals = [pt1[0],pt2[0]]
yvals = [pt1[1],pt2[1]]
zvals = [pt1[2],pt2[2]]
plot = ax.plot(xvals,yvals,zvals,'bo',linestyle="-",color='red',label='initial seed')

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_zlabel('z-axis')

plt.legend()
plt.show()







