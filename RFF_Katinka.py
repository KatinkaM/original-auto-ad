import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


HSI_img = "abu-airport-1"
HSI_mat_file = HSI_img + ".mat"

##Loading HSI and reshape from 3D to 2D
X_raw = sio.loadmat(HSI_mat_file)['data']
X = X_raw.reshape(X_raw.shape[0]*X_raw.shape[1],-1)
length_X = X.shape[0] #Amount of pixels/size of dataset

#Need to define z(X) = cos(w*X + b), X is the data, b is drawn from uniform distribution [0,2pi] 
#and w is drawn from p(w) which is the gaussian distribution

#Retrive b from uniform distribution
b = np.random.uniform(0,2*np.pi,length_X)

#Retrieve w from gaussian distribution
mu, sigma = 0, 0.5 # mean and standard deviation
w = np.random.normal(mu, sigma, length_X)


#Create Z matrix
Z = np.zeros((X.shape[0],X.shape[1]))
for i in range(X.shape[1]):
    Z[:,i] = np.sqrt(1/length_X)*np.cos(w.T*X[:,i] + b)

#Create Kernel matrix k = z.t*z
k = Z.T@Z

#Perform PCA on the Kernel Matrix

u,v = np.linalg.eigh(k) #Eigenvalue decomposition to find eigenvectors and eigenvalues
new_v = v[:,0:100] #Use first 100 dimensions

#Transform the data with the 100 eigenvectors
X_new =np.transpose(new_v.T@X.T)
