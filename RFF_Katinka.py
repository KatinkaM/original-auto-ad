import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


abu_list = ["abu-airport-1","abu-airport-2","abu-airport-3","abu-airport-4","abu-beach-1","abu-beach-2", "abu-beach-3", "abu-beach-4", "abu-urban-1", "abu-urban-2", "abu-urban-3", "abu-urban-4", "abu-urban-5"]
data_path = "C:/Users/katin/Documents/NTNU/Semester_10/data/ABU_data/"


for i in range(len(abu_list)):
    HSI_img = abu_list[i]
    residual_path = "RFF_data/laplace" + "/" + HSI_img + ".mat"
    HSI_mat_file = data_path + HSI_img + ".mat"

    ##Loading HSI and reshape from 3D to 2D
    X_raw = sio.loadmat(HSI_mat_file)['data']
    X = X_raw.reshape(X_raw.shape[0]*X_raw.shape[1],-1)
    length_X = X.shape[0] #Amount of pixels/size of dataset

    #Need to define z(X) = cos(w*X + b), X is the data, b is drawn from uniform distribution [0,2pi] 
    #and w is drawn from p(w) which is the gaussian distribution

    #Retrive b from uniform distribution
    b = np.random.uniform(0,2*np.pi,length_X)

    #Retrieve w from gaussian distribution
    # mu, sigma = 0, 0.5 # mean and standard deviation
    # w = np.random.normal(mu, sigma, length_X)

    #retrieve laplace distribution
    loc, scale = 0., 2.
    w = np.random.laplace(loc, scale, length_X)

    #Create Z matrix
    Z = np.zeros((X.shape[0],X.shape[1]))
    if(Z.shape[1]%2==0):  
        first_half = round(Z.shape[1])
        second_half = round(Z.shape[1])
    else:
        first_half = round(Z.shape[1]) + 1
        second_half = round(Z.shape[1])
    for i in range(X.shape[1]):
        if(i <= first_half):
            Z[:,i] = np.sqrt(1/length_X)*np.cos(w.T*X[:,i] + b)
        else:
            Z[:,i] = np.sqrt(1/length_X)*np.sin(w.T*X[:,i] + b)

    #Create Kernel matrix k = z.t*z
    k = Z.T@Z

    #Perform PCA on the Kernel Matrix

    u,v = np.linalg.eigh(k) #Eigenvalue decomposition to find eigenvectors and eigenvalues
    new_v = v[:,0:100] #Use first 100 dimensions

    #Transform the data with the 100 eigenvectors
    X_new =np.transpose(new_v.T@X.T)


    sio.savemat(residual_path, {'abu': X_new})
