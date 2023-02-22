import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


def calculate_AUC(image, abu_image):
    abu = "C:/Users/katin/Documents/NTNU/Semester_10/data/ABU_data/"+abu_image + ".mat"

    det = sio.loadmat(image)['detection']


    #Plotting detection map
    det = det
    

    back = sio.loadmat( image)['background']
    print(back.shape)
    back = back[:,:,0]


    
    
    plt.imshow(back)
    plt.show()

    # plot on the second subplot (top right)
    plt.imshow(det)
    plt.show()

    # plot on the third subplot (bottom left)
   


residual_root_path = "detection_nodim.mat"
abu = "abu-beach-4"
calculate_AUC(residual_root_path,abu)
