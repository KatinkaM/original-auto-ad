import scipy.io as sio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Functions.common_utils import auc_and_roc
import matplotlib.pyplot as plt



def calculate_AUC(image, abu):
    abu = "C:/Users/katin/Documents/NTNU/Semester_10/data/ABU_data/"+abu + ".mat"

    det = sio.loadmat(image)['detection']


    #Plotting detection map
    det = np.transpose(det)
    #plt.plot(det)
    # plt.imshow(det)
    # plt.show()

    #Calculatin AUC score 
    img_reshape = det.reshape(det.shape[0]*det.shape[1],-1)
    img_reshape = MinMaxScaler(feature_range = (0,1)).fit_transform(img_reshape)
    gt = sio.loadmat(abu)['map'][0:100,0:100]
    gt = gt.reshape(gt.shape[0]*gt.shape[1])
    AUC,fpr,tpr, threshold =auc_and_roc(gt,img_reshape)

    print("AUC score: " + str(AUC))


residual_root_path = "./detection_orignal1"
abu = "abu-airport-1"
calculate_AUC(residual_root_path, abu)