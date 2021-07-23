###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Testing code for DARN proposed in the paper titled "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network"

# Citation
# Y. Zheng, J. Li, Y. Li, J. Guo, X. Wu and J. Chanussot, "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network," 
# IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 11, pp. 8059-8076, Nov. 2020, doi: 10.1109/TGRS.2020.2986313.
###########################################################################

from __future__ import absolute_import, division
from keras.models import Model
import numpy as np
import h5py
import scipy.io as sio

from train_DARN import eval_darn

if __name__ == "__main__":
    
    inputs, outputs = eval_darn()
    model = Model(inputs=inputs, outputs=outputs)
    model.load_weights('./model_darn_pa.h5', by_name=True)
    
    for i in range(7):
        
        ind = i+1
        
        print ('processing for %d'%ind)

        # load DHP_SR_results (pre-upsampled Hup) for the subsequent summation
        data_predLHS = sio.loadmat('./data_process/DHP_SR/DHP_SR_results/mats_pavia/test_7mats/pred%d.mat'%ind)
        
        data_pred = np.float64((data_predLHS['pred']))

        # load the input (pre-concatenated Hin) for testing
        data = h5py.File('./data_process/cat_LHS_PAN/I_predLHSpa_Pan/test_7Hin/%d.mat'%ind)
        
        data = np.transpose(data['predLHSpa_Pan'])
        
        data = np.expand_dims(data,0)
    
        data_res = model.predict(data, batch_size=1, verbose=1)
        
        data_res = np.reshape(data_res, (160, 160, 102))
        
        data_res = np.array(data_res, dtype=np.float64)
        
        # Obtaining the fused HSI Hfus by summation operation
        data_fus = data_pred + data_res
        
        sio.savemat('./get_pa7Hfus/getHfus_%d.mat'%ind, {'Hfus': data_fus})        
    