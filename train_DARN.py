###########################################################################
# Created by: Yuxuan Zheng
# Email: yxzheng24@163.com
# Training code for DARN proposed in the paper titled "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network"

# Citation
# Y. Zheng, J. Li, Y. Li, J. Guo, X. Wu and J. Chanussot, "Hyperspectral Pansharpening Using Deep Prior and Dual Attention Residual Network," 
# IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 11, pp. 8059-8076, Nov. 2020, doi: 10.1109/TGRS.2020.2986313.
###########################################################################

from __future__ import absolute_import, division
from keras.layers import Input, Conv2D, Activation, add

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot
import h5py
from keras.callbacks import ModelCheckpoint

from csa_module import channel_spatial_attention_block

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def tf_log10(x):
    n = tf.log(x)
    d = tf.log(tf.constant(10, dtype = n.dtype))
    return n/d

def psnr(y_ture, y_pred):
    max_pixel =1.0
    return 10.0*tf_log10((max_pixel**2)/(K.mean(K.square(y_pred-y_ture))))
    
def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def eval_darn():
    inputs = l = Input((160, 160, 103), name='input')

    # conv11
    init = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=(1, 1), name='conv11')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv12')(init)
    l = Activation('relu', name='conv12_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv13')(l)
    # CSA module 1
    l = channel_spatial_attention_block(l)
    init1 = add([l, init])

    # conv14
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv14')(init1)
    l = Activation('relu', name='conv14_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv15')(l)
    # CSA module 2
    l = channel_spatial_attention_block(l)
    init2 = add([l, init1])    

    # conv16    
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv16')(init2)
    l = Activation('relu', name='conv16_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv17')(l)
    # CSA module 3
    l = channel_spatial_attention_block(l)
    init3 = add([l, init2])      

    # conv18
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv18')(init3)
    l = Activation('relu', name='conv18_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv19')(l)
    # CSA module 4
    l = channel_spatial_attention_block(l)
    init4 = add([l, init3])

    # conv20    
    l = Conv2D(102, (3, 3), padding='same', kernel_initializer='he_normal', strides=(1, 1), name='conv20')(init4)  
    
    # output
    outputs = l

    return inputs, outputs


def train_darn():
    inputs = l = Input((32, 32, 103), name='input')

    # conv11
    init = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=(1, 1), name='conv11')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv12')(init)
    l = Activation('relu', name='conv12_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv13')(l)
    # CSA module 1
    l = channel_spatial_attention_block(l)
    init1 = add([l, init])

    # conv14
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv14')(init1)
    l = Activation('relu', name='conv14_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv15')(l)
    # CSA module 2
    l = channel_spatial_attention_block(l)
    init2 = add([l, init1])    

    # conv16    
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv16')(init2)
    l = Activation('relu', name='conv16_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv17')(l)
    # CSA module 3
    l = channel_spatial_attention_block(l)
    init3 = add([l, init2])      

    # conv18
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=(1, 1), name='conv18')(init3)
    l = Activation('relu', name='conv18_relu')(l)
    l = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, name='conv19')(l)
    # CSA module 4
    l = channel_spatial_attention_block(l)
    init4 = add([l, init3])

    # conv20     
    l = Conv2D(102, (3, 3), padding='same', kernel_initializer='he_normal', strides=(1, 1), name='conv20')(init4)
    
    # output
    outputs = l

    return inputs, outputs
    
if __name__ == "__main__":
    data_dir = os.path.join('./train_predLHSpaPanMapRes_turn.h5')
    
    train_data, train_label = read_data(data_dir)
    
#    train_data = np.transpose(train_data,(0,2,3,1))
#    train_label = np.transpose(train_label,(0,2,3,1))
        
    inputs, outputs = train_darn()
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    optim = Adam(1e-3)
    loss = 'mae'
    model.compile(optim, loss, metrics=[psnr])
    
    checkpointer = ModelCheckpoint(filepath="./model_darn_pa.h5", verbose=1, save_best_only=True)
    
    x_train = train_data
    y_train = train_label
    
    history = model.fit(x_train, y_train, epochs=1500, batch_size=128, validation_split=0.2, initial_epoch=0, callbacks=[checkpointer])
    
    
    pyplot.plot(history.history['loss'], label='train_loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.legend()
    pyplot.show()
    
