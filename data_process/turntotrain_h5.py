
import numpy as np
import h5py

def turn_to_train(filename):
    
    file = h5py.File(filename)
    data = file['data']
    label = file['label']

    print(data.shape)
    print(label.shape)
    
    data = np.swapaxes(data, 1, 3)
    print(data.shape)

    label = np.swapaxes(label, 1, 3)
    print(label.shape)
    
    f = h5py.File('train_predLHSpaPanMapRes_turn.h5', 'w')
    f.create_dataset('data', data=data)
    f.create_dataset('label', data=label)
    

if __name__ == '__main__':
     turn_to_train('./train_predLHSpaPanMapRes.h5')
