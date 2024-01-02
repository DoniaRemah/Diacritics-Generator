import h5py
import numpy as np

def print_weights(group, prefix=''):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            txt_file.write(f"{prefix}/{key}:\n")
            weights = np.array(item)
            txt_file.write(str(weights) + '\n')
        elif isinstance(item, h5py.Group):
            txt_file.write(f"{prefix}/{key}:\n")
            print_weights(item, prefix=f"{prefix}/{key}")

# Open the text file in write mode
with open('weights.txt', 'w') as txt_file:
    # Load the weights from the .h5 file
    with h5py.File('models/weights/folder_30000_35000/weights_epoch_01.h5', 'r') as f:
        print_weights(f)