import h5py
import os
import pdb

# Check if file is done being written
def check_if_finished(file_path):
    done = False
    if os.path.exists(file_path):
        with h5py.File(file_path, 'r') as hf:
            if 'val' in file_path:
                done = len(hf.keys()) == 248
    return done
