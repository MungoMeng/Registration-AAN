import os, sys
import numpy as np
        

def gen_atlas(gen, atlas_vol_bs, batch_size=1):
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)
        yield ([X[0], atlas_vol_bs, X[1]], [atlas_vol_bs, zeros, zeros])
        
        
def gen_s2s(gen, batch_size=1):
    zeros = None
    while True:
        X1 = next(gen)
        X2 = next(gen)
        
        if zeros is None:
            volshape = X1[0].shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
            
        yield ([X1[0], X2[0], X1[1]], [X2[0], zeros, zeros])
        

def example_gen(vol_names, batch_size=1, return_segs=False, return_boundary=False):
    
    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)

        try:
            X_data = []
            for idx in idxes:
                X = load_volfile(vol_names[idx], np_var='vol')
                X = X[np.newaxis, ..., np.newaxis]
                X_data.append(X)

            if batch_size > 1:
                return_vals = [np.concatenate(X_data, 0)]
            else:
                return_vals = [X_data[0]]

            # also return segmentations
            if return_segs:
                X_data = []
                for idx in idxes:
                    X_seg = load_volfile(vol_names[idx], np_var='label')
                    X_seg = X_seg[np.newaxis, ..., np.newaxis]
                    X_data.append(X_seg)
            
                if batch_size > 1:
                    return_vals.append(np.concatenate(X_data, 0))
                else:
                    return_vals.append(X_data[0])
        
            # also return boundaey
            if return_boundary:
                X_data = []
                for idx in idxes:
                    X_boundary = load_volfile(vol_names[idx], np_var='boundary')
                    X_boundary = X_boundary[np.newaxis, ..., np.newaxis]
                    X_data.append(X_boundary)
            
                if batch_size > 1:
                    return_vals.append(np.concatenate(X_data, 0))
                else:
                    return_vals.append(X_data[0])

            yield tuple(return_vals)
        
        except:
            print('Failed to load '+ vol_names[idxes[0]])
            continue


def load_example_by_name(vol_name, return_boundary=False):

    X = load_volfile(vol_name, 'vol')
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(vol_name, np_var='label')
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)
    
    if return_boundary == True:
        X_boundary = load_volfile(vol_name, np_var='boundary')
        X_boundary = X_boundary[np.newaxis, ..., np.newaxis]
        
        return_vals.append(X_boundary)

    return tuple(return_vals)


def load_volfile(datafile, np_var):

    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        X = np.load(datafile)[np_var]

    return X
