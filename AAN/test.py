# py imports
import os
import sys
import glob
from argparse import ArgumentParser

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('./ext/')
import medipy
import networks
from medipy.metrics import dice
import datagenerators


def test(data_dir,
        fixed_image,
        label,
        device,
        load_model_file,
        DLR_model):

    assert DLR_model in ['VM','FAIM'], 'DLR_model should be one of VM or FAIM, found %s' % LBR_model
    
    # prepare data files
    # inside the folder are npz files with the 'vol' and 'label'.
    test_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    assert len(test_vol_names) > 0, "Could not find any testing data"
    
    fixed_vol = np.load(fixed_image)['vol'][np.newaxis, ..., np.newaxis]
    fixed_seg = np.load(fixed_image)['label']
    vol_size = fixed_vol.shape[1:-1]
    label = np.load(label)

    # device handling
    if 'gpu' in device:
        if '0' in device:
            device = '/gpu:0'
        if '1' in device:
            device = '/gpu:1'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'
    
    # load weights of model
    with tf.device(device):
        net = networks.AAN(vol_size, DLR_model)
        net.load_weights(load_model_file)

        # NN transfer model
        nn_trf_model_nearest = networks.nn_trf(vol_size, interp_method='nearest', indexing='ij')
        nn_trf_model_linear = networks.nn_trf(vol_size, interp_method='linear', indexing='ij')
    
    dice_result = [] 
    for test_image in test_vol_names:
        
        X_vol, X_seg, x_boundary = datagenerators.load_example_by_name(test_image, return_boundary=True)

        with tf.device(device):
            pred = net.predict([X_vol, fixing_vol, x_boundary])
            warp_vol = nn_trf_model_linear.predict([X_vol, pred[1]])[0,...,0]
            warp_seg = nn_trf_model_nearest.predict([X_seg, pred[1]])[0,...,0]
            
        vals, _ = dice(warp_seg, fixing_seg, label, nargout=2)
        dice_result.append(vals)

        print('Dice mean: {:.3f} ({:.3f})'.format(np.mean(vals), np.std(vals)))

    dice_result = np.array(dice_result)
    print('Average dice mean: {:.3f} ({:.3f})'.format(np.mean(dice_result), np.std(dice_result)))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./',
                        help="data folder")
    parser.add_argument("--fixed_image", type=str,
                        dest="fixed_image", default='./',
                        help="fixed image filename")
    parser.add_argument("--label", type=str,
                        dest="label", default='./',
                        help="label for testing")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--DLR_model", type=str,
                        dest="DLR_model", default='VM',
                        help="DLR model: VM or FAIM")

    args = parser.parse_args()
    test(**vars(args))
