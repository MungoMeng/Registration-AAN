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


def Get_Num_Neg_Ja(displacement):

    D_y = (displacement[1:,:-1,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_x = (displacement[:-1,1:,:-1,:] - displacement[:-1,:-1,:-1,:])
    D_z = (displacement[:-1,:-1,1:,:] - displacement[:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])
    Ja_value = D1-D2+D3
    
    return np.sum(Ja_value<0)


def test(data_dir,
        fixed_image,
        label,
        device,
        load_model_file,
        DLR_model):

    assert DLR_model in ['VM','DifVM','FAIM'], 'DLR_model should be one of VM, DifVM or FAIM, found %s' % LBR_model
    
    # prepare data files
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
        net = networks.AAN_enhanced_DLR(vol_size, DLR_model)
        net.load_weights(load_model_file)

        # NN transfer model
        if DLR_model in ['VM','FAIM']:
            nn_trf_model_nearest = networks.nn_trf(vol_size, interp_method='nearest', indexing='ij')
            nn_trf_model_linear = networks.nn_trf(vol_size, interp_method='linear', indexing='ij')
        else: # DLR_model == 'DifVM'
            nn_trf_model_nearest = networks.Sample_nn_trf(vol_size, interp_method='nearest', indexing='ij')
            nn_trf_model_linear = networks.Sample_nn_trf(vol_size, interp_method='linear', indexing='ij')
    

    dice_result = [] 
    Ja_result = []
    Runtime_result = []
    for test_image in test_vol_names:
        print(test_image)
        
        X_vol, X_seg, X_edge = datagenerators.load_example_by_name(test_image, return_edge=True)

        with tf.device(device):
            
            t = time.time()
            pred = net.predict([X_vol, fixed_vol, X_edge])
            Runtime_vals = time.time() - t
            
            if DLR_model in ['VM','FAIM']:
                warp_vol = nn_trf_model_linear.predict([X_vol, pred[1]])        
                warp_seg = nn_trf_model_nearest.predict([X_seg, pred[1]])
                warp_vol = warp_vol[0,...,0]
                warp_seg = warp_seg[0,...,0]
                flow = pred[1][0,...]
            else: # DLR_model == 'DifVM'
                [warp_vol,flow] = nn_trf_model_linear.predict([X_vol, pred[1]])
                [warp_seg,flow] = nn_trf_model_nearest.predict([X_seg, pred[1]])
                warp_vol = warp_vol[0,...,0]
                warp_seg = warp_seg[0,...,0]
                flow = flow[0,...]
        
        Dice_vals, _ = dice(warp_seg, fixed_seg, label, nargout=2)
        dice_result.append(Dice_vals)
        print('Dice mean: {:.3f} ({:.3f})'.format(np.mean(Dice_vals), np.std(Dice_vals)))
        
        Ja_vals = Get_Num_Neg_Ja(flow)
        Ja_result.append(Ja_vals)
        print('Jacobian mean: {:.3f}'.format(np.mean(Ja_vals)))
        
        Runtime_result.append(Runtime_vals)
        print('Runtime mean: {:.3f}'.format(np.mean(Runtime_vals)))

    dice_result = np.array(dice_result)
    print('Average dice mean: {:.3f} ({:.3f})'.format(np.mean(dice_result), np.std(dice_result)))
    Ja_result = np.array(Ja_result)
    print('Average Jabobian mean: {:.3f} ({:.3f})'.format(np.mean(Ja_result), np.std(Ja_result)))
    Runtime_result = np.array(Runtime_result)
    print('Average Runtime mean: {:.3f} ({:.3f})'.format(np.mean(Runtime_result), np.std(Runtime_result)))

    
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
                        help="DLR model: VM, DifVM, or FAIM")

    args = parser.parse_args()
    test(**vars(args))
