# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
#from keras.utils import multi_gpu_model 

# project imports
import datagenerators
import networks
import losses
sys.path.append('./ext/')

def train(data_dir,
         fixed_image,
         model_dir,
         device,
         lr,
         nb_epochs,
         AAN_param,
         steps_per_epoch,
         batch_size,
         load_model_file,
         initial_epoch,
         DLR_model):
 
    # prepare data files
    train_vol_names = glob.glob(os.path.join(data_dir, '*.npz'))
    random.shuffle(train_vol_names)  # shuffle volume list_
    assert len(train_vol_names) > 0, "Could not find any training data"
    vol_size = [144,192,160]
    
    # load atlas from provided files, if atlas-based registration
    if fixed_image != './':
        fixed_vol = np.load(fixed_image)['vol'][np.newaxis, ..., np.newaxis]

    assert DLR_model in ['VM','DifVM','FAIM'], 'DLR_model should be one of VM, DifVM or FAIM, found %s' % DLR_model    
    if DLR_model == 'FAIM':
        def FAIM_loss(y_true, y_pred):
            return losses.Grad('l2').loss(y_true, y_pred) + 1e-5*losses.NJ_loss(y_true, y_pred)
        reg_loss = FAIM_loss
        reg_param = 0.01
    elif DLR_model == 'DifVM':
        reg_loss = losses.KL(prior_lambda=100).kl_loss
        reg_param = 0.01*0.01
    else: # DLR_model == 'VM'
        reg_loss = losses.Grad('l2').loss
        reg_param = 0.01
        
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # device handling
    if 'gpu' in device:
        device = '/gpu:0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'

    # prepare the model
    with tf.device(device):
        model = networks.AAN_enhanced_DLR(vol_size, DLR_model)

        # load initial weights
        if load_model_file != './':
            print('loading', load_model_file)
            model.load_weights(load_model_file)

    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size, return_edge=True)
    if fixed_image != './':
        fixed_vol_bs = np.repeat(fixed_vol, batch_size, axis=0)
        data_gen = datagenerators.gen_atlas(train_example_gen, fixed_vol_bs, batch_size=batch_size)
    else:
        data_gen = datagenerators.gen_s2s(train_example_gen, batch_size=batch_size)

    # prepare callbacks
    save_file_name = os.path.join(model_dir, '{epoch:02d}.h5')
    save_log_name = os.path.join(model_dir, 'log.csv')

    # fit generator
    with tf.device(device):

        save_callback = ModelCheckpoint(save_file_name, save_weights_only=True)
        csv_logger = CSVLogger(save_log_name, append=True)
        
        # compile
        model.compile(optimizer=Adam(lr=lr), 
                      loss=['mse', reg_loss, losses.Grad('l1').Lstructure],
                      loss_weights=[1.0, reg_param, AAN_param])
            
        # fit
        model.fit_generator(data_gen, 
                            initial_epoch=initial_epoch,
                            epochs=nb_epochs,
                            callbacks=[save_callback,csv_logger],
                            steps_per_epoch=steps_per_epoch,
                            verbose=1)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str,
                        dest="data_dir", default='./',
                        help="data folder")
    parser.add_argument("--fixed_image", type=str,
                        dest="fixed_image", default='./',
                        help="fixed image filename")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--device", type=str, default='gpu',
                        dest="device", help="cpu or gpu")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=1000,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=100,
                        help="iterations of each epoch")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial_epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")
    parser.add_argument("--lambda", type=float,
                        dest="AAN_param", default=0.5,
                        help="Lstructure lambda parameter")
    parser.add_argument("--DLR_model", type=str,
                        dest="DLR_model", default='VM',
                        help="DLR model: VM, DifVM or FAIM")

    args = parser.parse_args()
    train(**vars(args))
