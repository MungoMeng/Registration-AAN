# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, Conv3DTranspose, ZeroPadding3D, AveragePooling3D
from keras.layers import LeakyReLU, Reshape, Lambda, PReLU, add
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

sys.path.append('./ext/')
import neuron.layers as nrn_layers
import neuron.models as nrn_models
import neuron.utils as nrn_utils
import losses


def AAN(vol_size, DLR_model='VM', indexing='ij', src=None, tgt=None, boundary=None, src_feats=1, tgt_feats=1):
    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    
    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    if boundary is None:
        boundary = Input(shape=[*vol_size, 1])
        
    x_in1 = concatenate([src, tgt, boundary])
    Appearance_transformation = U_net(x_in1)
    
    Appearance_transformed_src = KL.Add()([Appearance_transformation,src])
    Appearance_transformation_with_boundary = concatenate([Appearance_transformation, boundary])
    
    x_in2 = concatenate([Appearance_transformed_src, tgt])    
    if DLR_model == 'VM':
        flow = VM(x_in2)
    else: # DLR_model == 'FAIM'
        flow = FAIM(x_in2)
    
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([Appearance_transformed_src, flow])    

    return Model(inputs=[src, tgt, boundary], outputs=[y, flow, Appearance_transformation_with_boundary])
    

def U_net(x_in):
    
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    
    enc_nf = [8, 16, 32, 64]
    dec_nf = [64, 32, 16, 8]
    
    # down-sample path (encoder)
    x_enc = [x_in]
    x_enc.append(conv_block(x_enc[-1], enc_nf[0]))
    for i in range(1, len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    
    # transform the results into a difference map.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    Appearance_transformation = Conv(1, kernel_size=3, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    return Appearance_transformation    

    
def VM(x_in, model='vm2', full_size=True):

    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    enc_nf = [16, 32, 32, 32, 32]
    if model == 'vm1':
        dec_nf = [32, 32, 32, 8, 8]
    elif model == 'vm2':
        dec_nf = [32, 32, 32, 32, 16, 16]
    else: # 'vm2double': 
        enc_nf = [f*2 for f in nf_enc]
        dec_nf = [f*2 for f in [32, 32, 32, 32, 16, 16]]

    # down-sample path (encoder)
    x_enc = [x_in]
    x_enc.append(conv_block(x_enc[-1], enc_nf[0]))
    for i in range(1, len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[-5]])
        x = conv_block(x, dec_nf[4])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 6:
        x = conv_block(x, dec_nf[5])

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)
    return flow


def FAIM(x_in):

    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims   

    z1_1 = incept(x_in,4)
    z1_2 = Conv3D(32, kernel_size=3, strides = 2, padding = 'valid')(z1_1)
    z1_2 = PReLU(shared_axes = [1,2,3])(z1_2)

    z2_1 = Conv3D(32, kernel_size=3, padding = 'same')(z1_2)
    #z2_1 = PReLU(shared_axes = [1,2,3])(z2_1)
    z2_2 = Conv3D(32, kernel_size=3, strides = 2, padding = 'valid')(z2_1)
    z2_2 = PReLU(shared_axes =[1,2,3])(z2_2)

    z3 = Conv3D(32, (2,2,2), padding = 'same')(z2_2)
    #z3 = PReLU(shared_axes = [1,2,3])(z3)

    z3 = add([z3, z2_2])

    z4 = Conv3DTranspose(32, kernel_size=3, strides=2, padding = 'valid')(z3)
    #z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = Conv3D(32, kernel_size=3, padding = 'same', activation = 'linear')(z4)
    z4 = PReLU(shared_axes = [1,2,3])(z4)
    z4 = add([z4, z1_2])

    z5 = Conv3DTranspose(32, kernel_size=3, strides=2, padding = 'valid')(z4)
    #z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = Conv3D(16, kernel_size=3, padding = 'same', activation = 'linear')(z5)
    z5 = PReLU(shared_axes = [1,2,3])(z5)
    z5 = ZeroPadding3D(((0,1),(0,1),(0,1)))(z5)   #Extra padding to make size match

    z5 = add([z5, z1_1])

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow', kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(z5)
    return flow


def incept(inputs, num_channel, activation = 'linear'):
    z1 = Conv3D(num_channel, (2,2,2), padding = 'same', activation = activation)(inputs)
    z2 = Conv3D(num_channel, (3,3,3), padding = 'same', activation = activation)(inputs)
    z3 = Conv3D(num_channel, (5,5,5), padding = 'same', activation = activation)(inputs)
    z4 = Conv3D(num_channel, (7,7,7), padding = 'same', activation = activation)(inputs)
   
    z = concatenate([z4, z3, z2, z1])
    z = PReLU(shared_axes = [1,2,3])(z)
    return z


def nn_trf(vol_size,interp_method='nearest', indexing='xy'):
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size, 1), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    nn_output = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)


def conv_block(x_in, nf, strides=1):
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out
