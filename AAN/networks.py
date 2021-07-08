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
import losses


def AAN_enhanced_DLR(vol_size, DLR_model='VM', indexing='ij', src=None, tgt=None, boundary=None, src_feats=1, tgt_feats=1):
    
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
    Appearance_transformation = AAN(x_in1)
    
    Appearance_transformed_src = add([Appearance_transformation,src])
    Appearance_transformation_with_boundary = concatenate([Appearance_transformation, boundary])
    
    x_in2 = concatenate([Appearance_transformed_src, tgt])    
    if DLR_model == 'VM':
        flow = VM(x_in2)
    elif DLR_model == 'DifVM':
        flow, flow_params = DifVM(x_in2)
    else: # DLR_model == 'FAIM'
        flow = FAIM(x_in2)
    
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([Appearance_transformed_src, flow])    

    if DLR_model == 'DifVM':
        flow = flow_params
    
    return Model(inputs=[src, tgt, boundary], outputs=[y, flow, Appearance_transformation_with_boundary])
    

def AAN(x_in):
    
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


def DifVM(x_in, model='vm2', full_size=False, int_steps=7, vel_resize=1/2):
   
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
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[-5]])
        x = conv_block(x, dec_nf[4])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 6:
        x = conv_block(x, dec_nf[5])

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow_mean = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x)
    # we're going to initialize the velocity variance very low, to start stable.
    flow_log_sigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=keras.initializers.Constant(value=-10),
                            name='log_sigma')(x)
    flow_params = concatenate([flow_mean, flow_log_sigma])

    # velocity sample
    flow = Sample(name="z_sample")([flow_mean, flow_log_sigma])

    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(flow)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')

    return flow, flow_params


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


def Sample_nn_trf(vol_size, int_steps=7, vel_resize=1/2, interp_method='nearest', indexing='ij'):

    ndims = len(vol_size)
    flow_size = [i*vel_resize for i in vol_size]
    
    subj_input = Input((*vol_size, 1), name='subj_input')
    flow_input = Input((*flow_size, 2*ndims) , name='flow_input')

    flow_mean = Lambda(lambda x: x[...,0:ndims])(flow_input)
    flow_log_sigma = Lambda(lambda x: x[...,ndims:])(flow_input)
    
    # velocity sample
    flow = Sample(name="z_sample")([flow_mean, flow_log_sigma])
    
    # integrate if diffeomorphic (i.e. treating 'flow' above as stationary velocity field)
    flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(flow)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')

    # transform
    warp_vol = nrn_layers.SpatialTransformer(interp_method=interp_method, indexing=indexing)([subj_input, flow])   
    
    return Model(inputs=[subj_input, flow_input], outputs=[warp_vol, flow])

########################################################
# Helper functions
########################################################

def conv_block(x_in, nf, strides=1):
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out


def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


def trf_resize(trf, vel_resize, name='flow'):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return  nrn_layers.Resize(1/vel_resize, name=name)(trf)


class Sample(Layer):
    """ 
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    
class Negate(Layer):
    """ 
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape

    
class Rescale(Layer):
    """ 
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape

    
class RescaleDouble(Rescale):
    def __init__(self, **kwargs):
        self.resize = 2
        super(RescaleDouble, self).__init__(self.resize, **kwargs)

        
class ResizeDouble(nrn_layers.Resize):
    def __init__(self, **kwargs):
        self.zoom_factor = 2
        super(ResizeDouble, self).__init__(self.zoom_factor, **kwargs)
