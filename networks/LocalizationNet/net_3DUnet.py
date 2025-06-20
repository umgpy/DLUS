import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, concatenate, BatchNormalization 
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LeakyReLU, Reshape, Lambda
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras
import numpy as np

# Model network 
DEPTH = 4
RESIDUAL = True
DEEP_SUPERVISION = True
FILTER_GROW = True
NUM_CLASS = 2
BASE_FILTER = 16

def myConv(x_in, nf, strides=1, kernel_size = 3):
    
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x_out = Conv3D(nf, kernel_size=3, padding='same',kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = BatchNormalization()(x_out)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

def Unet3dBlock(l, n_feat):
    if RESIDUAL:
        l_in = l
    for i in range(2):
        l = myConv(l, n_feat)
    return add([l_in, l]) if RESIDUAL else l

def UnetUpsample(l, num_filters):
    l = UpSampling3D()(l)
    l = myConv(l, num_filters)
    return l

def unet3d(vol_size):
    inputs = Input(shape=vol_size)
    depth = DEPTH
    filters = []
    down_list = []
    deep_supervision = None
    layer = myConv(inputs, BASE_FILTER)
    
    for d in range(depth):
        if FILTER_GROW:
            num_filters = BASE_FILTER * (2**d)
        else:
            num_filters = BASE_FILTER
        filters.append(num_filters)
        layer = Unet3dBlock(layer, n_feat = num_filters)
        down_list.append(layer)
        if d != depth - 1:
            layer = myConv(layer, num_filters*2, strides=2)
        
    for d in range(depth-2, -1, -1):
        layer = UnetUpsample(layer, filters[d])
        layer = concatenate([layer, down_list[d]])
        layer = myConv(layer, filters[d])
        layer = myConv(layer, filters[d], kernel_size = 1)
        
        if DEEP_SUPERVISION:
            if 0< d < 3:
                pred = myConv(layer, NUM_CLASS)
                if deep_supervision is None:
                    deep_supervision = pred
                else:
                    deep_supervision = add([pred, deep_supervision])
                deep_supervision = UpSampling3D()(deep_supervision)
    
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    
    if DEEP_SUPERVISION:
        layer = add([layer, deep_supervision])
    layer = myConv(layer, NUM_CLASS, kernel_size = 1)
    x = Activation('softmax', name='softmax')(layer)
        
    model = Model(inputs=[inputs], outputs=[x])
    return model