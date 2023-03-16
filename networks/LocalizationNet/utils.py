import pandas as pd
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1.):
    
    y_true = tf.math.equal(y_true, 1)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = y_pred[:,:,:,:,1]
        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(K.pow(y_true_f,2)) + K.sum((K.pow(y_pred_f,2))) + smooth)

def get_weighted_sparse_categorical_crossentropy(weights):
    r"""L = - \sum_i weights[i] y_true[i] \log(y_pred[i])
    :param weights: a list of weights for each class.
    :return: loss function.
    """
    weights = K.constant(weights, dtype=K.floatx())

    def _loss(y_true, y_pred):
        print(y_true.shape)
        print(y_pred.shape)
        return K.gather(weights, K.cast(y_true, 'int32')) \
            * keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return _loss