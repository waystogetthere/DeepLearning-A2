from tensorflow.python.keras import backend as K
import tensorflow as tf


def mzz_losss(y_true, y_pred):
    result = K.sum(-tf.log(K.sum(y_true * y_pred, axis=1)))
    # result = K.sum(-1 * tf.log(y_true * y_pred)) / K.cast(tf.shape(y_true)[0], K.floatx())
    return result


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def mzz(y_true, y_pred):
    pred_indices = K.argmax(y_pred, axis=1)
    real = K.argmax(y_true * y_pred, axis=1)
    result = tf.cast(tf.count_nonzero(pred_indices - real), dtype=tf.int32)
    # sess = tf.Session()
    return K.argmax(y_pred, axis=1)


def mzz_metrics(y_true, y_pred):
    '''

    :param y_true: a tensor of shape: (batch_size, num_class), which represents the ground truth labels
    :param y_pred: a tensor of shape: (batch_size, num_class), which represents the predicted labels
    :return:
    '''
    pred_indices = K.argmax(y_pred, axis=1)
    real = K.argmax(y_true * y_pred, axis=1)
    result = (tf.shape(pred_indices)[0] - tf.cast(tf.count_nonzero(pred_indices - real), dtype=tf.int32)) / \
             tf.shape(pred_indices)[0]
    return result
    # return K.argmax(y_pred),
    # return K.sum(K.round(y_pred))
    # return K.cast(K.sum(K.clip(K.sum(y_true * K.round(y_pred), axis=1), 0, 1)), dtype=tf.int32) / tf.shape(y_true)[0]


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
