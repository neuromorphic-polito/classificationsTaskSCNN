import tensorflow as tf


class Masking(tf.keras.constraints.Constraint):
  def __init__(self, mask=0):
    self.mask = mask

  def __call__(self, w):
    return tf.multiply(w, self.mask)


def Relu(inputs):
    S = 201.0
    tau_syn = 0.005
    return tf.multiply(S*tau_syn, tf.math.maximum(0.0, inputs))


def netModelsComplete(structure, dataShape, datasetClass):
    modelCNN = None
    if structure == 'c06c12f2':
        modelCNN = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, input_shape=dataShape),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=768, activation='relu', use_bias=False),
            tf.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False)
        ])
    elif structure == 'c12c24f2':
        modelCNN = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, input_shape=dataShape),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=Relu, use_bias=False),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=768, activation='relu', use_bias=False),
            tf.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False)
        ])
    else:
        raise Exception('Inserted structure not available')

    return modelCNN


def netModelsPruned(structure, dataShape, mask, datasetClass):
    modelCNN = None
    if structure == 'c06c12f2':
        modelCNN = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[0]), input_shape=dataShape),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[1])),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=768, activation='relu', use_bias=False, kernel_constraint=Masking(mask[2])),
            tf.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False, kernel_constraint=Masking(mask[3]))
        ])
    elif structure == 'c12c24f2':
        modelCNN = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=12, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[0]), input_shape=dataShape),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=Relu, use_bias=False, kernel_constraint=Masking(mask[1])),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=768, activation='relu', use_bias=False, kernel_constraint=Masking(mask[2])),
            tf.keras.layers.Dense(units=datasetClass, activation='softmax', use_bias=False, kernel_constraint=Masking(mask[3]))
        ])
    else:
        raise Exception('Inserted structure not available')

    return modelCNN