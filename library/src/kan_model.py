import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from functions.main import Main
from tfkan.layers import DenseKAN

def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
        excitation = tf.keras.layers.Dense(units=out_dim / ratio, activation='relu')(squeeze)
        excitation = tf.keras.layers.Dense(out_dim,activation='sigmoid')(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = tf.keras.layers.multiply([input_layer, excitation])
        if conv:
            shortcut = tf.keras.layers.Conv2D(out_dim,kernel_size=1,strides=1,
                                            padding='same',kernel_initializer='he_normal')(input_layer)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = input_layer
        out = tf.keras.layers.add([shortcut, scale])
        return out

def __sreLu (input):
    return tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(input)

def __sConv(input,parameters,size,nstrides):
    return tf.keras.layers.Conv2D(parameters, (size,size), strides=(nstrides,nstrides),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)

def __sBN (input):
    return tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(input)

def __sGlobal_Avg_Pooling (input):
    return tf.keras.layers.GlobalAveragePooling2D()(input)

def __sDense (input, n_units, activate_c):
    if activate_c == "leakyrelu":
        activate_c = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)
    return tf.keras.layers.Dense(n_units,activation=activate_c)(input)

def __smultiply (input_1, input_2):
    return tf.keras.layers.multiply([input_1, input_2])

def __sadd (input_1, input_2):
    return tf.keras.layers.add([input_1, input_2])

def Block_1 (input, parameter):
        output = __sConv(input, parameter, 3, 1)
        output = __sBN(output)
        output = __sreLu(output)
        return output

def SE_Block(input, out_dim, ratio):
    output = __sGlobal_Avg_Pooling(input)
    output = __sDense(output, out_dim/ratio, 'relu')
    output = __sDense(output, out_dim, 'sigmoid')
    return output

def Block_2 (input, parameter):
        output = Block_1(input, parameter)
        output = __sConv(output, parameter, 3, 1)
        output = __sBN(output)
        multiplier = SE_Block(output,  parameter, parameter)
        output = __smultiply(multiplier, output)
        output = __sadd(output, input)
        return output

def Block_3 (input, parameter):
        addition = __sConv(input, parameter, 1, 2)
        addition = __sBN(addition)
        output = __sConv(input, parameter, 3, 2)
        output = __sBN(output)
        output = __sreLu(output)
        output = __sConv(output, parameter, 3, 1)
        output = __sBN(output)
        multiplier = SE_Block(output,  parameter, parameter)
        output = __smultiply(multiplier, output)
        output = __sadd(output, addition)
        return output  

def Block_4 (input, parameter):
        output = Block_1(input, parameter)
        output = __sConv(input, parameter, 3, 1)
        output = __sBN(output)
        return output  

def fully_connected_kan (input):
        output = DenseKAN(16)(input)
        output = DenseKAN(4)(output)
        return output

def Tanh3(x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

class BlocksKAN(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, learning_rate=None, lr_schedule=0.001, compile=True):
        super(BlocksKAN, self).__init__()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, weights, bias)
    
    def build_model(self, input_shape, srm_weights,biasSRM):
        tf.keras.backend.clear_session()

        #Layer 1
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        #Layer 2

        # L1-L2
        layers = Block_1(layers1, 64)
        layers = Block_1(layers, 64)

        # L3 - L7
        for i in range(5):
            layers = Block_2(layers, 64)

        # L8 - L11
        for i in [64, 64, 128, 256]:
            layers = Block_3(layers, i)

        layers = Block_4(layers, 512)

        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(layers)
        representation = tf.keras.layers.GlobalAvgPool2D()(representation)

        layers = fully_connected_kan(representation)
        layers = DenseKAN(2)(layers)
        predictions = tf.keras.layers.Softmax(axis=1)(layers)
        model = tf.keras.Model(inputs=input_shape, outputs=predictions)
        
        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.95)
        
        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        print("Arquitecture KAN created")
        return model
    
    def call(self, inputs, training=None):
        return self.model(inputs)

class TrainingKAN(Main):
    def __init__(self, epochs, batch_size, dataset):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATASET = dataset

    def train_kan_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = BlocksKAN(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        #prueba 1 learning_rate 5e-3 3FC 64-32-16
        #prueba 2 learning_rate 1e-3 2FC -16-4
        
        #self.plot_model_summary(architecture.model, 'kan_model_summary')

        X_train = np.load('../../database/BOSS/'+self.DATASET+'/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../../database/BOSS/'+self.DATASET+'/y_train.npy') # (12000, 2)
        X_valid = np.load('../../database/BOSS/'+self.DATASET+'/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../../database/BOSS/'+self.DATASET+'/y_valid.npy') # (4000, 2)
        X_test = np.load('../../database/BOSS/'+self.DATASET+'/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../../database/BOSS/'+self.DATASET+'/y_test.npy')   # (4000, 2)

        base_name="04-"+self.DATASET
        name="Model_"+'KAN_prueba1'+"_"+base_name
        _, history  = self.fit(
            architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, dataset=self.DATASET,
            custom_layers={'Tanh3':Tanh3}
        )

train = TrainingKAN(epochs=3, batch_size=8, dataset='S-UNIWARD')
train.train_kan_model()