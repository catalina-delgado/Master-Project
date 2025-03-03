import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from functions.main import Main


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

def fully_connected (input):
        output = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        output = tf.keras.layers.Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(output)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        output = tf.keras.layers.Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(output)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        return output

def Tanh3(x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

class CapsuleSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_clases=2, vec=16, input_dim=512, **kwargs):
        super(CapsuleSelfAttention, self).__init__(**kwargs)
        self.num_clases = num_clases
        self.vec = vec
        self.input_dim = input_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[1, 1, self.num_clases, self.vec, self.input_dim, ], # [2, 1, 512, ,16]
            initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),  #henormal fail, #glorotnormal fail
            trainable=True,
            name='weights_matrix'
        )
        self.b = self.add_weight(
            shape=[1, self.num_clases, self.num_clases,1], 
            initializer=tf.zeros_initializer(), 
            name='b'
        )
        #self.conv2D = tf.keras.layers.Conv2D(self.num_clases, kernel_size=3, strides=1, activation='linear', groups=self.num_clases, padding='same')

    def call(self, inputs):
        primary_capsule_output = self.primary_capsule(inputs)
        routing_output = self.routing(primary_capsule_output)
        return routing_output

    def primary_capsule(self, inputs):
        #x = self.conv2D(inputs)
        x = tf.reshape(inputs, (-1, 1, self.input_dim)) # [none, vec, dim]
        x = self.squash(x) 
        return x #[none, 1, input_dim]

    def routing(self, inputs):
        # SELF ATTENTION

        u = tf.expand_dims(inputs, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u = tf.matmul(self.W, u)  
        u = tf.squeeze(u, [4]) # [none, 1, 2, 16]
       
        u_flat = tf.reshape(u, [-1, self.num_clases, self.vec])  # (None, nun_clases, vec) 
        c = tf.matmul(u_flat, u_flat, transpose_b=True)  # Producto punto de u con sí mismo
        c = tf.expand_dims(c, -1) # (None, num_clases, num_clases, 1)

        # Normalización y softmax:
        c = c / tf.sqrt(tf.cast(self.vec, tf.float32))  
        c = tf.nn.softmax(c, axis=1) 
        c = c + self.b  # Sumar el sesgo

        s = tf.reduce_sum(c * u, axis=-2) 
        v = self.squash(s)  
        v = self.safe_norm(v) # [none, nun_clases, 1]

        return v 

    @staticmethod
    def squash(inputs, epsilon=1e-7):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (inputs / tf.sqrt(squared_norm + epsilon))


    @staticmethod
    def safe_norm(v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_clases": self.num_clases,
            "vec": self.vec,
            'input_dim': self.input_dim
        })
        return config

class BlocksCapsule(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, learning_rate=None, lr_schedule=0.001, compile=True):
        super(BlocksCapsule, self).__init__()
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

        Capsule = CapsuleSelfAttention(num_clases=2, vec=16)(representation)

        model = tf.keras.Model(inputs=input_shape, outputs=Capsule)
        
        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, momentum=0.95)
        
        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        print("Arquitecture capsnet created")
        return model
    
    def call(self, inputs, training=None):
        return self.model(inputs)

class TrainingCapsule(Main):
    def __init__(self, epochs, batch_size, dataset):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATASET = dataset

    def train_capsule_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = BlocksCapsule(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        
        X_train = np.load('../../database/BOSS/'+self.DATASET+'/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../../database/BOSS/'+self.DATASET+'/y_train.npy') # (12000, 2)
        X_valid = np.load('../../database/BOSS/'+self.DATASET+'/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../../database/BOSS/'+self.DATASET+'/y_valid.npy') # (4000, 2)
        X_test = np.load('../../database/BOSS/'+self.DATASET+'/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../../database/BOSS/'+self.DATASET+'/y_test.npy')   # (4000, 2)

        base_name="04-"+self.DATASET
        name="Model_"+'CAPSNET_selfAttention_prueba5'+"_"+base_name
        architecture.model.summary()
        _, history  = self.fit(
            architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, dataset=self.DATASET,
            custom_layers={'Tanh3':Tanh3, 'CapsuleSelfAttention':CapsuleSelfAttention}
        )

train = TrainingCapsule(epochs=3, batch_size=8, dataset='S-UNIWARD')
train.train_capsule_model()