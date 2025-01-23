from src.imports import tf, DenseKAN

class CNN():
    
    def squeeze_excitation_layer(self, input_layer, out_dim, ratio, conv):
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

    def __sreLu (self, input):
        return tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(input)

    def __sConv(self, input,parameters,size,nstrides):
        return tf.keras.layers.Conv2D(parameters, (size,size), strides=(nstrides,nstrides),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)

    def __sBN (self, input):
        return tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(input)

    def __sGlobal_Avg_Pooling (self, input):
        return tf.keras.layers.GlobalAveragePooling2D()(input)

    def __sDense (self, input, n_units, activate_c):
        if activate_c == "leakyrelu":
            activate_c = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)
        return tf.keras.layers.Dense(n_units,activation=activate_c)(input)

    def __smultiply (self, input_1, input_2):
        return tf.keras.layers.multiply([input_1, input_2])

    def __sadd (self, input_1, input_2):
        return tf.keras.layers.add([input_1, input_2])

    # Initial learning rate for the optimizer
    initial_learning_rate = 0.001

    # Learning rate schedule with exponential decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    def Block_1 (self, input, parameter):
        output = self.__sConv(input, parameter, 3, 1)
        output = self.__sBN(output)
        output = self.__sreLu(output)
        return output
        

    def SE_Block(self, input, out_dim, ratio):
        output = self.__sGlobal_Avg_Pooling(input)
        output = self.__sDense(output, out_dim/ratio, 'relu')
        output = self.__sDense(output, out_dim, 'sigmoid')
        return output
        
    
    def Block_2 (self, input, parameter):
        output = self.Block_1(input, parameter)
        output = self.__sConv(output, parameter, 3, 1)
        output = self.__sBN(output)
        multiplier = self.SE_Block(output,  parameter, parameter)
        output = self.__smultiply(multiplier, output)
        output = self.__sadd(output, input)
        return output
    

    def Block_3 (self, input, parameter):
        addition = self.__sConv(input, parameter, 1, 2)
        addition = self.__sBN(addition)
        output = self.__sConv(input, parameter, 3, 2)
        output = self.__sBN(output)
        output = self.__sreLu(output)
        output = self.__sConv(output, parameter, 3, 1)
        output = self.__sBN(output)
        multiplier = self.SE_Block(output,  parameter, parameter)
        output = self.__smultiply(multiplier, output)
        output = self.__sadd(output, addition)
        return output  
    

    def Block_4 (self, input, parameter):
        output = self.Block_1(input, parameter)
        output = self.__sConv(input, parameter, 3, 1)
        output = self.__sBN(output)
        return output    
    

    def fully_connected (self, input):
        output = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        output = tf.keras.layers.Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(output)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        output = tf.keras.layers.Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(output)
        output = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(output)
        return output
  
  
    def fully_connected_kan (self, input):
        output = DenseKAN(16)(input)
        output = DenseKAN(4)(output)
        return output
    
#class SEBlock(tf.keras.layers.Layer):
#    def __init__(self, n_units=64, **kwargs):
#        super(SEBlock, self).__init__(**kwargs)
#        self.n_units = n_units
#        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
#        self.dense_relu = tf.keras.layers.Dense(n_units/n_units, activation='relu')
#        self.dense_sigmoid = tf.keras.layers.Dense(n_units, activation='sigmoid')
#
#    def call(self, input):
#        x = self.global_avg_pooling(input)
#        x = self.dense_relu(x)
#        x = self.dense_sigmoid(x)
#        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]]) 
#        
#        x = input * x 
#        return x
#
#    def get_config(self):
#       config = super().get_config()
#        config.update({
#            'n_units': self.n_units
#        })
#        return config
#
#    @classmethod
#    def from_config(cls, config):
#        return cls(**config)
    
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, n_units=64, **kwargs):
        super(SEBlock, self).__init__()
        self.glogal_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_relu = tf.keras.layers.Dense(n_units/n_units, activation='relu')
        self.dense_sigmoid = tf.keras.layers.Dense(n_units, activation='sigmoid')

    def call(self, input):
        x = self.glogal_avg_pooling(input)
        x = self.dense_relu(x)
        x = self.dense_sigmoid(x)
        x = tf.reshape(x, [-1, 1, 1, x.shape[-1]]) 
        
        x = input * x 
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'glogal_avg_pooling': self.glogal_avg_pooling,
            'dense_relu': self.dense_relu,
            'dense_sigmoid': self.dense_sigmoid
        })
        return config
