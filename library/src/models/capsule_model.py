from src.imports import tf, K, keras
from src.layers import cnn, capsule

class BlocksCapsule(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, learning_rate=None, lr_schedule=0.001, compile=True):
        super(BlocksCapsule, self).__init__()
        self.CNN = cnn.CNN()
        self.Capsule = capsule.Capsule(num_clases=2, vec=16) 
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, weights, bias)
    
    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3
    
    def build_model(self, input_shape, srm_weights,biasSRM):
        tf.keras.backend.clear_session()

        #Layer 1
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=self.__Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=self.__Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        #Layer 2

        # L1-L2
        layers = self.CNN.Block_1(layers1, 64)
        layers = self.CNN.Block_1(layers, 64)

        # L3 - L7
        for i in range(5):
            layers = self.CNN.Block_2(layers, 64)

        # L8 - L11
        for i in [64, 64, 128, 256]:
            layers = self.CNN.Block_3(layers, i)

        layers = self.CNN.Block_4(layers, 512)

        representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(layers)
        representation = tf.keras.layers.GlobalAvgPool2D()(representation)

        primary_capsule = self.Capsule(representation)
        secondary_capsule = tf.keras.layers.Lambda(self.Capsule.safe_norm)(primary_capsule)

        secondary_capsule = tf.reshape(secondary_capsule, (-1, secondary_capsule.shape[2], secondary_capsule.shape[3]))
        predictions = tf.keras.layers.Lambda(self.Capsule.output_layer)(secondary_capsule)

        model = tf.keras.Model(inputs=input_shape, outputs=predictions)
        
        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.95)
        
        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        print("Arquitecture capsnet created")
        return model
    
    def call(self, inputs, training=None):
        return self.model(inputs)
