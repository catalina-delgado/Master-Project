from src.imports import tf, K, keras, np
from src.layers import cnn, transformer_keras

class CVT(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, hyperparams, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVT, self).__init__()
        self.CNN = cnn.CNN()
        self.transformer = transformer_keras.Transformer(hyperparams)
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.DCTfilter = self.DCT_filter()
        self.model = self.build_model(inputs, weights, bias, hyperparams)
    
    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3
    
    def DCT_filter(self):
        with tf.name_scope("DCT"):
            # Initialize DCT filters
            DCT_filter_n = np.zeros([3, 3, 64, 64])
            # Definition of 8x8 mesh grid
            XX, YY = np.meshgrid(range(3), range(3))
            # DCT basis as filters
            C=np.ones(5)
            C[0]=1/np.sqrt(2)
            for v in range(3):
                for u in range(3):
                    DCT_filter_n[:, :, 0, u + v * 3] = (2 * C[v] * C[u] / 3) * np.cos((2 * YY + 1) * v * np.pi / 6) * np.cos((2 * XX + 1) * u * np.pi / 6)

            DCT_filter=tf.constant(DCT_filter_n.astype(np.float32))

            return DCT_filter
    
    def build_model(self, input_shape, srm_weights,biasSRM, hyperparams):
        tf.keras.backend.clear_session()

        inputs = self.transformer(input_shape)

        #Preprocessing
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=self.__Tanh3, use_bias=True)(inputs)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=self.__Tanh3, use_bias=True)(inputs)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        # Block A
        layers = self.CNN.Block_1(layers1, 64)
        layers = self.CNN.Block_1(layers, 64)

        # Block B
        for i in range(2):
            layers = self.CNN.Block_2(layers, 64)

        layers_ty = tf.keras.layers.Conv2D(64, (3,3), weights=[self.DCTfilter,np.ones(64)], strides=(1,1), padding='same', trainable=False, activation=self.__Tanh3, use_bias=True)(layers)
        layers_tn = tf.keras.layers.Conv2D(64, (3,3), weights=[self.DCTfilter,np.ones(64)], strides=(1,1), padding='same', trainable=True, activation=self.__Tanh3, use_bias=True)(layers)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers2 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        for i in range(2):
            layers = self.CNN.Block_2(layers2, 64)

        # Block C
        for i in [64, 64, 128, 256]:
            layers = self.CNN.Block_3(layers, i)

        # Convolutional Visual Transformer
        #layers = self.CNN.Block_4(layers, 512)
        print('output last layer before transformer', layers.shape)

        representation = tf.keras.layers.LayerNormalization(epsilon=hyperparams['LAYER_NORM_EPS_2'])(layers)
        representation = tf.keras.layers.GlobalAvgPool2D()(representation)

        layers = self.CNN.fully_connected(representation)
        predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        model = tf.keras.Model(inputs=input_shape, outputs=predictions)
        
        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.95)
        
        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        print("Arquitecture1 creada")
        return model
    
    def call(self, inputs, training=None):
        return self.model(inputs)
