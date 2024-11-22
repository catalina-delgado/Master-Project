from src.imports import tf, K, keras, np
from src.layers import cnn
from src.layers.swin_transformer import PatchEmbedding, SwinTransformer, PatchMerging

class CVST(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, hyperparams, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVST, self).__init__()
        self.CNN = cnn.CNN()
        self.PatchEmbedding = PatchEmbedding(hyperparams)
        self.SwinTransformer = SwinTransformer(hyperparams)
        self.PatchMerging = PatchMerging(hyperparams)
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, weights, bias, hyperparams)
    
    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3
   
    def build_model(self, input_shape, srm_weights,biasSRM, hyperparams):
        tf.keras.backend.clear_session()

        #Preprocessing
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=self.__Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=self.__Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        # Block A
        layers = self.CNN.Block_1(layers1, 64)
        layers = self.CNN.Block_1(layers, 64)

        # Block B
        for i in range(5):
            layers = self.CNN.Block_2(layers, 64)

        # Block C
        for i in [64, 64, 128, 256]:
            layers = self.CNN.Block_3(layers, i)

        # Swin Transformer
        layers = self.CNN.Block_4(layers, 512)
        print('output last layer before transformer', layers.shape)

        layers = tf.keras.layers.LayerNormalization(epsilon=hyperparams['LAYER_NORM_EPS'])(layers)

        projection = tf.keras.layers.Conv2D(
            filters=hyperparams['PROJECTION_DIM'],
            kernel_size=(hyperparams['PATCH_SIZE'], hyperparams['PATCH_SIZE']),
            strides=(hyperparams['PATCH_SIZE'], hyperparams['PATCH_SIZE']),
            padding="same"
        )(layers)

        _, h, w, c = projection.shape
        projected_patches = tf.reshape(projection, (-1, h*w, c))
        encoded_patches = self.PatchEmbedding(projected_patches)

        print('output embd patch', encoded_patches.shape)
        layers = self.SwinTransformer(encoded_patches)
        layers = self.SwinTransformer(layers)
        layers = self.PatchMerging(layers)
        print('output merging', layers.shape)

        representation = tf.keras.layers.GlobalAvgPool1D()(layers)
        predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(representation)
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
