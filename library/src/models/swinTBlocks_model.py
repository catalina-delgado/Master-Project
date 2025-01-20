from src.imports import tf, K, keras
from src.layers.cnn import CNN, SEBlock
from src.layers.swint import ReshapeLayer, PatchEmbedding, PatchMerging, SwinTBlock, PPMConcat

class CVSTB(keras.layers.Layer):
    def __init__(self, inputs, srm_weights, biasSRM, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVSTB, self).__init__()
        self.CNN = CNN()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, srm_weights, biasSRM)

    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3
    
    def build_model(self, input_shape, srm_weights, biasSRM):
        tf.keras.backend.clear_session()

        #Preprocessing
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=self.__Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=self.__Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        # L1-L2
        layers = self.CNN.Block_1(layers1, 64)
        _, image_size, _, _ = layers.shape

        layers = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1, 1), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.0001), bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
        layers = SEBlock(64)(layers)
        print('output last layer before transformer', layers.shape)

        # Swin Transformer
        projection = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            bias_regularizer=tf.keras.regularizers.l2(0.0001)
        )(layers)

        _, h, w, c = projection.shape
        print('output projection', projection.shape)
        projected_patches = ReshapeLayer((-1, h * w, c))(projection)

        print('output projected_patches', projected_patches.shape)

        encoded_patches = PatchEmbedding(IMAGE_SIZE=image_size,PATCH_SIZE=2,PROJECTION_DIM=c)(projected_patches)
        print('output embd patch', encoded_patches.shape)

        layers2 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=2, PROJECTION_DIM=64, depth=3, NUM_HEADS=2, NUM_MLP=256, WINDOW_SIZE=4, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(encoded_patches)
        layers = PatchMerging(IMAGE_SIZE=image_size,PATCH_SIZE=2,PROJECTION_DIM=64)(layers2)

        layers3 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=4, PROJECTION_DIM=128, depth=2, NUM_HEADS=4, NUM_MLP=512, WINDOW_SIZE=4, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(layers)
        layers = PatchMerging(IMAGE_SIZE=image_size,PATCH_SIZE=4,PROJECTION_DIM=128)(layers3)

        layers4 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=8, PROJECTION_DIM=256, depth=2, NUM_HEADS=8, NUM_MLP=1024, WINDOW_SIZE=8, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(layers)
        layers = PatchMerging(IMAGE_SIZE=image_size, PATCH_SIZE=8, PROJECTION_DIM=256)(layers4)

        layers5 = SwinTBlock(IMAGE_SIZE=image_size, PATCH_SIZE=16, PROJECTION_DIM=512, depth=1, NUM_HEADS=16, NUM_MLP=4096, WINDOW_SIZE=8, DROPOUT_RATE=0.1, LAYER_NORM_EPS=1e-5)(layers)
        
        #layers = tf.keras.layers.LayerNormalization(epsilon=1e-5)(layers)
        #layers = tf.keras.layers.GlobalAvgPool1D()(layers)
        #representation = tf.keras.layers.Flatten()(layers)

        #layers = PPMConcat(pool_scales=(32, 64, 128))([layers2, layers3, layers4, layers5])
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers5)
        layers = tf.keras.layers.GlobalAvgPool1D()(layers)

        layers = self.CNN.fully_connected(layers)

        predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
        model = tf.keras.Model(inputs=input_shape, outputs=predictions)

        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        print("Arquitecture swin transformer created")
        return model

    def call(self, inputs, training=None):
        return self.model(inputs)
