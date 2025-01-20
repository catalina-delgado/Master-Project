from src.imports import tf, K, keras
from src.layers.cnn import CNN, SEBlock
from src.layers.residualTransformer import ResFormer, PPMConcat, FC

class ResNet(keras.layers.Layer):
    def __init__(self, inputs, srm_weights, biasSRM, learning_rate=None, lr_schedule=0.001, compile=True):
        super(ResNet, self).__init__()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.CNN = CNN()
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
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        layers = tf.keras.layers.LayerNormalization(epsilon=1e-5)(layers)
        _, image_size, _, c = layers.shape
        print('output last layer before transformer', layers.shape)

        # Transformer
        layers1 = self.CNN.Block_1(layers, 64)
        print(layers1.shape)

        layers1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1)
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers1)
        layers1 = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers1)
        print(layers1.shape)

        layers1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(2, 2),padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1)
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers1)
        print(layers1.shape)
        layers1 = SEBlock(64)(layers1)

        _, h, w, c = layers1.shape

        layers2 = ResFormer(IMG_SIZE=h, PATCH_SIZE=2, PROJECTION_DIM=c, NUM_HEADS=1, DEPTH=2, SR_RATIO=4)(layers1)
        print('layer2',layers2.shape)

        layers3 = ResFormer(IMG_SIZE=h, PATCH_SIZE=2, PROJECTION_DIM=2*c, NUM_HEADS=2, DEPTH=2, SR_RATIO=2)(layers2)
        print('layer3',layers3.shape)

        layers4 = ResFormer(IMG_SIZE=h, PATCH_SIZE=2, PROJECTION_DIM=4*c, NUM_HEADS=2, DEPTH=2, SR_RATIO=1)(layers3)
        print('layer4',layers4.shape)

        layers5 = ResFormer(IMG_SIZE=h, PATCH_SIZE=4, PROJECTION_DIM=4*c, NUM_HEADS=4, DEPTH=2, SR_RATIO=1)(layers4)
        print('layer5',layers5.shape)

        #layers = PPMConcat(pool_scales=(1, 2, 4))([layers2, layers3, layers4, layers5])
        layers = tf.keras.layers.GlobalAveragePooling2D()(layers5)
        layers = tf.keras.layers.LayerNormalization(epsilon=1e-5)(layers)

        representation = FC([64,32,16])(layers)
        predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(representation)
        model = tf.keras.Model(inputs=input_shape, outputs=predictions)

        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        print("Arquitecture Residual transformer created")
        return model

    def call(self, inputs, training=None):
        return self.model(inputs)
