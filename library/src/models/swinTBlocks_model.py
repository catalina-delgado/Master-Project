from src.imports import tf, K, keras
from src.layers import cnn
from src.layers.swin_transformer import PatchEmbedding, SwinTransformer, PatchMerging, SwinTBlock

class CVSTB(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, hyperparams, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVSTB, self).__init__()
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
        layers = self.CNN.Block_2(layers, 64)
        layers = self.CNN.Block_2(layers, 64)

        # Block C
        layers = self.CNN.Block_3(layers, 64)
        layers = self.CNN.Block_3(layers, 128)
        layers = self.CNN.Block_3(layers, 256)
        layers = tf.keras.layers.LayerNormalization(epsilon=hyperparams['LAYER_NORM_EPS'])(layers)
        print('output last layer before transformer', layers.shape)

        # Swin Transformer
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

        layers = SwinTBlock(hyperparams, img_size=h, patch_size=4, embed_dim=c, depth=2, num_heads=3, num_mlp=64)(projected_patches)
        layers = self.PatchMerging(layers)
        _, dim, c = layers.shape
        print('output merging_1', layers.shape)

        layers = SwinTBlock(hyperparams, img_size=tf.sqrt(dim), patch_size=4, embed_dim=c, depth=2, num_heads=6, num_mlp=128)(layers)
        layers = self.PatchMerging(layers)
        _, dim, c = layers.shape
        print('output merging_2', layers.shape)

        layers = SwinTBlock(hyperparams, img_size=tf.sqrt(dim), patch_size=8, embed_dim=c, depth=6, num_heads=12, num_mlp=256)(layers)
        layers = self.PatchMerging(layers)
        _, dim, c = layers.shape
        print('output merging_3', layers.shape)

        representation = tf.keras.layers.GlobalAvgPool1D()(layers)
        predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(representation)
        model = tf.keras.Model(inputs=input_shape, outputs=predictions)
        
        if self.learning_rate is not None:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.95)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule, weight_decay=0.95)
        
        if self.compile:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        
        print("Arquitecture1 creada")
        return model
    
    def call(self, inputs, training=None):
        return self.model(inputs)
