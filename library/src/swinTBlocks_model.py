import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from functions.main import Main


def __sreLu (input):
    return tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(input)

def __sConv(input,parameters,size,nstrides):
    return tf.keras.layers.Conv2D(parameters, (size,size), strides=(nstrides,nstrides),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(input)

def __sBN (input):
    return tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(input)

def Block_1 (input, parameter):
        output = __sConv(input, parameter, 3, 1)
        output = __sBN(output)
        output = __sreLu(output)
        return output
    
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

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self,
                window_size,
                projection_dim,
                num_heads,
                qkv_bias,
                dropout_rate,
                **kwargs
                ):
        super(WindowAttention, self).__init__(**kwargs)
        self.window_size = window_size
        self.PROJECTION_DIM = projection_dim
        self.NUM_HEADS = num_heads
        self.QKV_BIAS = qkv_bias
        self.DROPOUT_RATE = dropout_rate

    def build(self, input_shape):
        self.dim = self.PROJECTION_DIM
        self.scale = (self.dim // self.NUM_HEADS) ** -0.5
        self.qkv = tf.keras.layers.Dense(self.dim * 3, use_bias=self.QKV_BIAS)
        self.dropout = tf.keras.layers.Dropout(self.DROPOUT_RATE)
        self.proj = tf.keras.layers.Dense(self.dim)

        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.NUM_HEADS),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="relative_position_bias_table",
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        relative_coords = tf.cast(relative_coords, dtype=tf.int32)
        relative_coords += tf.convert_to_tensor([self.window_size[0] - 1, self.window_size[1] - 1], dtype=tf.int32)
        relative_coords = relative_coords[:, :, 0] * (2 * self.window_size[1] - 1) + relative_coords[:, :, 1]


        self.relative_position_index = tf.Variable(
            initial_value=tf.reshape(relative_coords, [-1]),
            trainable=False,
            name="relative_position_index",
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.NUM_HEADS
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.NUM_HEADS, head_dim))
        x_qkv = tf.transpose(x_qkv, (2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, (0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            relative_position_index_flat,
            axis=0
        )

        relative_position_bias = tf.reshape(
            relative_position_bias,
            (num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.shape[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0),
                "float32"
            )
            attn = tf.reshape(attn, (-1, nW, self.NUM_HEADS, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.NUM_HEADS, size, size))
            attn = tf.keras.activations.softmax(attn, axis=-1)
        else:
            attn = tf.keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, (0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, (-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv

    def get_config(self):
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'PROJECTION_DIM': self.PROJECTION_DIM,
            'NUM_HEADS': self.NUM_HEADS,
            'QKV_BIAS': self.QKV_BIAS,
            'DROPOUT_RATE': self.DROPOUT_RATE
        })
        return config

class SwinTransformer(tf.keras.layers.Layer):

    def __init__(self,
                IMAGE_SIZE = 256,
                PROJECTION_DIM = 128,
                WINDOW_SIZE = 4,
                SHIFT_SIZE = 1,
                PATCH_SIZE = 4,
                LAYER_NORM_EPS = 1e-5,
                DROPOUT_RATE = 0.03,
                NUM_HEADS = 1,
                NUM_MLP = 128,
                **kwargs
                 ):

        super(SwinTransformer, self).__init__(**kwargs)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.PROJECTION_DIM = PROJECTION_DIM
        self.WINDOW_SIZE = WINDOW_SIZE
        self.SHIFT_SIZE = SHIFT_SIZE
        self.PATCH_SIZE = PATCH_SIZE
        self.LAYER_NORM_EPS = LAYER_NORM_EPS
        self.DROPOUT_RATE = DROPOUT_RATE
        self.NUM_HEADS = NUM_HEADS
        self.NUM_MLP = NUM_MLP
        self.QKV_BIAS = True
        self.NUM_PATCHES = self.IMAGE_SIZE // self.PATCH_SIZE


    def build(self, input_shape):
        self.dim = self.PROJECTION_DIM
        self.tuple_window = (self.WINDOW_SIZE, self.WINDOW_SIZE)

        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.LAYER_NORM_EPS)

        self.attn = WindowAttention(
            window_size = self.tuple_window,
            projection_dim = self.PROJECTION_DIM,
            num_heads = self.NUM_HEADS,
            qkv_bias = self.QKV_BIAS,
            dropout_rate = self.DROPOUT_RATE
        )
        self.drop_path = tf.keras.layers.Dropout(self.DROPOUT_RATE)

        if self.SHIFT_SIZE == 0:
            self.attn_mask = None
        else:
            height, width = (self.NUM_PATCHES, self.NUM_PATCHES)
            h_slices = (
                slice(0, -self.WINDOW_SIZE),
                slice(-self.WINDOW_SIZE, -self.SHIFT_SIZE),
                slice(-self.SHIFT_SIZE, None)
            )
            w_slices = (
                slice(0, -self.WINDOW_SIZE),
                slice(-self.WINDOW_SIZE, -self.SHIFT_SIZE),
                slice(-self.SHIFT_SIZE, None)
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)
            # mask array to windows
            mask_windows = self.window_partition(mask_array, self.WINDOW_SIZE)
            mask_windows = tf.reshape(
                mask_windows, [-1, self.WINDOW_SIZE * self.WINDOW_SIZE]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.cast(attn_mask, tf.float32)

            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                dtype=tf.float32,
                trainable=False
            )

        self.mlp_layers = [
            tf.keras.layers.Dense(self.NUM_MLP, activation=tf.keras.activations.gelu),
            tf.keras.layers.Dense(self.PROJECTION_DIM)
        ]

    def window_partition(self, x, window_size):
        _, height, width, channels = x.shape
        patch_num_y = height // window_size
        patch_num_x = width // window_size
        x = tf.reshape(
            x,
            (
                -1,
                patch_num_y,
                window_size,
                patch_num_x,
                window_size,
                channels
            )
        )
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        windows = tf.reshape(x, (-1, window_size, window_size, channels))
        return windows

    def window_reverse(self, windows, window_size, height, width, channels):
        patch_num_y = height // window_size
        patch_num_x = width // window_size
        x = tf.reshape(
            windows,
            (
                -1,
                patch_num_y,
                patch_num_x,
                window_size,
                window_size,
                channels
            )
        )
        x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (-1, height, width, channels))
        return x

    def mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.drop_path(x)
        return x

    def call(self, x, training=False):

        height, width = self.NUM_PATCHES, self.NUM_PATCHES
        _, _, channels = x.shape
        x_skip = x
        x = self.norm(x)
        x = tf.reshape(x, (-1, height, width, channels))
        if self.SHIFT_SIZE > 0:
            shifted_x = tf.roll(
                x, shift=[-self.SHIFT_SIZE, -self.SHIFT_SIZE], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.WINDOW_SIZE)
        x_windows = tf.reshape(
            x_windows, (-1, self.WINDOW_SIZE * self.WINDOW_SIZE, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.WINDOW_SIZE, self.WINDOW_SIZE, channels)
        )
        shifted_x = self.window_reverse(
            attn_windows, self.WINDOW_SIZE, height, width, channels
        )
        if self.SHIFT_SIZE > 0:
            x = tf.roll(
                shifted_x, shift=[self.SHIFT_SIZE, self.SHIFT_SIZE], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x
        x_skip = x
        x = self.norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "IMAGE_SIZE": self.IMAGE_SIZE,
            "PROJECTION_DIM": self.PROJECTION_DIM,
            "QKV_BIAS": self.QKV_BIAS,
            "WINDOW_SIZE": self.WINDOW_SIZE,
            "SHIFT_SIZE": self.SHIFT_SIZE,
            "PATCH_SIZE": self.PATCH_SIZE,
            "LAYER_NORM_EPS": self.LAYER_NORM_EPS,
            "DROPOUT_RATE": self.DROPOUT_RATE,
            "NUM_HEADS": self.NUM_HEADS,
            "NUM_MLP": self.NUM_MLP,
            "NUM_PATCHES": self.NUM_PATCHES
        })
        return config

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                IMAGE_SIZE,
                PATCH_SIZE,
                PROJECTION_DIM,
                **kwargs
                ):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.PATCH_SIZE = PATCH_SIZE
        self.PROJECTION_DIM = PROJECTION_DIM
        self.NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE)**2

        pos_embed = tf.keras.layers.Embedding(input_dim=self.NUM_PATCHES, output_dim=self.PROJECTION_DIM, name="patches_embedding")
        self.pos_embed = pos_embed

    def call(self, projected_patches):
        pos = tf.range(start=0, limit=self.NUM_PATCHES, delta=1)
        return projected_patches + self.pos_embed(pos)

    def get_config(self):
        config = super().get_config()
        config.update({
            "IMAGE_SIZE": self.IMAGE_SIZE,
            "PATCH_SIZE": self.PATCH_SIZE,
            "PROJECTION_DIM": self.PROJECTION_DIM
        })
        return config

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self,
                IMAGE_SIZE=256,
                PATCH_SIZE=4,
                PROJECTION_DIM=128,
                **kwargs
                ):
        super(PatchMerging, self).__init__(**kwargs)
        self.PROJECTION_DIM = PROJECTION_DIM
        self.IMAGE_SIZE = IMAGE_SIZE
        self.PATCH_SIZE = PATCH_SIZE
        self.NUM_PATCHES = self.IMAGE_SIZE//self.PATCH_SIZE
        self.num_patch = (self.NUM_PATCHES, self.NUM_PATCHES)
        self.embed_dim = self.PROJECTION_DIM
        self.linear_trans = tf.keras.layers.Dense(2 * self.embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.shape
        x = tf.reshape(x, (-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, (-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "PROJECTION_DIM": self.PROJECTION_DIM,
            "IMAGE_SIZE": self.IMAGE_SIZE,
            "PATCH_SIZE": self.PATCH_SIZE
        })
        return config

class SwinTBlock(tf.keras.layers.Layer):
    def __init__(self,
                IMAGE_SIZE=256,
                PATCH_SIZE=4,
                PROJECTION_DIM=96,
                depth=2,
                NUM_HEADS=3,
                NUM_MLP=64,
                WINDOW_SIZE=7,
                DROPOUT_RATE=0.1,
                LAYER_NORM_EPS=1e-5,
                **kwargs):

        super(SwinTBlock, self).__init__(**kwargs)
        self.IMAGE_SIZE = IMAGE_SIZE
        self.depth = depth
        self.PATCH_SIZE = PATCH_SIZE
        self.PROJECTION_DIM = PROJECTION_DIM
        self.NUM_HEADS = NUM_HEADS
        self.NUM_MLP = NUM_MLP
        self.WINDOW_SIZE = WINDOW_SIZE
        self.DROPOUT_RATE = DROPOUT_RATE
        self.LAYER_NORM_EPS = LAYER_NORM_EPS

        self.drop_path = tf.keras.layers.Dropout(self.DROPOUT_RATE)

        # Blocks
        self.layers = []
        for i in range(self.depth):
            self.layers.append(
                SwinTransformer(
                IMAGE_SIZE = self.IMAGE_SIZE,
                PATCH_SIZE = self.PATCH_SIZE,
                PROJECTION_DIM = self.PROJECTION_DIM,
                NUM_HEADS = self.NUM_HEADS,
                NUM_MLP = self.NUM_MLP,
                WINDOW_SIZE = self.WINDOW_SIZE,
                DROPOUT_RATE = self.DROPOUT_RATE,
                LAYER_NORM_EPS = self.LAYER_NORM_EPS,
                SHIFT_SIZE = 0 if i % 2 == 0 else self.WINDOW_SIZE // 2
                )
            )

        self.built = True

    def call(self, inputs):
        x = self.drop_path(inputs)
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super(SwinTBlock, self).get_config()
        config.update({
            "IMAGE_SIZE": self.IMAGE_SIZE,
            "depth": self.depth,
            "PATCH_SIZE": self.PATCH_SIZE,
            "PROJECTION_DIM": self.PROJECTION_DIM,
            "NUM_HEADS": self.NUM_HEADS,
            "NUM_MLP": self.NUM_MLP,
            "WINDOW_SIZE": self.WINDOW_SIZE,
            "DROPOUT_RATE": self.DROPOUT_RATE,
            "LAYER_NORM_EPS": self.LAYER_NORM_EPS
        })
        return config

    def compute_output_shape(self, input_shape):
        batch_size, num_patches, embed_dim = input_shape
        # Ajusta según tu cálculo de ventanas
        return (batch_size, num_patches, embed_dim)

class ReshapeLayer(tf.keras.layers.Layer):
  def __init__(self, target_shape, **kwargs):
      super(ReshapeLayer, self).__init__(**kwargs)
      self.target_shape = target_shape

  def call(self, inputs):
      return tf.reshape(inputs, self.target_shape)
  
  def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({
            "target_shape": self.target_shape
        })
        return config

class CVSTB(keras.layers.Layer):
    def __init__(self, inputs, srm_weights, biasSRM, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVSTB, self).__init__()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, srm_weights, biasSRM)
    
    def build_model(self, input_shape, srm_weights, biasSRM):
        tf.keras.backend.clear_session()

        #Preprocessing
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        # L1-L2
        layers = Block_1(layers1, 64)
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
        
        layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers5)
        layers = tf.keras.layers.GlobalAvgPool1D()(layers)

        layers = fully_connected(layers)

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

class Training(Main):
    def __init__(self, epochs, batch_size, dataset):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATASET = dataset

    def train_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = CVSTB(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        #prueba 1 - lr_schedule 5e-4
        #prueba 2 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 6
        #prueba 3 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 2 - step4_depth 6
        #prueba 4 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 6 - step4_depth 11
        #prueba 5 - learning rate 5e-3 - block3 - step1_depth 3 - step2_depth 3 - step3_depth 2 - step4_depth 2
        #prueba 6 - learning rate 5e-3 - step1_depth 2 - step2_depth 2 - step3_depth 3 - step4_depth 3
        #prueba 7 - learning rate 5e-3 - block1 - step1_depth 3 - step2_depth 2 - step3_depth 2 - step4_depth 2
        #prueba 8 - learning rate 5e-3 - block1 - step1_depth 4 - step2_depth 2 - step3_depth 2 - step4_depth 1
        #prueba 9 - learning rate 5e-3 - step1_depth 4 - step2_depth 2 - step3_depth 2 - step4_depth 1 - PPMconcat - FC

        self.plot_model_summary(architecture.model, 'swinTBlocks_summary')

        X_train = np.load('../../database/BOSS/'+self.DATASET+'/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../../database/BOSS/'+self.DATASET+'/y_train.npy') # (12000, 2)
        X_valid = np.load('../../database/BOSS/'+self.DATASET+'/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../../database/BOSS/'+self.DATASET+'/y_valid.npy') # (4000, 2)
        X_test = np.load('../../database/BOSS/'+self.DATASET+'/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../../database/BOSS/'+self.DATASET+'/y_test.npy')   # (4000, 2)

        base_name="04-"+self.DATASET
        name="Model_"+'SWINTBlocks_prueba10'+"_"+base_name
        _, history  = self.fit(
            architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, dataset=self.DATASET,
            custom_layers={
                'Tanh3':Tanh3, 
                'SEBlock':SEBlock,
                'ReshapeLayer':ReshapeLayer,
                'PatchEmbedding':PatchEmbedding,
                'PatchMerging':PatchMerging,
                'SwinTBlock':SwinTBlock
            }
        )        

train = Training(epochs=8, batch_size=3, dataset='S-UNIWARD')
train.train_model()
