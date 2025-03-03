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

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, 
                 window_size, 
                 projection_dim,
                 num_heads,
                 qkv_bias,
                 dropout_rate, 
                 **kwargs):
        super(WindowAttention, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.window_size = window_size

    def build(self, input_shape):
        self.dim = self.projection_dim
        self.num_heads = self.num_heads
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = tf.keras.layers.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.proj = tf.keras.layers.Dense(self.dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )

        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
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
        relative_coords += tf.convert_to_tensor([self.window_size[0] - 1, self.window_size[1] - 1])
        relative_coords = relative_coords[:, :, 0] * (2 * self.window_size[1] - 1) + relative_coords[:, :, 1]
        
        self.relative_position_index = tf.Variable(
            initial_value=tf.reshape(relative_coords, [-1]),
            trainable=False,
            name="relative_position_index"
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, (-1, size, 3, self.num_heads, head_dim))
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
            attn = tf.reshape(attn, (-1, nW, self.num_heads, size, size)) + mask_float
            attn = tf.reshape(attn, (-1, self.num_heads, size, size))
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
            'projection_dim': self.projection_dim,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate
        })
        return config
     
class SwinTransformer(tf.keras.layers.Layer):
    
    def __init__(self, 
                image_size = 16,
                projection_dim = 128,
                qkv_bias = True,
                window_size = 8,
                shift_size = 2,
                patch_size = 2,
                layer_norm_eps = 1e-5,
                dropout_rate = 0.1,
                num_heads = 4,
                num_mlp = 512,
                **kwargs
                ):
        super(SwinTransformer, self).__init__(**kwargs)
        self.image_size = image_size
        self.projection_dim = projection_dim
        self.qkv_bias = qkv_bias
        self.window_size = window_size
        self.shift_size = shift_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.num_mlp = num_mlp
        self.num_patches = self.image_size // self.patch_size

    def build(self, input_shape):
        self.dim = self.projection_dim
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps)
        self.attn = WindowAttention(
            window_size = (self.window_size, self.window_size),
            projection_dim = self.projection_dim,
            num_heads = self.num_heads,
            qkv_bias = self.qkv_bias,
            dropout_rate = self.dropout_rate,
            name="attention"
        )
        self.drop_path = tf.keras.layers.Dropout(self.dropout_rate)

        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = (self.num_patches, self.num_patches)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.window_size),
                slice(-self.shift_size, None)
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = self.window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, [-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                dtype=attn_mask.dtype,
                trainable=False,
                name="attn_mask"
            )
            self.mlp_layers = [
                tf.keras.layers.Dense(self.num_mlp, activation=tf.keras.activations.gelu),
                tf.keras.layers.Dense(self.projection_dim)
            ]

    #@staticmethod
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

    #@staticmethod
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
    
    #@staticmethod
    def mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.drop_path(x)
        return x

    def call(self, x, training=False):

        height, width = self.num_patches, self.num_patches
        _, _, channels = x.shape
        x_skip = x
        x = self.norm(x)
        x = tf.reshape(x, (-1, height, width, channels))
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.window_size, self.window_size, channels)
        )
        shifted_x = self.window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
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
            'image_size': self.image_size,
            'projection_dim': self.projection_dim,
            'qkv_bias': self.qkv_bias,
            'window_size': self.window_size,
            'shift_size': self.shift_size,
            'patch_size': self.patch_size,
            'layer_norm_eps': self.layer_norm_eps,
            'dropout_rate': self.dropout_rate,
            'num_heads': self.num_heads,
            'num_mlp': self.num_mlp
        })
        return config
    
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, projection_dim = 128, num_patches = 16 // 2, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_patches = num_patches
        self.num_patch = self.num_patches**2
        
        pos_embed = tf.keras.layers.Embedding(input_dim=self.num_patch, output_dim=self.projection_dim, name="patches_embedding")
        self.pos_embed = pos_embed
        
    def call(self, projected_patches):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return projected_patches + self.pos_embed(pos)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'num_patches': self.num_patches
        })
        return config

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patches = 16 // 2, projection_dim = 128, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim= projection_dim
        self.num_patch = (self.num_patches, self.num_patches)
        self.embed_dim = self.projection_dim
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
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim
        })
        return config

class CVST(keras.layers.Layer):
    def __init__(self, inputs, weights, bias, learning_rate=None, lr_schedule=0.001, compile=True):
        super(CVST, self).__init__()
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.compile = compile
        self.model = self.build_model(inputs, weights, bias)
   
    def build_model(self, input_shape, srm_weights,biasSRM):
        tf.keras.backend.clear_session()

        #Preprocessing
        layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(input_shape)
        layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(input_shape)

        layers = tf.keras.layers.add([layers_ty, layers_tn])
        layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

        # Block A
        layers = Block_1(layers1, 64)
        layers = Block_1(layers, 64)

        # Block B
        for i in range(5):
            layers = Block_2(layers, 64)

        # Block C
        for i in [64, 64, 128, 256]:
            layers = Block_3(layers, i)

        # Swin Transformer
        layers = Block_4(layers, 512)
        print('output last layer before transformer', layers.shape)

        layers = tf.keras.layers.LayerNormalization(epsilon=1e-5)(layers)

        projection = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )(layers)

        _, h, w, c = projection.shape
        projected_patches = tf.reshape(projection, (-1, h*w, c))
        encoded_patches = PatchEmbedding()(projected_patches)

        print('output embd patch', encoded_patches.shape)
        layers = SwinTransformer()(encoded_patches)
        layers = SwinTransformer()(layers)
        layers = SwinTransformer()(layers)
        layers = SwinTransformer()(layers)
        layers = SwinTransformer()(layers)
        layers = SwinTransformer()(layers)
        layers = SwinTransformer()(layers)
        layers = PatchMerging()(layers)
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

class TrainingCVST(Main):
    def __init__(self, epochs, batch_size, dataset):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATASET = dataset

    def train_cvst_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = CVST(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        #prueba 1 - 4 block_2 - windows size 2, patch size 2, dropout rate 0.03, mlp 128 - shift 1, num_heads 8
        #prueba 2 - 5 block_2 - windows size 2, patch size 2, dropout rate 0.03, mlp 128 - shift 1, num_heads 8
        #prueba 3 - 5 block 2 - windows size 2, patch size 2, dropout rate 0.03, mlp 512 - shift 1, num_heads 8
        #prueba 4 - 5 block 2 - windows size 4, patch size 2, dropout rate 0.1, mlp 512 - shift 2, num_heads 8
        #prueba 5 - 5 block 2 - windows size 4, patch size 2, dropout rate 0.1, mlp 512 - shift 1, num_heads 4

        self.plot_model_summary(architecture.model, 'swinTransformer_summary')

        X_train = np.load('../../database/BOSS/'+self.DATASET+'/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../../database/BOSS/'+self.DATASET+'/y_train.npy') # (12000, 2)
        X_valid = np.load('../../database/BOSS/'+self.DATASET+'/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../../database/BOSS/'+self.DATASET+'/y_valid.npy') # (4000, 2)
        X_test = np.load('../../database/BOSS/'+self.DATASET+'/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../../database/BOSS/'+self.DATASET+'/y_test.npy')   # (4000, 2)

        base_name="04-"+self.DATASET
        name="Model_"+'swinTransformer_prueba6'+"_"+base_name
        _, history  = self.fit(
            architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, 
            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, dataset=self.DATASET,
            custom_layers={
                'Tanh3':Tanh3,  
                'PatchEmbedding':PatchEmbedding, 
                'SwinTransformer':SwinTransformer,
                'PatchMerging':PatchMerging
                }
        )

train = TrainingCVST(epochs=8, batch_size=3, dataset='S-UNIWARD')
train.train_cvst_model()
