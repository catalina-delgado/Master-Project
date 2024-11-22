from src.imports import tf, np

class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, window_size, hyperparams):
        super(WindowAttention, self).__init__()
        self.hyperparams = hyperparams
        self.window_size = window_size

    def build(self, input_shape):
        self.dim = self.hyperparams['PROJECTION_DIM']
        self.num_heads = self.hyperparams['NUM_HEADS']
        self.scale = (self.dim // self.num_heads) ** -0.5
        self.qkv = tf.keras.layers.Dense(self.dim * 3, use_bias=self.hyperparams['QKV_BIAS'])
        self.dropout = tf.keras.layers.Dropout(self.hyperparams['DROPOUT_RATE'])
        self.proj = tf.keras.layers.Dense(self.dim)

        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )

        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
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
            "hyperparams": self.hyperparams,
            'window_size': self.window_size
        })
        return config
      

class SwinTransformer(tf.keras.layers.Layer):
    
    def __init__(self, hyperparams):
        super(SwinTransformer, self).__init__()
        self.hyperparams = hyperparams

    def build(self, input_shape):
        self.dim = self.hyperparams['PROJECTION_DIM'] 
        self.norm = tf.keras.layers.LayerNormalization(epsilon=self.hyperparams['LAYER_NORM_EPS'])
        self.attn = WindowAttention(
            window_size = (self.hyperparams['WINDOW_SIZE'], self.hyperparams['WINDOW_SIZE']),
            hyperparams = self.hyperparams
        )
        self.drop_path = tf.keras.layers.Dropout(self.hyperparams['DROPOUT_RATE'])

        if self.hyperparams['SHIFT_SIZE'] == 0:
            self.attn_mask = None
        else:
            height, width = (self.hyperparams['NUM_PATCHES'], self.hyperparams['NUM_PATCHES'])
            h_slices = (
                slice(0, -self.hyperparams['WINDOW_SIZE']),
                slice(-self.hyperparams['WINDOW_SIZE'], -self.hyperparams['SHIFT_SIZE']),
                slice(-self.hyperparams['SHIFT_SIZE'], None)
            )
            w_slices = (
                slice(0, -self.hyperparams['WINDOW_SIZE']),
                slice(-self.hyperparams['WINDOW_SIZE'], -self.hyperparams['SHIFT_SIZE']),
                slice(-self.hyperparams['SHIFT_SIZE'], None)
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = self.window_partition(mask_array, self.hyperparams['WINDOW_SIZE'])
            mask_windows = tf.reshape(
                mask_windows, [-1, self.hyperparams['WINDOW_SIZE'] * self.hyperparams['WINDOW_SIZE']]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask,
                dtype=attn_mask.dtype,
                trainable=False
            )
            self.mlp_layers = [
                tf.keras.layers.Dense(self.hyperparams['NUM_MLP'], activation=tf.keras.activations.gelu),
                tf.keras.layers.Dense(self.hyperparams['PROJECTION_DIM'])
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

        height, width = self.hyperparams['NUM_PATCHES'], self.hyperparams['NUM_PATCHES']
        _, _, channels = x.shape
        x_skip = x
        x = self.norm(x)
        x = tf.reshape(x, (-1, height, width, channels))
        if self.hyperparams['SHIFT_SIZE'] > 0:
            shifted_x = tf.roll(
                x, shift=[-self.hyperparams['SHIFT_SIZE'], -self.hyperparams['SHIFT_SIZE']], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x, self.hyperparams['WINDOW_SIZE'])
        x_windows = tf.reshape(
            x_windows, (-1, self.hyperparams['WINDOW_SIZE'] * self.hyperparams['WINDOW_SIZE'], channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows,
            (-1, self.hyperparams['WINDOW_SIZE'], self.hyperparams['WINDOW_SIZE'], channels)
        )
        shifted_x = self.window_reverse(
            attn_windows, self.hyperparams['WINDOW_SIZE'], height, width, channels
        )
        if self.hyperparams['SHIFT_SIZE'] > 0:
            x = tf.roll(
                shifted_x, shift=[self.hyperparams['SHIFT_SIZE'], self.hyperparams['SHIFT_SIZE']], axis=[1, 2]
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
            "hyperparams": self.hyperparams
        })
        return config
    

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, hyperparams):
        super(PatchEmbedding, self).__init__()
        self.num_patch = hyperparams['NUM_PATCHES']**2
        self.pos_embed = tf.keras.layers.Embedding(input_dim=self.num_patch, output_dim=hyperparams['PROJECTION_DIM'], name="patches_embedding")

    def call(self, projected_patches):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return projected_patches + self.pos_embed(pos)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patch": self.num_patch,
            'pos_embed': self.pos_embed
        })
        return config


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, hyperparams):
        super(PatchMerging, self).__init__()
        self.num_patch = (hyperparams['NUM_PATCHES'], hyperparams['NUM_PATCHES'])
        self.embed_dim = hyperparams['PROJECTION_DIM']
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
            "num_patch": self.num_patch,
            'embed_dim': self.embed_dim,
            'linear_trans': self.linear_trans
        })
        return config


class SwinTBlock(tf.keras.layers.Layer):
    def __init__(self, 
                hyperparams, 
                img_size=256, 
                patch_size=4, 
                embed_dim=96, 
                depth=2, 
                num_heads=3,
                num_mlp=64,
                window_size=7,
                **kwargs):
        
        super().__init__()
        self.hyperparams = hyperparams
        self.image_size = img_size
        self.depth = depth
        self.patch_size = patch_size
        self.num_patches = img_size//self.patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_mlp = num_mlp
        self.windows_size = window_size

    def build(self, input_shape):
        self.hyperparams['IMAGE_SIZE'] = self.image_size
        self.hyperparams['PROJECTION_DIM'] = self.embed_dim
        self.hyperparams['NUM_MLP'] = self.num_mlp
        self.hyperparams['NUM_PATCHES'] = self.num_patches
        self.hyperparams['PATCH_SIZE'] = self.patch_size
        self.hyperparams['NUM_HEADS'] = self.num_heads
        self.hyperparams['WINDOWS_SIZE'] = self.windows_size

        self.drop_path = tf.keras.layers.Dropout(self.hyperparams['DROPOUT_RATE'])
        
        # Blocks
        self.layers = []
        for i in range(self.depth):
            if i%2 ==0:
                self.hyperparams['SHIFT_SIZE']=0
            else:
                self.hyperparams['WINDOW_SIZE']//2
            
            layer = SwinTransformer(self.hyperparams)
            self.layers.append(layer)

    def call(self, inputs):
        x = self.drop_path(inputs)
        for layer in self.layers:
            x = layer(x)
            print('layer', x)
        return x
     
