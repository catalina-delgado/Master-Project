from src.imports import tf, np

class DWConv(tf.keras.layers.Layer):
    def __init__(self, dim=512, **kawargs):
        super(DWConv, self).__init__(**kawargs)
        self.dwconv = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
            groups=dim
        )
    
    def call(self, inputs, H, W):
        _, N, C = inputs.shape
        x = tf.reshape(inputs, [-1, H, W, C])
        x = self.dwconv(x)
        x = tf.reshape(x, [-1, -1, C])
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dwconv': self.dwconv
        })
        return config


class MixFFN(tf.keras.layers.Layer):
    def __init__(self, units, hidden_units, **kwargs):
        super(MixFFN, self).__init__(**kwargs)
        self.mlp_layers = [
            tf.keras.layers.Dense(hidden_units),
            tf.keras.layers.Dense(units)
        ]
        self.drop_path = tf.keras.layers.Dropout(0.1)
        self.dwconv = DWConv(units)
        self.gelu = tf.keras.activations.gelu

    def call(self, inputs, H, W):
        x = self.mlp_layers[0](inputs)
        x = self.dwconv(x, H, W)
        x = self.gelu(x)
        x = self.drop_path(x)
        x = self.mlp_layers[1](x)
        x = self.drop_path(x)
       
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_layers': self.mlp_layers,
            'drop_path': self.drop_path,
            'dwconv': self.dwconv,
            'gelu': self.gelu
        })
        return config

class Attention(tf.keras.layers.Layer):
    def __init__(self, 
                 projection_dim, 
                 num_heads, 
                 qkv_bias=True, 
                 dropout_rate=0.1, 
                 sr_ratio=1, 
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.sr_ratio = sr_ratio
    
    def build(self, input_shape):
        self.q = tf.keras.layers.Dense(self.projection_dim, use_bias=self.qkv_bias)
        self.kv = tf.keras.layers.Dense(self.projection_dim * 2, use_bias=self.qkv_bias)
        self.scale = (self.projection_dim // self.num_heads) ** -0.5
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

        if self.sr_ratio > 1:
            self.sr = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    self.projection_dim, 
                    kernel_size=(self.sr_ratio, self.sr_ratio), 
                    strides=(self.sr_ratio, self.sr_ratio), 
                    padding="valid"
                ),
                tf.keras.layers.LayerNormalization(epsilon=1e-5)
            ])
        
        self.attn_drop = tf.keras.layers.Dropout(self.dropout_rate)
        self.proj = tf.keras.layers.Dense(self.projection_dim)
        self.proj_drop = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, x, H, W):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Generar consultas
        q = self.q(x)  # [B, N, C]
        q = tf.reshape(q, (B, N, self.num_heads, C // self.num_heads))  # [B, N, num_heads, head_dim]
        q = tf.transpose(q, perm=[0, 2, 1, 3])  # [B, num_heads, N, head_dim]

        # Generar claves y valores
        if self.sr_ratio > 1:
            x_ = tf.reshape(x, (B, H, W, C))  # [B, H, W, C]
            x_ = self.sr(x_)  # Aplicar reducción espacial
            x_ = tf.reshape(x_, (B, -1, self.projection_dim))  # [B, N', C]
            kv = self.kv(x_)  # [B, N', 2*C]
        else:
            kv = self.kv(x)  # [B, N, 2*C]

        kv = tf.reshape(kv, (B, -1, 2, self.num_heads, C // self.num_heads))  # [B, N, 2, num_heads, head_dim]
        kv = tf.transpose(kv, perm=[2, 0, 3, 1, 4])  # [2, B, num_heads, N, head_dim]
        k, v = kv[0], kv[1]  # Separar claves y valores

        # Atención
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))  # [B, num_heads, N, N]
        attn = attn * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # Aplicar atención sobre los valores
        x = tf.matmul(attn, v)  # [B, num_heads, N, head_dim]
        x = tf.transpose(x, perm=[0, 2, 1, 3])  # [B, N, num_heads, head_dim]
        x = tf.reshape(x, (B, N, C))  # [B, N, C]

        # Proyección final
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'projection_dim': self.projection_dim,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'sr_ratio': self.sr_ratio,
        })
        return config
   

class Block(tf.keras.layers.Layer):
    def __init__(self, 
                dim, 
                num_heads,
                sr_ratio,
                qkv_bias,
                dropout_rate,
                mlp_ratio,
                **kwargs):
        super(Block, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.mlp_ratio = mlp_ratio
        self.sr_ratio = sr_ratio

    def build(self, input_shape):
        self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.attn = Attention(self.dim,
                            self.num_heads,
                            self.qkv_bias,
                            self.dropout_rate,
                            self.sr_ratio)
        self.drop_path = tf.keras.layers.Dropout(self.dropout_rate)
        self.mlp = MixFFN(self.dim, self.dim*self.mlp_ratio)

    def call(self, inputs, H, W):
        x = inputs + self.drop_path(self.attn(self.norm(inputs), H, W))
        x = x + self.drop_path(self.mlp(self.norm(x), H, W))
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'dropout_rate': self.dropout_rate,
            'mlp_ratio': self.mlp_ratio,
            'sr_ratio': self.sr_ratio,
        })
        return config


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=224, patch_size=7, dim=768, stride=1):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.H = self.img_size[0] // stride
        self.W = self.img_size[1] // stride
        self.num_patches = self.H * self.W

        self.proj = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=self.patch_size,
            strides=stride,
            padding='same',  
            use_bias=True
        )

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        
        x = tf.reshape(x, (B, H * W, C)) 
        x = self.norm(x)

        return x, H, W
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'H': self.H,
            'W': self.W,
            'num_patches': self.num_patches,
            'proj': self.proj,
            'norm': self.norm
        })
        return config


class ResFormer(tf.keras.layers.Layer):
  def __init__(self, 
               IMG_SIZE, 
               PATCH_SIZE,
               PROJECTION_DIM, 
               NUM_HEADS, 
               DEPTH,
               SR_RATIO,
               **kwargs):
      super(ResFormer, self).__init__(**kwargs) 
      self.blocks = [Block(
                        PROJECTION_DIM,
                        NUM_HEADS,
                        SR_RATIO,
                        qkv_bias=True,
                        dropout_rate=0.1,
                        mlp_ratio=2) 
                        for i in range(DEPTH)]
      self.norm = tf.keras.layers.BatchNormalization(epsilon=1e-5)
      self.patchembed = PatchEmbed(img_size=IMG_SIZE//PATCH_SIZE, patch_size=PATCH_SIZE, dim=PROJECTION_DIM, stride=2)

  def call(self, inputs):
      x, H, W = self.patchembed(inputs)
      for _, blk in enumerate(self.blocks):
          x = blk(x, H, W)
      x = self.norm(x)
      B = tf.shape(x)[0]
      C = x.shape[-1]
      x = tf.reshape(x, (B, H, W, C))
      return x
  
  def get_config(self):
        config = super().get_config()
        config.update({
            'blocks': self.blocks,
            'norm': self.norm,
            'patchembed': self.patchembed
        })
        return config


class PPMConcat(tf.keras.layers.Layer):
    """
    Pyramid Pooling Module

    """

    def __init__(self, pool_scales=(1, 2, 4, 8)):
        super(PPMConcat, self).__init__()
        self.adaptive_pools = [
            tf.keras.layers.AveragePooling2D(pool_size=(scale, scale), strides=(scale, scale), padding='valid')
            for scale in pool_scales
        ]

    def call(self, inputs):
        ppm_outs = []
        for tensor in inputs:
            tensor_outs = []
            
            for pool in self.adaptive_pools:
                # Aplicar el pooling
                pooled = pool(tensor)
                B = tf.shape(pooled)[0] 
                H = tf.shape(pooled)[1] 
                W = tf.shape(pooled)[2] 
                C = tf.shape(pooled)[3]
                # Aplanar las dimensiones espaciales
                flattened = tf.reshape(pooled, [B, W*H, C])
                tensor_outs.append(flattened)
            # Concatenar las características de todas las escalas
            concatenated = tf.concat(tensor_outs, axis=1)
            B = tf.shape(concatenated)[0] 
            N = tf.shape(concatenated)[1] 
            C = tf.shape(concatenated)[2]
            # Aplanar las caracteristicas
            flattened = tf.reshape(concatenated, [B, N*C])
            ppm_outs.append(flattened)
        # Concatenar las características de todos los tensores
        final_out = tf.concat(ppm_outs, axis=-1)
        return final_out
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'adaptive_pools': self.adaptive_pools
        })
        return config


class FC(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(FC, self).__init__(**kwargs)
        self.layers_list = []
        
        for unit in units[:-1]:
            self.layers_list.append(tf.keras.layers.Dense(
                units=unit, 
                kernel_initializer='glorot_normal', 
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                bias_regularizer=tf.keras.regularizers.l2(0.0001)
            ))
            self.layers_list.append(tf.keras.layers.ReLU(negative_slope=0.1, threshold=0))
        

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'layers_list': self.layers_list
        })
        return config
