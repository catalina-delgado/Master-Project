from src.imports import tf
from src.layers.cnn import CNN

class Transformer(tf.keras.layers.Layer): 
    def __init__(self, hyperparams):
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams
        self.CNN = CNN()

    def build(self, input_shape):
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.hyperparams['NUM_PATCHES_2'], 
            output_dim=self.hyperparams['PROJECTION_DIM_2']
        )
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.MultiHeadAttention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.hyperparams['NUM_HEADS_2'], 
            key_dim=self.hyperparams['PROJECTION_DIM_2'], 
            dropout=0.1
        )
        self.conv_projection = tf.keras.layers.Conv2D(
            filters=self.hyperparams['PROJECTION_DIM_2'],
            kernel_size=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            strides=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            padding="same",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=self.hyperparams['LAYER_NORM_EPS_2'])
        
        self.mlp_layers = [
            tf.keras.layers.Dense(units, activation=tf.nn.gelu) for units in self.hyperparams['MLP_UNITS_2']
        ]

    def position_embedding(self, projected_patches):
        positions = tf.range(start=0, limit=self.hyperparams['NUM_PATCHES_2'], delta=1)
        encoded_positions = self.position_embedding_layer(positions)
        return projected_patches + encoded_positions

    def mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.dropout(x)
        return x

    def transformer_block(self, encoded_patches):
        x1 = self.layer_norm(encoded_patches)
        attention_output = self.MultiHeadAttention(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        
        x3 = self.layer_norm(x2)
        x4 = self.mlp(x3)
        return tf.keras.layers.Add()([x4, x2])

    def call(self, inputs):

        projected_patches = self.conv_projection(inputs)
        _, h, w, c = projected_patches.shape
        projected_patches = tf.reshape(projected_patches, (-1, h * w, c))
        
        encoded_patches = self.position_embedding(projected_patches)
        encoded_patches = self.dropout(encoded_patches)

        for _ in range(self.hyperparams['NUM_LAYERS_2']):
            encoded_patches = self.transformer_block(encoded_patches)

        encoded_patches = tf.reshape(encoded_patches, (-1, h, w, c))  # (None, 16, 16, 128)

        return encoded_patches
