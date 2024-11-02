from src.imports import tf
from src.layers.cnn import CNN

class Transformer(tf.keras.layers.Layer): 
    def __init__(self, hyperparams):
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams
        self.CNN = CNN()
        
        self.position_embedding_layer = tf.keras.layers.Embedding(
            input_dim=hyperparams['NUM_PATCHES_2'], 
            output_dim=hyperparams['PROJECTION_DIM_2']
        )
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=hyperparams['LAYER_NORM_EPS_2'])
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=hyperparams['LAYER_NORM_EPS_2'])
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=hyperparams['NUM_HEADS_2'], 
            key_dim=hyperparams['PROJECTION_DIM_2'], 
            dropout=0.1
        )
        self.mlp_layers = [
            tf.keras.layers.Dense(units, activation=tf.nn.gelu)
            for units in hyperparams['MLP_UNITS_2']
        ]
        self.mlp_dropout = tf.keras.layers.Dropout(0.1)

    def build(self, input_shape):
        self.conv_projection = tf.keras.layers.Conv2D(
            filters=self.hyperparams['PROJECTION_DIM_2'],
            kernel_size=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            strides=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            padding="VALID",
        )

    def position_embedding(self, projected_patches):
        positions = tf.range(start=0, limit=self.hyperparams['NUM_PATCHES_2'], delta=1)
        encoded_positions = self.position_embedding_layer(positions)
        return projected_patches + encoded_positions

    def mlp(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.mlp_dropout(x)
        return x

    def transformer_block(self, encoded_patches):
        x1 = self.layer_norm1(encoded_patches)
        attention_output = self.attention(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = self.layer_norm2(x2)
        x4 = self.mlp(x3)
        return tf.keras.layers.Add()([x4, x2])

    def call(self, inputs):
        inputs1 = self.CNN.squeeze_excitation_layer(inputs, out_dim=512, ratio=32.0, conv=False)

        projected_patches = self.conv_projection(inputs1)
        _, h, w, c = projected_patches.shape
        projected_patches = tf.reshape(projected_patches, (-1, h * w, c))
        encoded_patches = self.position_embedding(projected_patches)
        encoded_patches = self.dropout(encoded_patches)

        for _ in range(self.hyperparams['NUM_LAYERS_2']):
            encoded_patches = self.transformer_block(encoded_patches)

        return encoded_patches


#