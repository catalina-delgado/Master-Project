from src.imports import tf
from src.layers.cnn import CNN

class Transformer: 
    def __init__(self, hyperparams):
        self.CNN = CNN()
        self.hyperparams = hyperparams

    def __position_embedding(self, projected_patches):
        """
        Parameters:
        projected_patches: Tensor
        """
        # Los patches proyectados de entrada.
        num_patches=self.hyperparams['NUM_PATCHES_2']
        # El número de patches en la entrada. Debe ser un entero.
        projection_dim=self.hyperparams['PROJECTION_DIM_2']
        
        positions = tf.range(start=0, limit=num_patches, delta=1)

        # Encode the positions with an Embedding layer
        encoded_positions = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )(positions)

        # Add encoded positions to the projected patches
        return projected_patches + encoded_positions

    def __mlp(self, x, dropout_rate, hidden_units):
        # Iterate over the hidden units and add Dense => Dropout layers
        for units in hidden_units:
            x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)  # Dense layer with GELU activation
            x = tf.keras.layers.Dropout(dropout_rate)(x)  # Dropout layer
        return x

    def __transformer_2(self, encoded_patches):
        """
        Parameters:
        encoded_patches: Tensor
        """
        layer_norm_eps= self.hyperparams['LAYER_NORM_EPS_2']
        # Valor epsilon para la normalización por capas, utilizado para evitar divisiones por cero.
        projection_dim= self.hyperparams['PROJECTION_DIM_2']
        # La dimensión de proyección para las capas de atención multi-cabezal.
        num_heads= self.hyperparams['NUM_HEADS_2']
        # El número de cabezas en la capa de atención multi-cabezal.
        mlp_units= self.hyperparams['MLP_UNITS_2']
        #Una lista que contiene el número de unidades en cada capa de la red MLP (perceptrón multicapa).
        
        # Apply layer normalization
        x1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)

        # Multi-Head Self Attention layer
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Apply layer normalization again
        x3 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)(x2)

        # Apply MLP layer
        x4 = self.__mlp(x3, hidden_units=mlp_units, dropout_rate=0.1)

        # Second skip connection
        encoded_patches = tf.keras.layers.Add()([x4, x2])
        return encoded_patches

    def Transform_sh_2(self, inputs):
        # Apply squeeze and excitation layer
        inputs1 = self.CNN.squeeze_excitation_layer(inputs, out_dim=512, ratio=32.0, conv=False)
        print(inputs1.shape)

        # Project input patches using a Conv2D layer
        projected_patches = tf.keras.layers.Conv2D(
            filters=self.hyperparams['PROJECTION_DIM_2'],
            kernel_size=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            strides=(self.hyperparams['PATCH_SIZE_2'], self.hyperparams['PATCH_SIZE_2']),
            padding="VALID",
        )(inputs1)

        # Get the shape of the projected patches
        _, h, w, c = projected_patches.shape

        # Reshape the projected patches
        projected_patches = tf.keras.layers.Reshape((h * w, c))(projected_patches)  # (B, number_patches, projection_dim)

        # Add positional embeddings to the projected patches
        encoded_patches = self.__position_embedding(projected_patches)  # (B, number_patches, projection_dim)

        # Apply dropout
        encoded_patches = tf.keras.layers.Dropout(0.1)(encoded_patches)

        # Iterate over the number of layers and stack transformer blocks
        for i in range(self.hyperparams['NUM_LAYERS_2']):
            # Add a transformer block
            encoded_patches = self.__transformer_2(encoded_patches)

        return encoded_patches