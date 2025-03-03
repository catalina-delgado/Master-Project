import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from functions.main import Main
import os


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

def Tanh3(x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3


################################## Functions ################################################################

# ViT ARCHITECTURE
#Hyperparameters 1 tRANSFORMER
# ViT ARCHITECTURE
LAYER_NORM_EPS_1 = 1e-6
PROJECTION_DIM_1 = 16
NUM_HEADS_1 = 4
NUM_LAYERS_1 = 4
MLP_UNITS_1 = [
    PROJECTION_DIM_1 * 2,
    PROJECTION_DIM_1,
]
# OPTIMIZER
LEARNING_RATE_2 = 1e-3
WEIGHT_DECAY_2 = 1e-4

IMAGE_SIZE_2 =  16# We will resize input images to this size.
PATCH_SIZE_2 = 4  # Size of the patches to be extracted from the input images.
NUM_PATCHES_2 = (IMAGE_SIZE_2 // PATCH_SIZE_2) ** 2
print(NUM_PATCHES_2)
# ViT ARCHITECTURE
LAYER_NORM_EPS_2 = 1e-6
PROJECTION_DIM_2 = 128
NUM_HEADS_2 = 4
NUM_LAYERS_2 = 4
MLP_UNITS_2 = [
    PROJECTION_DIM_2 * 2,
    PROJECTION_DIM_2
]

def position_embedding(projected_patches, num_patches=NUM_PATCHES_2, projection_dim=PROJECTION_DIM_2):
    # Build the positions.

    positions = tf.range(start=0, limit=num_patches, delta=1)

    # Encode the positions with an Embedding layer.
    encoded_positions = tf.keras.layers.Embedding(
        input_dim=num_patches, output_dim=projection_dim
    )(positions)

    # Add encoded positions to the projected patches.
    return projected_patches + encoded_positions

def mlp(x, dropout_rate, hidden_units):
    # Iterate over the hidden units and
    # add Dense => Dropout.
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def transformer_1(encoded_patches):
    # Layer normalization 1.
    x1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_1)(encoded_patches)
    
    # Multi Head Self Attention layer 1.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS_1, key_dim=PROJECTION_DIM_1, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_1)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS_1, dropout_rate=0.1)

    # Skip connection 2.
    encoded_patches = tf.keras.layers.Add()([x4, x2])
    return encoded_patches

def Transform_sh_1(inputs):
    projected_patches = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(16,16),
        strides=(1,1),
        padding="same",
    )(inputs)
    _, h, w, c = projected_patches.shape
    print(c)
    projected_patches = tf.keras.layers.Reshape((h * w, c))(
        projected_patches
    )  # (B, number_patches, projection_dim)

    # Iterate over the number of layers and stack up blocks of
    # Transformer.
    for i in range(NUM_LAYERS_1):
        # Add a Transformer block.
        encoded_patches = transformer_1(projected_patches)
        # Add TokenLearner layer in the middle of the
        # architecture. The paper suggests that anywhere
        # between 1/2 or 3/4 will work well.
        _, hh, c = encoded_patches.shape
        h = int(tf.math.sqrt(hh))
        encoded_patches = tf.keras.layers.Reshape((h, h, c))(encoded_patches)
    print(encoded_patches.shape)
        #print(encoded_patches.shape)
    return encoded_patches

def transformer_2(encoded_patches):

    x1 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(encoded_patches)

    # Multi Head Self Attention layer 1.
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS_2, key_dim=PROJECTION_DIM_2, dropout=0.1
    )(x1, x1)

    # Skip connection 1.
    x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

    # Layer normalization 2.
    x3 = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(x2)

    # MLP layer 1.
    x4 = mlp(x3, hidden_units=MLP_UNITS_2, dropout_rate=0.1)

    # Skip connection 2.
    encoded_patches = tf.keras.layers.Add()([x4, x2])
    return encoded_patches

def Transform_sh_2(inputs):
    inputs1 = squeeze_excitation_layer(inputs, out_dim=512, ratio=32.0, conv=False)
    print(inputs1.shape)
    projected_patches = tf.keras.layers.Conv2D(
          filters=PROJECTION_DIM_2,
          kernel_size=(PATCH_SIZE_2, PATCH_SIZE_2),
          strides=(PATCH_SIZE_2, PATCH_SIZE_2),
          padding="VALID",
      )(inputs1)
    _, h, w, c = projected_patches.shape
    projected_patches = tf.keras.layers.Reshape((h * w, c))(
          projected_patches
      )  # (B, number_patches, projection_dim)
      # Add positional embeddings to the projected patches.
    encoded_patches = position_embedding(
          projected_patches
      )  # (B, number_patches, projection_dim)
    
    encoded_patches = tf.keras.layers.Dropout(0.1)(encoded_patches)

      # Iterate over the number of layers and stack up blocks of
      # Transformer.
    for i in range(NUM_LAYERS_2):
          # Add a Transformer block.
        encoded_patches = transformer_2(encoded_patches)

    return encoded_patches
    

################################## Model  ################################################################
 
srm_weights = np.load('../../filters/SRM_Kernels.npy') 
biasSRM = np.ones(30) 

def new_arch():
  tf.keras.backend.clear_session()
  inputs = tf.keras.Input(shape=(256,256,1), name="input_1")
  #Layer 1
  layers_ty = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=False, activation=Tanh3, use_bias=True)(inputs)
  layers_tn = tf.keras.layers.Conv2D(30, (5,5), weights=[srm_weights,biasSRM], strides=(1,1), padding='same', trainable=True, activation=Tanh3, use_bias=True)(inputs)

  layers = tf.keras.layers.add([layers_ty, layers_tn])
  layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  #Layer 2
  
  # L1
  layers = Block_1(layers1,64)

  # L2
  layers = Block_1(layers,64)

  # L3 - L7
  for i in range(5):
    layers = Block_2(layers,64)

  # L8 - L11
  for i in [64, 64, 128, 256]:
    layers = Block_3(layers,i)

  # L12
  layers = Block_4(layers,512)
  #CVT=Transform_sh_1(layers)
  #CVT_2=Transform_sh_1(CVT)
  CVT1=Transform_sh_2(layers)

  representation = tf.keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPS_2)(CVT1)
  representation = tf.keras.layers.GlobalAvgPool1D()(representation)
  #---------------------------------------------------Fin de Transformer 2------------------------------------------------------------------------#
  # Classify outputs.
      #FC
  layers = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(representation)
  layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
  layers = tf.keras.layers.Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
  layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
  layers = tf.keras.layers.Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
  layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)

  #Softmax
  predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
  model =tf.keras.Model(inputs = inputs, outputs=predictions)
  #Compile
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)

  if compile:
      model.compile(optimizer= optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
      
      print ("Transformer_create")

  return model


model2 = new_arch()  
  
  
  
path_log_base = 'Transformer_modelos_finales/logs'
path_img_base = 'Transformer_modelos_finales/images'

if not os.path.exists(path_log_base):
    os.makedirs(path_log_base)
if not os.path.exists(path_img_base):
    os.makedirs(path_img_base)  
  
  
class TrainingTransformer_1(Main):
    def __init__(self, epochs, batch_size, dataset):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.DATASET = dataset

    def train_transformer_model(self):

        X_train = np.load('../../database/BOSS/'+self.DATASET+'/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../../database/BOSS/'+self.DATASET+'/y_train.npy') # (12000, 2)
        X_valid = np.load('../../database/BOSS/'+self.DATASET+'/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../../database/BOSS/'+self.DATASET+'/y_valid.npy') # (4000, 2)
        X_test = np.load('../../database/BOSS/'+self.DATASET+'/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../../database/BOSS/'+self.DATASET+'/y_test.npy')   # (4000, 2)

        base_name="04-"+self.DATASET
        name="Model_"+'Transformer_1_prueba1'+"_"+base_name
        _, history  = self.fit(
            model2, X_train, y_train, X_valid, y_valid, X_test, y_test, 
            batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, dataset=self.DATASET,
            custom_layers={'Tanh3':Tanh3, 'Transform_sh_2':Transform_sh_2}
        )

train = TrainingTransformer_1(epochs=3, batch_size=8, dataset='S-UNIWARD')
train.train_transformer_model()
