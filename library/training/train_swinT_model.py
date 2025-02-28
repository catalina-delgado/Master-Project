from src.imports import tf, np
from src.functions.main import Main
from src.models.swinTransformer_model import CVST

class TrainingCVST(Main):
    def __init__(self, epochs, batch_size):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def __hyperparams(self):
        # Dimensiones y proyecciones
        IMAGE_SIZE = 16
        PROJECTION_DIM = 128

        # Windows
        QKV_BIAS = True
        WINDOW_SIZE = 8
        SHIFT_SIZE = 2

        # Patches
        PATCH_SIZE = 2
        NUM_PATCHES = IMAGE_SIZE // PATCH_SIZE

        LAYER_NORM_EPS = 1e-5
        DROPOUT_RATE = 0.1
        NUM_HEADS = 4
        NUM_MLP = 512

        return {
            'LAYER_NORM_EPS': LAYER_NORM_EPS,
            'PROJECTION_DIM': PROJECTION_DIM,
            'NUM_HEADS': NUM_HEADS,
            'IMAGE_SIZE': IMAGE_SIZE,
            'NUM_PATCHES': NUM_PATCHES,
            'QKV_BIAS': QKV_BIAS,
            'WINDOW_SIZE': WINDOW_SIZE,
            'SHIFT_SIZE': SHIFT_SIZE,
            'DROPOUT_RATE': DROPOUT_RATE,
            'NUM_MLP': NUM_MLP,
            'PATCH_SIZE': PATCH_SIZE
        }

    def train_cvst_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)
        hyperparams = self.__hyperparams()

        architecture = CVST(inputs, srm_weights, biasSRM, hyperparams, learning_rate=5e-3)
        #prueba 1 - 4 block_2 - windows size 2, patch size 2, dropout rate 0.03, mlp 128 - shift 1, num_heads 8
        #prueba 2 - 5 block_2 - windows size 2, patch size 2, dropout rate 0.03, mlp 128 - shift 1, num_heads 8
        #prueba 3 - 5 block 2 - windows size 2, patch size 2, dropout rate 0.03, mlp 512 - shift 1, num_heads 8
        #prueba 4 - 5 block 2 - windows size 4, patch size 2, dropout rate 0.1, mlp 512 - shift 2, num_heads 8
        #prueba 5 - 5 block 2 - windows size 4, patch size 2, dropout rate 0.1, mlp 512 - shift 1, num_heads 4

        self.plot_model_summary(architecture.model, 'swinTransformer_model_summary')

        #Train
        X_train = np.load('../database/data_gbras/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../database/data_gbras/y_train.npy') # (12000, 2)
        #Valid
        X_valid = np.load('../database/data_gbras/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../database/data_gbras/y_valid.npy') # (4000, 2)
        #Test
        X_test = np.load('../database/data_gbras/X_test.npy') # (4000, 256, 256, 1)
        y_test = np.load('../database/data_gbras/y_test.npy') # (4000, 2)

    
        base_name="04S-UNIWARD"
        name="Model_"+'swinTransformer_prueba6'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, num_test='library')

