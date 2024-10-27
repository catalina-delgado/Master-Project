from src.imports import tf, np
from src.functions.main import Main
from src.models.cvt_model import CVT

class TrainingCVT(Main):
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def __hyperparams(self):
        # Dimensiones y proyecciones
        IMAGE_SIZE_2 = 16
        PROJECTION_DIM_2 = 128

        # Patches
        PATCH_SIZE_2 = 4
        NUM_PATCHES_2 = (IMAGE_SIZE_2 // PATCH_SIZE_2) ** 2

        # Unidades MLP basadas en las dimensiones de proyección
        MLP_UNITS_2 = [PROJECTION_DIM_2 * 2, PROJECTION_DIM_2]

        LAYER_NORM_EPS_2 = 1e-6
        NUM_HEADS_2 = 4
        NUM_LAYERS_2 = 4

        return {
            'LAYER_NORM_EPS_2': LAYER_NORM_EPS_2,
            'PROJECTION_DIM_2': PROJECTION_DIM_2,
            'NUM_HEADS_2': NUM_HEADS_2,
            'NUM_LAYERS_2': NUM_LAYERS_2,
            'MLP_UNITS_2': MLP_UNITS_2,
            'IMAGE_SIZE_2': IMAGE_SIZE_2,
            'PATCH_SIZE_2': PATCH_SIZE_2,
            'NUM_PATCHES_2': NUM_PATCHES_2
        }

    def train_cvt_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../SRM_Kernels.npy') 
        biasSRM = np.ones(30)
        hyperparams = self.__hyperparams()

        model = CVT(inputs, srm_weights, biasSRM, hyperparams, learning_rate=1e-3)
        
        self.plot_model_summary(model.model, 'cvt_model_summary')
