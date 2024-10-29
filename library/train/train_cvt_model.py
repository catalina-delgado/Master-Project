from src.imports import tf, np
from src.functions.main import Main
from src.models.cvt_model import CVT

class TrainingCVT(Main):
    def __init__(self, epochs, batch_size):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

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
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)
        hyperparams = self.__hyperparams()

        architecture = CVT(inputs, srm_weights, biasSRM, hyperparams, learning_rate=5e-3)
        
        self.plot_model_summary(architecture.model, 'cvt_model_summary')

        #Train
        X_train = np.load('../data/data_gbras/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../data/data_gbras/y_train.npy') # (12000, 2)
        #Valid
        X_valid = np.load('../data/data_gbras/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../data/data_gbras/y_valid.npy') # (4000, 2)
        #Test
        X_test = np.load('../data/data_gbras/X_test.npy') # (4000, 256, 256, 1)
        y_test = np.load('../data/data_gbras/y_test.npy') # (4000, 2)

        base_name="04S-UNIWARD"
        name="Model_"+'CVT_prueba1'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, num_test='library')

