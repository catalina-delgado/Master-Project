from src.imports import tf, np
from src.functions.main import Main
from src.models.swinTBlocks_model import CVSTB

class Training(Main):
    def __init__(self, epochs, batch_size):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def train_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = CVSTB(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        #prueba 1 - lr_schedule 5e-4
        #prueba 2 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 6
        #prueba 3 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 2 - step4_depth 6
        #prueba 4 - learning rate 5e-3 - block3 - step1_depth 2 - step2_depth 2 - step3_depth 6 - step4_depth 11
        #prueba 5 - learning rate 5e-3 - block3 - step1_depth 3 - step2_depth 3 - step3_depth 2 - step4_depth 2
        #prueba 6 - learning rate 5e-3 - step1_depth 2 - step2_depth 2 - step3_depth 3 - step4_depth 3
        #prueba 7 - learning rate 5e-3 - block1 - step1_depth 3 - step2_depth 2 - step3_depth 2 - step4_depth 2
        #prueba 8 - learning rate 5e-3 - block1 - step1_depth 4 - step2_depth 2 - step3_depth 2 - step4_depth 1
        #prueba 9 - learning rate 5e-3 - step1_depth 4 - step2_depth 2 - step3_depth 2 - step4_depth 1 - PPMconcat - FC

        #self.plot_model_summary(architecture.model, 'swint__model_summary')

        X_train = np.load('../database/BOSS/WOW/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../database/BOSS/WOW/y_train.npy') # (12000, 2)
        X_valid = np.load('../database/BOSS/WOW/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../database/BOSS/WOW/y_valid.npy') # (4000, 2)
        X_test = np.load('../database/BOSS/WOW/X_test.npy') # (4000, 256, 256, 1)
        y_test = np.load('../database/BOSS/WOW/y_test.npy') # (4000, 2)

        base_name="04S-WOW"
        name="Model_"+'SWINTBlocks_prueba10'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, num_test='library-WOW')
                
