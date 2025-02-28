from src.imports import tf, np
from src.functions.main import Main
from src.models.residualTransformer_model import ResNet

class Training(Main):
    def __init__(self, epochs, batch_size):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def train_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = ResNet(inputs, srm_weights, biasSRM, learning_rate=5e-3)

        self.plot_model_summary(architecture.model, 'resT__model_summary')

        X_train = np.load('../database/data_gbras/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../database/data_gbras/y_train.npy') # (12000, 2)
        X_valid = np.load('../database/data_gbras/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../database/data_gbras/y_valid.npy') # (4000, 2)
        X_test = np.load('../database/data_gbras/X_test.npy') # (4000, 256, 256, 1)
        y_test = np.load('../database/data_gbras/y_test.npy') # (4000, 2)

        base_name="04S-UNIWARD"
        name="Model_"+'ResFormer_prueba1'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, num_test='library')
                
