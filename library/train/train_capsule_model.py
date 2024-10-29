from src.imports import tf, np
from src.functions.main import Main
from src.models.capsule_model import BlocksCapsule

class TrainingCapsule(Main):
    def __init__(self, epochs, batch_size):
        self.EPHOCS = epochs
        self.BATCH_SIZE = batch_size

    def train_capsule_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = BlocksCapsule(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        
        self.plot_model_summary(architecture.model, 'capsule_model_summary')

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
        name="Model_"+'CAPSNET_prueba1'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPHOCS, model_name=name, num_test='library')
