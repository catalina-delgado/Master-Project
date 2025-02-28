from src.imports import tf, np
from src.functions.main import Main
from src.models.kan_model import BlocksKAN

class TrainingKAN(Main):
    def __init__(self, epochs, batch_size):
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    def train_kan_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = BlocksKAN(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        #prueba 1 learning_rate 5e-3 3FC 64-32-16
        #prueba 2 learning_rate 1e-3 2FC -16-4
        
        #self.plot_model_summary(architecture.model, 'kan_model_summary')

        #Train
        X_train = np.load('../database/BOSS/WOW/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../database/BOSS/WOW/y_train.npy') # (12000, 2)
        #Valid
        X_valid = np.load('../database/BOSS/WOW/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../database/BOSS/WOW/y_valid.npy') # (4000, 2)
        #Test
        X_test = np.load('../database/BOSS/WOW/X_test.npy') # (4000, 256, 256, 1)
        y_test = np.load('../database/BOSS/WOW/y_test.npy') # (4000, 2)

        base_name="04S-WOW"
        name="Model_"+'KAN_prueba2'+"_"+base_name
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS, model_name=name, num_test='library-WOW')
