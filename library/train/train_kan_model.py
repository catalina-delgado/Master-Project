from src.imports import tf, np
from src.functions.main import Main
from src.models.kan_model import BlocksKAN

class TrainingKAN(Main):
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def train_kan_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        model = BlocksKAN(inputs, srm_weights, biasSRM, learning_rate=1e-3)
        
        self.plot_model_summary(model.model, 'kan_model_summary')
