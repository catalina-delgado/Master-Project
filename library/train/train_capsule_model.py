from src.imports import tf, np
from src.functions.main import Main
from src.models.capsule_model import BlocksCapsule

class TrainingCapsule(Main):
    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

    def train_capsule_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        model = BlocksCapsule(inputs, srm_weights, biasSRM, learning_rate=1e-3)
        
        self.plot_model_summary(model.model, 'capsule_model_summary')
