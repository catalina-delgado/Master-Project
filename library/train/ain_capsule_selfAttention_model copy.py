from src.imports import tf, np
from src.functions.main import Main
from src.models.capsule_selfAttention_model import BlocksCapsule

class TrainingCapsuleHinton(Main):
    def __init__(self, epochs, batch_size):
        self.EPHOCS = epochs
        self.BATCH_SIZE = batch_size

    def train_capsuleHinton_model(self):

        inputs = tf.keras.Input(shape=(256, 256, 1))
        srm_weights = np.load('../filters/SRM_Kernels.npy') 
        biasSRM = np.ones(30)

        architecture = BlocksCapsule(inputs, srm_weights, biasSRM, learning_rate=5e-3)
        # Prieba1 learning_rate 5e-3 batch_size 8 epochs 94 - epsilon hinton 1e-7 
        # Prieba2 learning_rate 5e-3 batch_size 8 epochs 14 - epsilon hinton 1e-21 fail
        # prueba3 learning_rate 5e-3 batch_size 8 epochs 14 - original squash - epsilon k.epsilon faile
        # prueba4 learning_rate 5e-3 batch_size 8 epochs 188 - original squash - epsilon 1e-7
        # prueba5 learning_rate 5e-3 batch_size 8 epochs 94 - original squash - epsilon 1e-7
        
        self.plot_model_summary(architecture.model, 'capsule__cluster_model_summary')

        X_train = np.load('../database/data_gbras/X_train.npy') # (12000, 256, 256, 1)
        y_train = np.load('../database/data_gbras/y_train.npy') # (12000, 2)
        X_valid = np.load('../database/data_gbras/X_valid.npy') # (4000, 256, 256, 1)
        y_valid = np.load('../database/data_gbras/y_valid.npy') # (4000, 2)
        X_test = np.load('../database/data_gbras/X_test.npy')   # (4000, 256, 256, 1)
        y_test = np.load('../database/data_gbras/y_test.npy')   # (4000, 2)

        base_name="04S-UNIWARD"
        name="Model_"+'CAPSNET_selfAttention_prueba5'+"_"+base_name
        architecture.model.summary()
        _, history  = self.fit(architecture.model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=self.BATCH_SIZE, epochs=self.EPHOCS, model_name=name, num_test='library')
