from src.imports import tf, cv2, np, plt, K
from src.functions.kerascam import GradCAM
from src.layers.transformer import Transformer


class MapCVT_prueba1(GradCAM):
    def __init__(self):
        self.code_image = "985"
        self.image_path = f"gradcam\images\{self.code_image}.pgm" 
        self.input_image = self.get_img_array(self.image_path, (256, 256))
        self.pred_index = 1

    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

    def generate_gradcam(self):
        model_path = "trained_models\Model_CVT_prueba1_04S-UNIWARD_1730289339.6206295\saved-model.hdf5"
        layer_name = 'conv2d_27'
        model = tf.keras.models.load_model(model_path, custom_objects={
            '__Tanh3':self.__Tanh3,
            'transformer':Transformer
            }) 
        model.layers[-1].activation = None
        
        # Print what the top predicted class is
        preds = model.predict(self.input_image)
        print("Predicted:", self.decode_predictions(preds)) 
        
        gradcam = self.make_gradcam_heatmap(self.input_image, model, layer_name, pred_index=self.pred_index)  
        
        self.save_and_overlay_gradcam(self.image_path, gradcam, f"Model_CVT_prueba1-{self.code_image}", layer_name)