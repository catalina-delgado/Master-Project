from src.imports import tf, cv2, np, plt, K
from src.functions.shap_values import GradCAM
from src.layers.transformer import Transformer


class MapKAN_prueba2(GradCAM):
    def __init__(self):
        self.code_image = "image_stego"
        self.image_path = f"gradcam\images\{self.code_image}.pgm" 
        self.input_image = self.get_img_array(self.image_path, (256, 256))

        self.img_array = tf.keras.preprocessing.image.load_img(self.image_path, target_size=(256, 256))
        self.img_array = np.expand_dims(np.array(self.img_array), axis=0)    

    def __Tanh3(self, x):
        T3 = 3
        tanh3 = K.tanh(x)*T3
        return tanh3

    def generate_gradcam(self):
        model_path = "trained_models\Model_CVT_prueba2_04S-UNIWARD_1737733248.263399\saved-model.hdf5"
        model = tf.keras.models.load_model(model_path, custom_objects={
            '__Tanh3':self.__Tanh3,
            'transformer':Transformer
            })
        
        # Print what the top predicted class is
        preds = model.predict(self.input_image)
        print("Predicted:", self.decode_predictions(preds)) 
        
        shape_values = self.make_shap_heatmap(self.input_image, model)  
        
        self.save_figure_with_shape_values(self.img_array, shape_values, f"Model_CVT_prueba2-{self.code_image}")