from src.imports import tf, cv2, np, plt
from src.functions.kerascam import GradCAM

class MapCapsuleModel(GradCAM):
    def __init__(self):
        self.image_path = "gradcam\images\9406.pgm" 
        self.input_image = self.get_img_array(self.image_path, (256, 256))
        self.pred_index = 1

    def generate_gradcam(self):
        model_path = "trained_models\Model_CAPSNET_prueba3_04S-UNIWARD_1731234285.0956836\saved-model.hdf5"
        layer_name = 'conv2d_27'
        model = tf.keras.models.load_model(model_path) 
        model.layers[-1].activation = None
        
        # Print what the top predicted class is
        preds = model.predict(self.input_image)
        print("Predicted:", self.decode_predictions(preds)) 
        
        gradcam = self.make_gradcam_heatmap(self.input_image, model, layer_name, pred_index=self.pred_index)  
        
        self.save_and_overlay_gradcam(self.image_path, gradcam, ".hdf5", model_path, layer_name)