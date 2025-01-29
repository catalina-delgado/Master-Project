from src.imports import tf, keras, np, mpl
import os

class GradCAM():
    def __init__(self):
        super(GradCAM, self).__init__()
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
    def get_img_array(self, img_path, size):
        image = keras.utils.load_img(img_path, color_mode="grayscale", target_size=size)
        input_image = keras.utils.img_to_array(image)
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        
        return input_image
    
    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
            
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def save_and_overlay_gradcam(self, img_path, heatmap, model_name, layer_name):
        img = keras.utils.load_img(img_path)
        img = keras.utils.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        
        output_dir = os.path.join(os.getcwd(), "gradcam_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"gradcam_outputs/gradcam_{model_name}_{layer_name}.png"

        superimposed_img.save(output_filename)

        print(f"Grad-CAM saved in: {output_dir}")
        
    def decode_predictions(self, preds):
        """
        Decodes the model predictions to convert indices into readable labels.

        Args:
            preds (numpy.ndarray): Model output, an array with class probabilities.
        Returns:
            list: List of tuples (class_id, label, probability).
        """
        class_labels = {0: "cover", 1: "stego"}
        # Get index with highest probability
        if len(preds.shape) == 1: 
            idx = int(np.argmax(preds)) 
            confidence = preds[idx]
        else:  
            idx = int(np.argmax(preds, axis=1)[0])  
            confidence = preds[0][idx]

        print("idx:", idx) 
        
        label = class_labels[idx]
        confidence = preds[0][idx]
        return [(idx, label, confidence)]
