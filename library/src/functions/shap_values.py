from src.imports import tf, keras, np, mpl, plt, cv2, shap
import os

class GradCAM():
    def __init__(self):
        super(GradCAM, self).__init__()
        os.environ["KERAS_BACKEND"] = "tensorflow"
        
    def get_img_array(self, img_path, size):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
        image = cv2.resize(image, size)  
        input_image = np.expand_dims(image, axis=0)  # Add batch dimension
        input_image = np.expand_dims(input_image, axis=-1)  # Add channel for compatibility
        return input_image
    
    def make_shap_heatmap(self, img_array, model):

        masker = shap.maskers.Image("blur(10,10)", img_array.shape[1:])
        explainer = shap.Explainer(model, masker, output_names=["cover", "stego"])
        shap_values = explainer(img_array, max_evals=1000, batch_size=8)

        shap_img = shap_values.values[0] 
        
        return shap_img
    
    def save_figure_with_shape_values(self, img_array, shap_values, model_name):
        class_names = ["cover", "stego"]

        fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(15, 5))
        
        axes[0].imshow(img_array[0].astype(np.uint8))
        axes[0].set_title("Original")
        axes[0].axis('off')

        for i in range(len(class_names)):
            shap_img = shap_values[..., i]  # SHAP para la clase i
            shap_img = np.sum(shap_img, axis=-1)  # Sumar sobre canales
            axes[i + 1].imshow(shap_img, cmap='coolwarm')
            axes[i + 1].set_title(f"Class {i}")
            axes[i + 1].axis('off')
        
        plt.colorbar(
            axes[i + 1].imshow(shap_img, cmap='coolwarm'), 
            ax=axes, 
            orientation='horizontal', 
            fraction=0.05,  # Aumenta el tamaño de la barra de color
            pad=0.1,        # Ajusta el espaciado
            aspect=40       # Controla la longitud de la barra (valores mayores la hacen más larga)
        )        
        output_dir = os.path.join(os.getcwd(), "gradcam_outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"gradcam_outputs/shape_values_{model_name}.png"

        plt.savefig(output_filename)

        print(f"Shap_values saved in: {output_dir}")
        
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
