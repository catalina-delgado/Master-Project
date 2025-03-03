import importlib
import os
import sys
import inspect
import argparse

def run_train_methods(epochs, batch_size):

    train_dir = os.path.join(os.path.dirname(__file__), 'training')
    sys.path.insert(0, train_dir)  
    
    for filename in os.listdir(train_dir):
        if filename.startswith("train") and filename.endswith(".py"):
            module_name = filename[:-3]         
    
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.startswith("Train"):
                    print(f"Ejecutando métodos de {name} con epochs={epochs}, batch_size={batch_size}")
                    instance = obj(epochs=epochs, batch_size=batch_size) 
                    
                    for method_name, method in inspect.getmembers(instance, inspect.ismethod):
                        if "train" in method_name:
                            print(f"Ejecutando {name}.{method_name}()")
                            method()

def run_gradcam_scripts():
    gradcam_dir = os.path.join(os.path.dirname(__file__), 'gradcam')
    sys.path.insert(0, gradcam_dir)

    for filename in os.listdir(gradcam_dir):
        if filename.startswith("map") and filename.endswith(".py"):
            module_name = filename[:-3]
            module = importlib.import_module(module_name)
            print(f"Importado módulo: {module_name}")
            
            gradcam_class = None
            for name, obj in vars(module).items():
                if name.lower().startswith("map") and isinstance(obj, type):
                    gradcam_class = obj
                    break
            
            if gradcam_class:
                print(f"Ejecutando {module_name}.{gradcam_class.__name__}()")
                instance = gradcam_class()

                for method_name in dir(instance):
                    if method_name.startswith("generate"):
                        method = getattr(instance, method_name)
                        if callable(method):
                            print(f"Ejecutando {module_name}.{gradcam_class.__name__}.{method_name}()")
                            method()
            else:
                print(f"{module_name} no tiene una clase que empieza con `GradCAM`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training or Grad-CAM scripts.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'gradcam'], help='Select mode: train or gradcam')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    
    args = parser.parse_args()

    if args.mode == 'train':
        run_train_methods(batch_size=args.batch_size, epochs=args.epochs)
    elif args.mode == 'gradcam':
        run_gradcam_scripts()

