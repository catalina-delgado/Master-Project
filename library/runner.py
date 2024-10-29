import importlib
import os
import sys
import inspect

def run_train_methods(epochs, batch_size):

    train_dir = os.path.join(os.path.dirname(__file__), 'train')
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    run_train_methods(epochs=args.epochs, batch_size=args.batch_size)
