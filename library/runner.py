import importlib
import os
import sys
import inspect

EPOCHS = 10
BATCH_SIZE = 32

def run_train_methods():

    train_dir = os.path.join(os.path.dirname(__file__), 'train')
    sys.path.insert(0, train_dir)  
    
    for filename in os.listdir(train_dir):
        if filename.startswith("train") and filename.endswith(".py"):
            module_name = filename[:-3]         
    
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.startswith("Train"):
                    print(f"Ejecutando métodos de {name} con epochs={EPOCHS}, batch_size={BATCH_SIZE}")
                    instance = obj(epochs=EPOCHS, batch_size=BATCH_SIZE) 
                    
                    for method_name, method in inspect.getmembers(instance, inspect.ismethod):
                        if "train" in method_name:
                            print(f"Ejecutando {name}.{method_name}()")
                            method()

if __name__ == "__main__":
    run_train_methods()
