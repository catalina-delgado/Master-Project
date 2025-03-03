import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from time import time
import time as tm
from contextlib import redirect_stdout

import os
import mlflow
from mlflow.tensorflow import MlflowCallback

class Main:
    def __init__(self):
        print('New function initialized')

    def fit(self, model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, initial_epoch=0, model_name="", dataset="", custom_layers={}):
        file_path = 'trained_models/'
        os.makedirs(file_path, exist_ok=True)
        
        start_time = tm.time()
        log_dir = os.path.join(file_path, f"{model_name}_{int(start_time)}")
        os.makedirs(log_dir, exist_ok=True)
        
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        checkpoint_path = os.path.join(log_dir, "saved-model.hdf5")
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        )

        if custom_layers:
            for name, layer in custom_layers.items():  
                print(f"Adding custom layer: {name}")
                tf.keras.utils.get_custom_objects()[name] = layer

        model.reset_states()
        if mlflow.active_run():
            mlflow.end_run()
        
        with mlflow.start_run(nested=True, run_name=f"/{model_name}_{int(start_time)}") as run:
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            history = model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_data=(X_valid, y_valid), initial_epoch=initial_epoch,
                callbacks=[tensorboard, checkpoint, MlflowCallback(run)]
            )
            
            model.load_weights(checkpoint_path)
            metrics = model.evaluate(X_test, y_test, verbose=0, callbacks=[MlflowCallback(run)])
            mlflow.log_metrics({k: v for k, v in zip(model.metrics_names, metrics)})
            
            max_val_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("max_val_accuracy", max_val_accuracy)

            saved_model_path = os.path.join(log_dir, "saved_model")
            model.save(saved_model_path, save_format="tf")
            mlflow.log_artifact(saved_model_path)
        
        self.plot_training_curves(history, model_name, log_dir)
        
        elapsed_time = tm.time() - start_time
        print(f"Duración de entrenamiento {model_name}: {elapsed_time:.2f} segundos")
        
        return {k: v for k, v in zip(model.metrics_names, metrics)}

    @staticmethod
    def plot_training_curves(history, model_name, results_dir):
        os.makedirs(results_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        for ext in ['eps', 'svg', 'pdf']:
            plt.savefig(os.path.join(results_dir, f'Accuracy_{model_name}.{ext}'), format=ext)
        plt.close()
        
        plt.figure(figsize=(10, 10))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        for ext in ['eps', 'svg', 'pdf']:
            plt.savefig(os.path.join(results_dir, f'Loss_{model_name}.{ext}'), format=ext)
        plt.close()

    def log_gradients(self, model, data, log_dir, steps_per_epoch=1500):
        file_writer = tf.summary.create_file_writer(log_dir)
        
        for step, (x_batch_train, y_batch_train) in enumerate(data.take(steps_per_epoch)):
            with tf.GradientTape() as tape:
                predictions = model(x_batch_train, training=True)
                loss = model.compiled_loss(y_batch_train, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            
            with file_writer.as_default():
                for grad, var in zip(gradients, model.trainable_variables):
                    tf.summary.histogram(f'{var.name}/gradients', grad, step=step)
                tf.summary.scalar('loss', loss, step=step)
            
            if step % 10 == 0:
                print(f"Logged gradients for step {step}")

    def plot_model_summary(self, model, file_name):
        file_path = os.path.join('src/graphs', file_name)
        os.makedirs(file_path, exist_ok=True)
        
        with open(os.path.join(file_path, f'{file_name}.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        plot_model(model, to_file=os.path.join(file_path, f'{file_name}.png'), show_shapes=True, show_layer_names=True)
