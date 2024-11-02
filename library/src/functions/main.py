from src.imports import tf, plt, tm, os, plot_model, redirect_stdout, mlflow, MlflowCallback
from contextlib import redirect_stdout

class Main():
    def __init__(self):
        print('new function')

    
    def fit(self, model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, initial_epoch = 0, model_name="", num_test=""):
        start_time = tm.time()
        log_dir="D:/testing_by_"+num_test+"/"+model_name+"_"+"{}".format(tm.time())
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir)
        filepath = log_dir+"/saved-model.hdf5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath, 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max',
            verbose=1
        )
        model.reset_states()
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(nested=True, run_name=num_test+"/"+model_name+"_"+"{}".format(tm.time())) as run:
            
            mlflow.log_param("model_name", model_name)
            mlflow.tensorflow.log_model(model, model_name)
            mlflow.keras.log_model(model, model_name)

            history=model.fit(X_train, y_train, epochs=epochs, 
                                callbacks=[tensorboard,  checkpoint, MlflowCallback(run)], 
                                batch_size=batch_size,
                                validation_data=(X_valid, y_valid),
                                initial_epoch=initial_epoch)

            model.load_weights(filepath)
            metrics = model.evaluate(X_test, y_test, verbose=0,  callbacks=[MlflowCallback(run)])
            mlflow.log_metrics({k: v for k, v in zip(model.metrics_names, metrics)})
        
            max_val_accuracy = max(history.history['val_accuracy'])
            mlflow.log_metric("max_val_accuracy", max_val_accuracy)
            
        results_dir="D:/testing_by_"+num_test+"/"+model_name+"/"
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
            plt.figure(figsize=(10, 10))
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Accuracy Vs Epochs')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid('on')
            plt.savefig(results_dir+'Accuracy_'+model_name+'.eps', format='eps')
            plt.savefig(results_dir+'Accuracy_'+model_name+'.svg', format='svg')
            plt.savefig(results_dir+'Accuracy_'+model_name+'.pdf', format='pdf')
            
            plt.figure(figsize=(10, 10))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Loss Vs Epochs')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.grid('on')
            plt.savefig(results_dir+'Loss_'+model_name+'.eps', format='eps')
            plt.savefig(results_dir+'Loss_'+model_name+'.svg', format='svg')
            plt.savefig(results_dir+'Loss_'+model_name+'.pdf', format='pdf')

        TIME = tm.time() - start_time
        print("Time "+model_name+" = %s [seconds]" % TIME)
        return {k:v for k,v in zip (model.metrics_names, metrics)}


    def plot_model_summary(self, model, file_name):
        
        file_path = os.path.join('src/graphs/'+file_name, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path + '.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        
        plot_model(model, to_file=file_path + '.png', show_shapes=True, show_layer_names=True)