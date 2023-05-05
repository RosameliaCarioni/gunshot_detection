from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold

# Method used to calculate different performance measures of a model. 
# It returns confusion_matrices, histories
def performance(location_model, x, y, epoch=15, batch_size=8):
    # Set values 
    epoch = epoch
    batch = batch_size
    splits = 10
    location = location_model
    lr = 0.02661877777328162 # result from param optimization
    loss = "BinaryCrossentropy"

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=123)
    acc_scores = [] # of test data 
    histories = []
    confusion_matrices = []

    for train, test in kfold.split(x, y):
        # 1. Load model 
        model = keras.models.load_model(location)

        # 2. Add precision, recall as metrics
        model.compile(loss= loss, optimizer=keras.optimizers.legacy.SGD(learning_rate = lr), metrics = ['accuracy', 'Recall', 'Precision']) 
    
        # 3. Fit the model
        x_train = np.array(x)[train.astype(int)]
        y_train = np.array(y)[train.astype(int)]
        x_test = np.array(x)[test.astype(int)]
        y_test = np.array(y)[test.astype(int)]
        
        hist = model.fit(x_train, y_train, epochs=epoch, batch_size=batch, verbose=0, validation_data = (x_test, y_test))
        
        # Save information about model 
        histories.append(hist)
        
        # Display accuracy of validation set 
        # hist.history returns all the metrics. By adding: ['val_accuracy'][-1] we get only the accuracy of the testing set at the last epoch
        print("%s: %.2f%%" % (model.metrics_names[1], hist.history['val_accuracy'][epoch-1] *100))
        acc_scores.append(hist.history['val_accuracy'][epoch-1] * 100)

        # Store confusion matrix 
        y_pred = model.predict(x_test)
        y_pred = [1 if prediction > 0.5 else 0 for prediction in y_pred]
        confusion_mtx = tf.math.confusion_matrix(y_test, y_pred)
        confusion_matrices.append(confusion_mtx)
    
    print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))
    return confusion_matrices, histories


def get_metrics(epoch, histories):

    # Initialize the lists that will be used and returned 
    list_loss = [], list_val_loss = []
    list_precision = [], list_val_precision = []
    list_recall = [], list_val_recall = []
    list_accuracy = [], list_val_accuracy = []

    for i in range(epoch):
        temp_loss = [ hist.history['loss'][i] for hist in histories ]
        list_loss.append(np.mean(temp_loss))
        temp_val_loss = [ hist.history['val_loss'][i] for hist in histories ]
        list_val_loss.append(np.mean(temp_val_loss))

        temp_precision = [ hist.history['precision'][i] for hist in histories ]
        list_precision.append(np.mean(temp_precision))
        temp_val_precision = [ hist.history['val_precision'][i] for hist in histories ]
        list_val_precision.append(np.mean(temp_val_precision))

        temp_recall = [ hist.history['recall'][i] for hist in histories ]
        list_recall.append(np.mean(temp_recall))
        temp_val_recall = [ hist.history['val_recall'][i] for hist in histories ]
        list_val_recall.append(np.mean(temp_val_recall))

        temp_accuracy = [ hist.history['accuracy'][i] for hist in histories ]
        list_accuracy.append(np.mean(temp_accuracy))
        temp_val_accuracy = [ hist.history['val_accuracy'][i] for hist in histories ]
        list_val_accuracy.append(np.mean(temp_val_accuracy))
    return list_loss, list_val_loss, list_precision, list_val_precision, list_recall, list_val_recall, list_accuracy, list_val_accuracy

