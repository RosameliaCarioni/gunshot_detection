import tensorflow as tf

def train(model:tf.keras.Model, x_train:list, y_train:list, x_val:list, y_val:list, batch:int, epoch:int) -> tuple: 
    """Train a model by slicing the data into "batches" of size batch_size, and repeatedly iterating over the entire dataset for a given number of epochs.

    Args:
        model (tf.keras.Model): model to fit
        x_train (list): list of signals to train model
        y_train (list): list of labels for x_train (1: gunshot, 0: no gunshot)
        x_val (list): list of signals to validate model 
        y_val (list): list of labels for x_val (1: gunshot, 0: no gunshot)
        batch (int): number of samples that will be propagated through the network
        epoch (int): how many times mdoel will be trained. In other words, how many times model goes over training set. 

    Returns:
        tuple: model and history of training process
    """        
    
    # if the validation loss doens't improve after 5 epochs, then we stop the training
    stop_early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience=5) 

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch,
        epochs=epoch,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
        callbacks=[stop_early],
)
    return model, history 