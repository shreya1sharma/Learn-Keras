'''
Keras <2.3 do not provide precision, recall and f-measure metrics for training the model. 
Keras used to implement the f1 score in its metrics; however, the developers decided to remove it in Keras 2.0, 
since this quantity is evaluated for each batch, which is more misleading than helpful, since this metric is only
meaningful for the whole dataset. Fortunately, Keras allows us to access the validation data during training via 
a Callback function, on which we can extend to compute the desired quantities.
Following code gives both the batch and the callback version for calculating these metrics
'''

# batch version (removed from Keras 2.0)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 
metrics = [precision, recall]
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=metrics)
model.fit(X_train, y_train
          validation_data=(X_test, y_test),
          nb_epoch=10,
          batch_size=64)
          
          
          
# Callback version

from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import Callback

class prec_recall_f1Callback(Callback):
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs):
        pred = self.model.predict(self.X_val)
        prec_val = precision_score(self.y_val, np.round(pred), average='binary')
        recall_val = precision_score(self.y_val, np.round(pred), average='binary')
        f1_val = f1_score(self.y_val, np.round(pred), average='binary')
        print("prec_val = {}, recall_val = {}, f1_val = {}".format(prec_val, recall_val, f1_val))
        
# pass the above callback in model.fit

model.fit(X_train, y_train, 
          validation_data=(X_test, y_test),
          nb_epoch=10,
          batch_size=64,
          callbacks=[prec_recall_f1Callback(model, X_test, y_test)]) #prints out the metrics at the end of epoch
          
        
