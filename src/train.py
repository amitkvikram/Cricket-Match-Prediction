import pandas as pd
import numpy as np
from preprocessing import PreprocessingHelper
# from model import get_model
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Masking

preprocessingHelper = PreprocessingHelper()
train_df = pd.read_csv("../data/train_data.csv", index_col = 0)
val_df = pd.read_csv("../data/val_data.csv", index_col = 0)
train_df, train_df_c = preprocessingHelper.preprocess(train_df)
val_df, val_df_c = preprocessingHelper.preprocess(val_df)
X_train, Y_train = preprocessingHelper.getX_Y(train_df)
X_val, Y_val = preprocessingHelper.getX_Y(val_df)

lstm_model = get_model(preprocessingHelper)

filepath="../weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = lstm_model.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
            batch_size=64, verbose=1, epochs=200, shuffle=True, callbacks = callbacks_list)
lstm_model.load_weights("../weights.best.hdf5")
test_df = pd.read_csv("../data/test_data.csv", index_col = 0)
test_df, test_df_c = preprocessingHelper.preprocess(test_df)
X_test, Y_test = preprocessingHelper.getX_Y(test_df)

def getAcc(X, Y):
    pred_y = lstm_model.predict(X, batch_size = 64)
    pred_y = pred_y.reshape(pred_y.shape[0], -1)
    Y_t = Y.reshape(Y.shape[0], -1)
    masking = Masking(mask_value=0.0, input_shape=(preprocessingHelper.max_sequence_length, len(preprocessingHelper.feature_cols)))
    mask = (masking(X)._keras_mask.numpy()).reshape(pred_y.shape).astype(np.float)
    pred_y = (pred_y >= 0.5).astype(np.float)
    acc = np.sum((Y_t == pred_y)*mask, axis = 0)/np.sum(mask, axis = 0)

    return acc