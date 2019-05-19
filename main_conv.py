import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint

dfTrain = pd.read_csv('train.csv')

dfTest = pd.read_csv('test.csv')

dfSample = pd.read_csv('sample_submission.csv')

# deal with +-inf and NaN's
dfTrain = dfTrain.replace([np.inf, -np.inf], np.nan)
dfTrain = dfTrain.fillna(0)

dfTest = dfTest.replace([np.inf, -np.inf], np.nan)
dfTest = dfTest.fillna(0)

# make colnames for train/test data
feature_cols = []
for i in range(1612):
	feature_cols.append("f" + str(i))
target_col = ["y"]

# grab train data from colomns
X_Train = dfTrain[feature_cols].values
Y_Train = dfTrain[target_col].values

# grab test data from colomns
X_Test = dfTest[feature_cols].values

# preprocessing
# scale input data to range (0, 1)
scaler = MinMaxScaler(feature_range = (0, 1))
X_Train = scaler.fit_transform(X_Train).reshape((X_Train.shape[0], X_Train.shape[1], 1))
X_Test = scaler.fit_transform(X_Test).reshape((X_Test.shape[0], X_Test.shape[1], 1))

# grab true targets from sample
y_true = dfSample[target_col].values

# make model for training
inputs = Input(shape=(X_Train.shape[1], 1))
x = Conv1D(32, kernel_size=3, kernel_initializer='orthogonal', activation='relu')(inputs)
x = MaxPooling1D()(x)
x = Conv1D(64, kernel_size=3, kernel_initializer='orthogonal', activation='relu')(x)
x = MaxPooling1D()(x)
x = Flatten()(x)
x = Dense(64, activation='tanh')(x)
x = Dense(8)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', save_best_only=True, save_weights_only=True, mode='auto', period=1)

model.load_weights('weights.hdf5')

# fit the model
#model.fit(X_Train, Y_Train, batch_size=128, epochs=100, callbacks=[checkpoint], validation_split=0.3, shuffle=True)

# make predictions
y_pred = model.predict(X_Test, batch_size=128)

# put results into .csv file
dataset = pd.DataFrame({'sample_id':dfTest['sample_id'].values,'y': y_pred.reshape((y_pred.shape[0],))})
dataset.to_csv('output_predictions.csv', sep=',', index=False)
