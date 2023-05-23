#Load dependency packages
from keras.models import Sequential,Model, model_from_json
from keras import optimizers, initializers, regularizers, layers
from keras.layers import Dense, Dropout
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from pandas import Series
import tensorflow
import os
import time
import scipy.io as io
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import joblib

# Set basic directory definitions and fix random seed for reproducibility
Dir_results = 'G:\\共享云端硬盘\\Head Models (KTH, GHBMC, ML)\\ML\\Results'
Dir_data_X = 'G:\\共享云端硬盘\\Head Models (KTH, GHBMC, ML)\\ML\\Training Dataset\\MLHM2 X'
Dir_data_Y_MPSR = 'G:\\共享云端硬盘\\Head Models (KTH, GHBMC, ML)\\ML\\Training Dataset\\MLHM2 Y\\MPSR'
Dir_code = 'G:\\共享云端硬盘\\Head Models (KTH, GHBMC, ML)\\ML\\Code'
np.random.seed(7)

#Data expansion by adding noises
def train_augment(trainX, trainY):
  print('Using 3X Data Augmentation!')
  row = len(trainX[:,0])
  column = len(trainX[0,:])
  standard_deviation = np.std(trainX,0)
  add_01 = np.zeros([row,column])
  add_02 = np.zeros([row,column])
  add_03 = np.zeros([row,column])
  for id in range(0,row):
      add_01[id,:] = 0.01 * standard_deviation * np.random.randn(1, column)
      add_02[id,:] = 0.02 * standard_deviation * np.random.randn(1, column)
      #add_03[id,:] = 0.03 * standard_deviation * np.random.randn(1, column)
  augment_trainX = np.row_stack((trainX,trainX+add_01,trainX+add_02))#, trainX+add_03))
  augment_trainY = np.row_stack((trainY,trainY,trainY))#,trainY))
  return augment_trainX, augment_trainY

def buildBaseModel(input_nodes, hidden_layer, dropout, output_nodes, lr, initialization, regularization, loss='mean_squared_error'):
  model = Sequential()
  model.add(Dense(hidden_layer[0], input_dim=input_nodes, kernel_initializer=initialization,
                  kernel_regularizer=regularizers.l2(regularization), activation='relu', name='layer_input'))
  for i in range(1,len(hidden_layer)):
      model.add(Dropout(dropout))
      model.add(
        Dense(hidden_layer[i], kernel_initializer=initialization, kernel_regularizer=regularizers.l2(regularization),name='layer_'+str(i)+'_neurons'+str(hidden_layer[i]),
              activation='relu'))
  model.add(Dense(output_nodes, kernel_initializer=initialization))
  # Compile model
  Adam = optimizers.adam(lr = lr, decay=5e-8)
  model.compile(loss=loss, optimizer=Adam)
  return model

def modelFit(X_train, Y_train, X_val, Y_val, model, epoch, lr, batch_size=128, augment = True, verbose = False):
  if augment:
    X, Y = train_augment(X_train, Y_train)  # Standardized version
  import time
  print('Start Training: ')
  tik = time.clock()
  if verbose == True:
      history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch_size, verbose=0)
  else:
      model.fit(X, Y, validation_data=(X_val, Y_val), epochs=epoch, batch_size=batch_size, verbose=0)
  tok = time.clock()
  print('Training Time(s): ',(tok-tik))
  if verbose == True:
      plt.title("learning curve epoch: {}, lr: {}".format(str(epoch), str(lr)))
      loss, = plt.plot(history.history['loss'])
      val_loss, = plt.plot(history.history['val_loss'])
      plt.legend([loss, val_loss], ['loss', 'Val_loss'])
      plt.show()
      return model, (tok-tik), plt
  else:
      return model, (tok-tik)

def YPreprocessing(Y, method):
  if method == 'STD':
    Yscaler = StandardScaler()
    Yscaler.fit(Y)
    Y_out = Yscaler.transform(Y)
  elif method == 'LOG':
    Y_out = np.log(Y)
    Yscaler = None
  elif method == 'LOGSTD':
    Y_log = np.log(Y)
    Yscaler = StandardScaler()
    Yscaler.fit(Y_log)
    Y_out = Yscaler.transform(Y_log)
  else:
    Y_out = Y
    Yscaler = None
  return Y_out, Yscaler

def YTransform(Y, method, Yscaler=None):
  if method == 'STD':
    Y_out = Yscaler.transform(Y)
  elif method == 'LOG':
    Y_out = np.log(Y)
  elif method == 'LOGSTD':
    Y_log = np.log(Y)
    Y_out = Yscaler.transform(Y_log)
  else:
    Y_out = Y
  return Y_out

def YReconstruct(Y, method, Yscaler):
  if method == 'No':
      Y_out = Y
  elif method == 'LOG':
    Y_out = np.exp(Y)
  elif method == 'STD':
    Y_out = Yscaler.inverse_transform(Y)
  elif method == 'LOGSTD':
    Y_out = np.exp(Yscaler.inverse_transform(Y))
  return Y_out

#Problem Definition
task = 'MLHM2'
method = 'hand510'
dataset = 'HM2CF' #HM/HM2CF/HM2MMA/HM2NASCARR
outcome = 'MPSR'
Ymethod = 'LOG' #LOG/STD/NSTD(nothing)
test_ratio = 1
feature_excluded = ''
print('Problem Definition: ' + task + ' ' + method + ' ' + Ymethod + ' ' + dataset + ' ' + outcome)

#Load Dataset
print('Loading Data!')
os.chdir(Dir_data_X)
HMXYZ_X = io.loadmat('HMXYZ_X.mat')['MLHM2_X']
HMXNYZ_X = io.loadmat('HMXNYZ_X.mat')['MLHM2_X']
HMXZY_X = io.loadmat('HMXZY_X.mat')['MLHM2_X']
HMXZNY_X = io.loadmat('HMXZNY_X.mat')['MLHM2_X']
HMYXZ_X = io.loadmat('HMYXZ_X.mat')['MLHM2_X']
HMYZX_X = io.loadmat('HMYZX_X.mat')['MLHM2_X']
HMZXY_X = io.loadmat('HMZXY_X.mat')['MLHM2_X']
HMNYZX_X = io.loadmat('HMNYZX_X.mat')['MLHM2_X']
HMZYX_X = io.loadmat('HMZYX_X.mat')['MLHM2_X']
HMZXNY_X = io.loadmat('HMZXNY_X.mat')['MLHM2_X']
HMZNYX_X = io.loadmat('HMZNYX_X.mat')['MLHM2_X']
HMNYXZ_X = io.loadmat('HMNYXZ_X.mat')['MLHM2_X']
X = np.row_stack((HMXYZ_X, HMXNYZ_X, HMXZY_X, HMXZNY_X, HMYXZ_X, HMYZX_X, HMZXY_X, HMNYZX_X, HMZYX_X, HMZXNY_X,
                      HMZNYX_X, HMNYXZ_X))

AF_X = io.loadmat('AF_X.mat')['MLHM2_X']
PAC_X= io.loadmat('PAC12_X.mat')['MLHM2_X']
CF_X = np.row_stack([AF_X,PAC_X])
assert CF_X.shape[0] == 302

os.chdir(Dir_data_Y_MPSR)
HMXYZ_Y = io.loadmat('HMXYZ_Y.mat')['label']
HMXNYZ_Y = io.loadmat('HMXNYZ_Y.mat')['label']
HMXZY_Y = io.loadmat('HMXZY_Y.mat')['label']
HMXZNY_Y = io.loadmat('HMXZNY_Y.mat')['label']
HMYXZ_Y = io.loadmat('HMYXZ_Y.mat')['label']
HMYZX_Y = io.loadmat('HMYZX_Y.mat')['label']
HMZXY_Y = io.loadmat('HMZXY_Y.mat')['label']
HMNYZX_Y = io.loadmat('HMNYZX_Y.mat')['label']
HMZYX_Y = io.loadmat('HMZYX_Y.mat')['label']
HMZXNY_Y = io.loadmat('HMZXNY_Y.mat')['label']
HMZNYX_Y = io.loadmat('HMZNYX_Y.mat')['label']
HMNYXZ_Y = io.loadmat('HMNYXZ_Y.mat')['label']
Y = np.row_stack((HMXYZ_Y, HMXNYZ_Y, HMXZY_Y, HMXZNY_Y, HMYXZ_Y, HMYZX_Y, HMZXY_Y, HMNYZX_Y, HMZYX_Y, HMZXNY_Y,
                      HMZNYX_Y, HMNYXZ_Y)).reshape(X.shape[0], -1)

AF_Y = io.loadmat('AF_Y.mat')['label']
PAC_Y= io.loadmat('PAC_Y.mat')['label']
CF_Y = np.row_stack([AF_Y,PAC_Y])
assert CF_Y.shape[0] == 302

###--- Transfer to CF dataset ---###
#0. Load models and x, y scalers
os.chdir(Dir_results+'\\'+ task + '\\' + method + '\\' + dataset)
specifics = 'HM_MPSR_transfer_base_LOG'
model_json_path = "."+"\\"+ outcome + "\\model\\" + specifics + "_model.json"
model_joblib_path = "."+"\\"+ outcome + "\\model\\" + specifics + "_model.joblib"
model_weight_path = "."+"\\"+ outcome + "\\model\\" + specifics + '_model.json_weight.h5'
#x_scaler_path = "."+"\\"+ outcome + "\\model\\" + specifics + "_xscaler.joblib"
y_scaler_path = "."+"\\"+ outcome + "\\model\\" + specifics + "_yscaler.joblib"
#Load invariant models (no fine-tuning)
#HM_base = joblib.load(model_joblib_path)
json_file = open(model_json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
HM_base = model_from_json(loaded_model_json)
# load weights into new model
HM_base.load_weights(model_weight_path)
print("Loaded model from disk!")
#xscaler = joblib.load(x_scaler_path)
yscaler = joblib.load(y_scaler_path)
HM_base.summary()

xscaler = StandardScaler()
xscaler.fit_transform(np.row_stack((X)))

X_test = CF_X
Y_test = CF_Y
xscaler.fit(X)
X_test_std = xscaler.transform(X_test)

tik = time.clock()
Y_predict_test_base_raw = HM_base.predict(X_test_std)
tok = time.clock()
predict_time = tok - tik
Y_predict_test_base = YReconstruct(Y=Y_predict_test_base_raw, method=Ymethod, Yscaler=yscaler)

MAB = mean_absolute_error(Y_test, Y_predict_test_base)
MSB = mean_squared_error(Y_test, Y_predict_test_base)
RMSB = np.sqrt(MSB)
R2_base = r2_score(Y_test, Y_predict_test_base)
print('Y_test shape: ', Y_test.shape)
print('Y_predict shape: ', Y_predict_test_base.shape)
print("MPS MSB: %.2f(%.2f)", MSB)
print("MPS MAB: %.2f(%.2f)", MAB)
print("MPS RMSB: %.2f(%.2f)", RMSB)
print("MPS R2_test: %.2f(%.2f)", R2_base)
print('MPS Predict Time: %.2f(%.2f)', predict_time)