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

class DRCA():
    '''
    The DRCA Class
    '''

    def __init__(self, n_components=2, alpha=None, mode='raw'):
        '''
        The function to initialize the DRCA class
        :param n_components: The intended dimensionality of projection hyperplane smaller than the initial dimensionality
        :param alpha: weighting factor for target domain data within class scatter
        :param mode: the mode of DRCA:
            'raw': consider source domain data (S) and target domain data (T) as two groups
            'number': consider type-specific source domain data and target domain data based on the average number of cases in S and T
            'mean': equal weights for each class
        '''
        self.mode = mode
        self.Sw_s = None
        self.Sw_t = None
        self.mu_s = None
        self.mu_t = None
        self.alpha = alpha
        self.D_tilde = n_components

    pass

    def fit(self, Xs, Xt, Ys=None, Yt=None):
        '''
        This function fit the DRCA model with the data and labels given by users
        :param Xs: the feature matrix of shape (Ns, D) in source domain, np.array
        :param Xt: the feature matrix of shape (Nt, D) in target domain, np.array
        :param Ys: the label of the data of shape (Ns,) in source domain, np.array, int
        :param Yt: the label of the data of shape (Nt,) in target domain, np.array, int
        '''
        ### --- Summarize statistics --- ###
        if self.mode != 'raw':
            Ys = Ys.reshape(-1, )  # we need to use Y and make sure the Y is the intended form
            Yt = Yt.reshape(-1, )
        Ns = Xs.shape[0]
        Nt = Xt.shape[0]
        D = Xs.shape[1]

        ### --- Within-domain scatter --- ###
        self.mu_s = np.mean(Xs, axis=0, keepdims=True)  # 1*D
        self.mu_t = np.mean(Xt, axis=0, keepdims=True)
        self.Sw_s = (Xs - self.mu_s).T @ (Xs - self.mu_s)  # D*D
        self.Sw_t = (Xt - self.mu_t).T @ (Xt - self.mu_t)  # D*D
        if self.alpha == None:
            self.alpha = Ns / Nt
        self.nominator = self.Sw_s + self.Sw_t * self.alpha

        ### --- Eliminate sensor drifts --- ###
        if self.mode == 'raw':  # S and T as two entities
            self.denominator = (self.mu_s - self.mu_t).T @ (self.mu_s - self.mu_t)  # D*D
        elif self.mode == 'number':  # Focus on the same classes appeared in target domain
            Kt = np.unique(Yt).shape[0]  # Assume that the target domain classes are fewer
            self.denominator = np.empty((D, D))
            for i in range(Kt):
                Ns = np.mean(Ys == Kt[i])
                Nt = np.mean(Yt == Kt[i])
                N = 0.5 * (self.Ns + self.Nt)  # self. ???????????????????
                mu_s_matrix = np.mean(Xs[Ys == Kt[i], :], axis=0, keepdims=True)
                mu_t_matrix = np.mean(Xt[Yt == Kt[i], :], axis=0, keepdims=True)
                Sb_matrix = (self.mu_s_matrix - self.mu_t_matrix).T @ (self.mu_s_matrix - self.mu_t_matrix)
                self.denomiator += N * Sb_matrix
        elif self.mode == 'mean':  # Equal weights for every class
            Kt = np.unique(Yt).shape[0]  # Assume that the target domain classes are fewer
            self.denominator = np.empty((D, D))
            for i in range(Kt):
                mu_s_matrix = np.mean(Xs[Ys == Kt[i], :], axis=0, keepdims=True)  # 1*D
                mu_t_matrix = np.mean(Xt[Yt == Kt[i], :], axis=0, keepdims=True)  # 1*D
                Sb_matrix = (self.mu_s_matrix - self.mu_t_matrix).T @ (self.mu_s_matrix - self.mu_t_matrix)
                self.denomiator += Sb_matrix  # D*D

        eigenValues, eigenVectors = np.linalg.eig(np.linalg.pinv(self.denominator) @ self.nominator)  # D*D

        idx = np.abs(eigenValues).argsort()[::-1]
        self.eigenValues = eigenValues[idx]
        self.eigenVectors = eigenVectors[:, idx]
        self.W = self.eigenVectors[:, 0:self.D_tilde]  # shape=(D,D_tilde)

    pass

    def transform(self, X):
        '''
        This function use the fitted SRLDA model
        :param X: the data in np.array of shape (N,D) that needs to be projected to the lower dimension
        :return: X_tilde: the projected data in the lower dimensional space in np.array of shape (N, D_tilde)
        '''
        return np.matmul(X, self.W)  # goal:  (N,D_tilde)      (D_tilde*D)@(D*N).T     (N*D)(D*D_tilde)

    pass

    def fit_transform(self, Xs, Xt, Ys=None, Yt=None):
        '''
        :param Xs: the feature matrix of shape (Ns, D) in source domain, np.array
        :param Xt: the feature matrix of shape (Nt, D) in target domain, np.array
        :param Ys: the label of the data of shape (Ns,) in source domain, np.array, int
        :param Yt: the label of the data of shape (Nt,) in target domain, np.array, int '''

        self.fit(Xs, Xt, Ys, Yt)
        return np.real(self.transform(Xs)), np.real(self.transform(Xt))  # N * D_tilde

    pass

#Problem Definition
task = 'MLHM2'
method = 'hand510'
dataset = 'HM2MMA' #HM/HM2CF/HM2MMA/HM2NASCARR
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

MMA1_X = io.loadmat('MMA_Tiernan_X.mat')['MLHM2_X']
MMA2_X= io.loadmat('MMA79_X.mat')['MLHM2_X']
MMA_X = np.row_stack([MMA1_X,MMA2_X])
assert MMA_X.shape[0] == 457

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

MMA1_Y = io.loadmat('MMA_Tiernan_Y.mat')['label']
MMA2_Y= io.loadmat('MMA79_Y.mat')['label']
MMA_Y = np.row_stack([MMA1_Y,MMA2_Y])
assert MMA_X.shape[0] == 457

###--- DRCA Transfer to CF dataset ---###
#1. Perform Standardization
X_test = MMA_X
Y_test = MMA_Y
xscaler = StandardScaler()
xscaler.fit(np.row_stack((X,X_test)))
X_std = xscaler.transform(X)
X_test_std = xscaler.transform(X_test)
Y_std, yscaler = YPreprocessing(Y=Y, method = Ymethod)

#2. Perform DRCA
components = [50,100,300]
alphas = [1,10,100]

MAE_recorder = []
RMSE_recorder = []
R2_recorder = []
for D in components:
    for alpha in alphas:
        print('Current Dimension: ', D)
        print('Current alpha: ',alpha)
        drca = DRCA(n_components=D,alpha=alpha)
        X_drca, X_test_drca = drca.fit_transform(Xs=X_std,Xt=X_test_std)

        
        #3. Develop model and training
        input_nodes = D
        hidden_layer = [500, 300, 100]
        lr = 0.001
        output_nodes = Y.shape[1]
        epoch = 600
        dropout = 0.5
        regularization = 0.01
        initialization = "normal"
        loss = "mean_squared_error"

        #3. Initialize model and compile model
        model = buildBaseModel(input_nodes, hidden_layer, dropout, output_nodes, lr, initialization, regularization,
                               loss='mean_squared_error')
        X_std_aug, Y_std_aug = train_augment(X_drca, Y_std)
        Adam = optimizers.adam(lr = lr, decay=5e-8)
        model.fit(X_std_aug, Y_std_aug, epochs=epoch, batch_size=512, verbose=0)

        #4. Perform test evaluation
        tik = time.clock()
        Y_predict_test_base_raw = model.predict(X_test_drca)
        tok = time.clock()
        predict_time = tok - tik
        Y_predict_test_base = YReconstruct(Y=Y_predict_test_base_raw, method=Ymethod, Yscaler=yscaler)

        MAB = mean_absolute_error(Y_test, Y_predict_test_base)
        MSB = mean_squared_error(Y_test, Y_predict_test_base)
        RMSB = np.sqrt(MSB)
        R2_base = r2_score(Y_test, Y_predict_test_base)
        print("MPSR MAB: %.2f(%.2f)", MAB)
        print("MPSR RMSB: %.2f(%.2f)", RMSB)
        print("MPSR R2_test: %.2f(%.2f)", R2_base)
        MAE_recorder.append(MAB)
        RMSE_recorder.append(RMSB)
        R2_recorder.append(R2_base)


print('\t'.join(['MAB']+[str(round(MAE_recorder[i],4)) for i in range(9)]))
print('\t'.join(['RMSB']+[str(round(RMSE_recorder[i],4)) for i in range(9)]))
print('\t'.join(['R2B']+[str(round(R2_recorder[i],4)) for i in range(9)]))