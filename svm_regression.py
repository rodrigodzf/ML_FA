from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cross_validation, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import wavio

import pandas

def run_sklearn_poly(filename):
	dataframe = pandas.read_csv(filename, header=None)
	dataset = dataframe.values
	X = dataset[:,0].astype(float)
	# Y = dataset[:,1:].astype(float)
	# Y = dataset[:,257:].astype(float) # predict just one curve
	Y = dataset[:,1:].astype(float) # predict just one curve

	m = dataframe.shape[0] # ROWS or test samples
	X_test = X[m-1]
	Y_test = Y[m-1]

	# preprocess
	# min_max_scaler = preprocessing.MinMaxScaler()
	# min_max_scaler.fit(X)
	# X = min_max_scaler.transform(X)

	# min_max_scaler = preprocessing.MinMaxScaler()
	# min_max_scaler.fit(Y)
	# Y = min_max_scaler.transform(Y)

	# print(Y)
	# Y = Y.reshape(-1, 1)


	# 1000: 56
	# 2000: 60




	##
	X = X.reshape(-1, 1)

	# X = np.sort(5 * np.random.rand(40, 1), axis=0)
	# Y = np.sin(X).ravel()
	# print(X)
	# print(Y)
	# model = make_pipeline(PolynomialFeatures(500), Ridge())
	# svr_rbf = SVR(kernel='rbf')

	svr_multi = MultiOutputRegressor(SVR(kernel='rbf', C=1e6),n_jobs=-1)
	svr_multi.fit(X, Y)
	y_rbf = svr_multi.predict(X)

	# svr_rbf = SVR(kernel='rbf', C=10000)
	# y_rbf = svr_rbf.fit(X, Y).predict(X)

	# svr_poly = SVR(kernel='poly', C=1e3, degree=2)

	# Score
	print("SCORE=%.2f" % svr_multi.score(X, y_rbf))

	f, subplots = plt.subplots(2)

	# model.fit(X, Y)
	subplots[0].plot(Y[2])
	# subplots[1].plot(Y)

	Out = np.fft.ifft(Y[2])
	Out = Out * 100

	subplots[1].plot(Out)

	# mx = np.max(Out)
	mx = 32767
	audio = np.fromiter((s * mx for s in Out), dtype=np.int16)
	wavio.write('out.wav', audio, 44100)
	# plt.plot(X, y_rbf)
	# plt.plot(X, Y)
	
	np.savetxt("predsvm.csv", y_rbf, fmt='%.7f', delimiter=",")


	# Plot against freqs
	Fs = 44100
	samples = 512
	f = Fs * np.mgrid[0:512/2 + 1]/512
	# plt.plot(Y)

	plt.show()

	# np.array()
	# f_ = Fs*(0:(L/2))/L



# load_data('fftstore.csv')
# run_sklearn()
# run_linear_regression()
run_sklearn_poly('fftstore.csv')