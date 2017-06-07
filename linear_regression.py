
# import keras.backend as Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input
from keras import optimizers
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, cross_validation, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import pandas

def run_linear_regression():
	# Define the Variables and Model
	x = Keras.placeholder() # Input
	target = Keras.placeholder() # Target
	lr = Keras.variable(0.1) # Learning Rate
	delta = Keras.variable(np.random.random())
	b = Keras.variable(np.random.random())

	# Model and cost
	y = delta * x + b
	cost = Keras.mean(Keras.square(y - target))

	grads = Keras.gradients(cost, [delta, b])
	updates = [(delta, delta - lr * grads[0]), (b, b - lr * grads[1])]
	train = Keras.function(inputs=[x, target], outputs=[cost], updates=updates)

	# Generate Dummy data
	dummy_x = np.random.random(1000)
	dummy_target = 0.96 * dummy_x + 0.24


	# Training
	loss_history = []
	for epoch in range(200):
		current_loss = train([dummy_x, dummy_target])[0]
		loss_history.append(current_loss)
		if epoch % 20 == 0:
			print("Loss: %.03f, w, b: [%.02f, %.02f]" % (current_loss, Keras.eval(delta), Keras.eval(b)))

	# Plot history
	plt.plot(loss_history)
	plt.show()

def load_data(filename):
	dataframe = pandas.read_csv(filename, header=None)
	dataset = dataframe.values
	X = dataset[:,0].astype(float)
	# Y = dataset[:,1:].astype(float)
	Y = dataset[:,250].astype(float) # predict just one curve

	m = dataframe.shape[0] # ROWS or test samples
	X_test = X[m-1]
	Y_test = Y[m-1]

	# preprocess
	min_max_scaler = preprocessing.MinMaxScaler()
	min_max_scaler.fit(X)
	X = min_max_scaler.transform(X)

	model = Sequential()

	# See this to understand input params
	# http://keras.dhpit.com/
	# model.add(Dense(1, input_dim=1, activation='linear'))
	# model.add(Dense(257))

	model.add(Dense(input_dim=1, output_dim=500))
	model.add(Dense(output_dim=1))
	# sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)
	model.compile(loss='mean_absolute_error', optimizer='rmsprop')
	model.summary()
	print("Inputs: {}".format(model.input_shape))
	print("Outputs: {}".format(model.output_shape))
	# print("Actual input: {}".format(data.shape))
	# print("Actual output: {}".format(target.shape))
	# see https://stackoverflow.com/a/38494022 for explanation about epoch and batch
	model.fit(X, Y, epochs=20, batch_size=1)
	# score = model.evaluate(X_test, Y_test, batch_size=1)

	# evaluate the model
	# scores = model.evaluate(X, Y)
	# print("%.2f%%" % (scores[0] * 100))
	# print(scores)
	print("Predicting 8000")
	# print()
	pred = model.predict(X)
	# plt.scatter(Y, pred)
	plt.plot(X, pred)
	plt.show()
	# Plot history
	# Fs = 44100
	# f = Fs * 

	# plt.plot(pred)
	# plt.show()
	np.savetxt("pred.csv", pred, fmt='%.7f', delimiter=",")


def run_sklearn():
	data_reg = datasets.make_regression(n_samples=5000, n_features=100,n_informative=100,n_targets=1, noise=100, random_state=0)

	# Scaling
	scale = abs(data_reg[1]).max()
	data_reg_new = data_reg[1] / scale

	# lm_lr = linear_model.LinearRegression()
	# lm_lr.fit(data_reg[0], data_reg[1])
	# plt.scatter(data_reg[1], lm_lr.predict(data_reg[0]))
	# plt.show()
	model = Sequential()
	model.add(Dense(1, input_dim=100, activation='linear'))
	# model.add(Dense(10))
	sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=False)
	model.compile(loss="mse", optimizer=sgd)
	model.fit(data_reg[0], data_reg_new, nb_epoch=10, batch_size=16)

	
	# print('linear regression: ', model.metrics.mean_squared_error(data_reg_new, lm_lr.predict(data_reg[0])))
	plt.scatter(data_reg_new, model.predict(data_reg[0]))
	plt.show()

def run_sklearn_poly(filename):
	dataframe = pandas.read_csv(filename, header=None)
	dataset = dataframe.values
	X = dataset[:,0].astype(float)
	# Y = dataset[:,1:].astype(float)
	# Y = dataset[:,257:].astype(float) # predict just one curve
	Y = dataset[:,255].astype(float) # predict just one curve

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

	##
	X = X.reshape(-1, 1)

	# X = np.sort(5 * np.random.rand(40, 1), axis=0)
	# Y = np.sin(X).ravel()
	# print(X)
	# print(Y)
	# model = make_pipeline(PolynomialFeatures(500), Ridge())
	# svr_rbf = SVR(kernel='rbf')

	# svr_multi = MultiOutputRegressor(SVR(kernel='rbf', C=1000),n_jobs=-1)
	# svr_multi.fit(X, Y)
	# y_rbf = svr_multi.predict(X)

	svr_rbf = SVR(kernel='rbf', C=10000)
	y_rbf = svr_rbf.fit(X, Y).predict(X)

	# svr_poly = SVR(kernel='poly', C=1e3, degree=2)


	# model.fit(X, Y)
	plt.plot(X, y_rbf)
	plt.plot(X, Y)
	
	np.savetxt("predsvm.csv", y_rbf, fmt='%.7f', delimiter=",")

	plt.show()



# load_data('fftstore.csv')
# run_sklearn()
# run_linear_regression()
run_sklearn_poly('fftstore.csv')
