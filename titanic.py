import tensorflow
import keras

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import pandas as pd


def main():
	train_df = pd.read_csv("train.csv")

	# separate half of data for training, another portion for evaluate

	# training data pre-process
	train_predictors = train_df.loc[0:train_df.shape[0]/2, ["Pclass", "Fare"]]
	train_n_cols = train_predictors.shape[1]
	train_target = to_categorical(train_df.loc[0:train_df.shape[0]/2, ["Survived"]])

	# testing data pre-process
	test_predictors = train_df.loc[train_df.shape[0]/2 + 1: , ["Pclass", "Fare"]]
	test_n_cols = test_predictors.shape[1]
	test_target = to_categorical(train_df.loc[train_df.shape[0]/2 + 1: , ["Survived"]])

	# Set up the model
	model = Sequential()

	# Add the first layer
	model.add(Dense(50, activation="relu", input_shape = (train_n_cols,)))

	model.add(Dense(32, activation="relu"))
	
	# output layer
	model.add(Dense(2))

	# Compile the model
	model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
	#model.compile(optimizer="adam", loss="mean_squared_error")

	print("------- summary -------------")
	model.summary()

	print("------- fit -------------")
	# Fit the model
	model.fit(train_predictors, train_target, batch_size=100, epochs = 100, verbose=1, 
		validation_data=(test_predictors, test_target))

	
	print("------- evaluate -------------")
	# evaluate
	#score = model.evaluate(test_predictors, test_target, verbose=0)
	#print(score)
	#print('Test loss:', score[0])
	#print('Test accuracy:', score[1])

if __name__ == "__main__":
    # execute only if run as a script
    main()
