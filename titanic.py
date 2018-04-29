import tensorflow
import keras

from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd


def main():
    train_df = pd.read_csv("train.csv")

    # training data pre-process
    train_predictors = train_df.drop(["Survived","PassengerId","Name","Ticket"], axis=1)
    train_target = to_categorical(train_df[["Survived"]])

    train_predictors["Sex"] = train_predictors["Sex"].map({"female":0, "male":1}).astype(int)
    train_predictors = pd.get_dummies(data = train_predictors, columns = ["Embarked"])
    train_predictors = pd.get_dummies(data = train_predictors, columns = ["Cabin"])
    train_predictors["Age"] = train_predictors["Age"].fillna(train_predictors["Age"].mean())
    train_predictors["Fare"] = train_predictors["Fare"].fillna(train_predictors["Fare"].mean())

    # Set up the model
    model = Sequential()

    # Add the first layer
    model.add(Dense(100, activation="relu", input_shape = (train_predictors.shape[1],)))

    model.add(Dense(300, activation="relu"))

    model.add(Dense(32, activation="relu"))
    
    model.add(Dropout(0.2))

    # output layer
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer="sdg", loss="mean_squared_error")

    print("------- fit -------------")
    early_stopping_monitor = EarlyStopping(patience = 10)

    # Fit the model
    model.fit(train_predictors, train_target, batch_size=100, epochs = 100, verbose=1, validation_split=0.3, callbacks=[early_stopping_monitor])
	

if __name__ == "__main__":
    # execute only if run as a script
    main()
