import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    xs=np.array([1,2,3,4,5],dtype=int)
    ys=np.array([100,150,200,250,300],dtype=int)
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs,ys,epochs=2000)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)