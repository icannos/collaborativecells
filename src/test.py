import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, InputLayer, Add, Input
from tensorflow.python.ops.parallel_for import jacobian
import tensorflow as tf
import numpy as np

model = Sequential()
model.add(InputLayer((100,), name="in_1"))
model.add(Dense(32))
model.add(Activation('relu'))



model2 = Sequential()
model2.add(InputLayer((100,), name="in_2"))
model2.add(Dense(32))
model2.add(Activation('relu'))

i1 = Input((100,), name="in1")
i2 = Input((100,), name="in2")
output = Add()([model(i1), model2(i2)])

M = Model(inputs=[i1, i2], outputs=output)

M.compile(optimizer="adam", loss="mse")

i1 = tf.placeholder("float32", shape=(None,100))
i2 = tf.placeholder("float32", shape=(None, 100))

tensor = M([M.inputs[0],i2])

j = jacobian(tensor, M.trainable_weights)

print(j)


X1 = np.random.uniform(-5, 5, (10, 100))
X2 = np.random.uniform(-5, 5, (10, 100))
Y = np.random.uniform(-5, 5, (10, 100))


with K.get_session() as s:
    print(M.predict({M.inputs[0]: X1, M.inputs[0]:X2}))


