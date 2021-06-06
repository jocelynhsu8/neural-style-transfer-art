from tensorflow import keras

model = keras.applications.VGG19()

print(model.summary())
