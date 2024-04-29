import tensorflow as tf; 
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# The LeNet-5 architechture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5,5), padding='same', input_shape=(28,28,1), activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Conv2D(16, kernel_size=(5,5), input_shape=(10,10,1), activation='sigmoid'),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation='sigmoid'),
    tf.keras.layers.Dense(84, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

# Setup for training model
predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])