import tensorflow as tf
from PIL import Image

# Load the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# One hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Filter the data to only keep the cat and squirrel classes
cat_index = 3
squirrel_index = 6

train_filter = (y_train[:, cat_index] == 1) | (y_train[:, squirrel_index] == 1)
x_train = x_train[train_filter]
y_train = y_train[train_filter]

test_filter = (y_test[:, cat_index] == 1) | (y_test[:, squirrel_index] == 1)
x_test = x_test[test_filter]
y_test = y_test[test_filter]

# Convert the labels back to binary class
y_train = (y_train[:, cat_index] == 1).astype('float32')
y_test = (y_test[:, cat_index] == 1).astype('float32')


# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))


# Save the model to disk
model.save("cat_squirrel_model.h5")


# Load the model from disk
loaded_model = tf.keras.models.load_model("cat_squirrel_model.h5")


squirel = r"squirel.jpg"
cat = r"cat_feline_cats_eye_215231.jpg"
obvious_cat = r"obvious_cat.jpg"
obvious_squirel = r"obvious_squirel.jpg"

pics = [cat,squirel,obvious_cat,obvious_squirel ]

for x in pics:
    new_image = Image.open(x)
    new_image = new_image.resize((32, 32))
    new_image = tf.keras.preprocessing.image.img_to_array(new_image)
    new_image = new_image / 255.0
    new_image = new_image.reshape(1, 32, 32, 3)
    
    
    prediction = loaded_model.predict(new_image)
    
    print("This Image is {}% likely to be a cat".format(float(prediction[0])*100))
   
