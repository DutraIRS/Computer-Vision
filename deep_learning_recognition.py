#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

#%%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
classes = ['Airplane', 'Car', 'Bird', 'Cat', 'Deet', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

#%%
img_idx = 0

image = x_train[img_idx]
label = y_train[img_idx][0]

plt.imshow(image)
plt.axis('off')
plt.title('Classe: ' + classes[label])
plt.show()
#%%
# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to vectors (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
#%% Create CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model with the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
# %%
model.summary()

# %%
new_img = plt.imread('assets/dog_a.png')
new_entry = np.expand_dims(new_img, axis=0)
prediction = model.predict(new_entry)
predicted_class = np.argmax(prediction)

print("Prediction: ", prediction)
print(f"Predicted class: {predicted_class}")

# plot the image
plt.imshow(new_img)
plt.show()
print(classes[predicted_class])
# %%
print(np.sum(prediction))
# %% Visualise first layer filters
first_layer_weights = model.layers[0].get_weights()[0]
pesos_normalizados = (first_layer_weights - np.min(first_layer_weights)) / (np.max(first_layer_weights) - np.min(first_layer_weights))
pesos_normalizados *= 255

num_rows = 4
num_cols = 8

fig, axs = plt.subplots(num_rows, num_cols, figsize=(32, 8))

for i in range(32):
    ax = axs[i // num_cols, i % num_cols]
    filtro = pesos_normalizados[:, :, :, i]
    filtro_img = np.reshape(filtro, (3, 3, 3))
    
    filtro_pb = cv2.cvtColor(filtro_img, cv2.COLOR_BGR2GRAY)
    filtro_pb = filtro_pb.astype(np.uint8)

    ax.imshow(filtro_pb, cmap='gray')
    ax.set_title(f'Filter {i + 1}')
    ax.axis('off')

plt.tight_layout()
plt.show()