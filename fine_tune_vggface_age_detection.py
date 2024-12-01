
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load and preprocess the IMDB-WIKI dataset
def load_imdb_wiki_data():
    # This function is a placeholder. You should load the actual IMDB-WIKI dataset here.
    # X = Load image data
    # y = Load age labels
    return X, y

# Create a function to preprocess the images
def preprocess_images(X):
    return preprocess_input(X, version=2)  # Use VGGFace2 preprocessing

# Create a modified VGGFace model
def create_vggface_age_model():
    # Load VGGFace model pre-trained on VGGFace2 dataset
    base_model = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3))

    # Freeze all layers except the last few layers for fine-tuning
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Add custom layers for age detection
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='linear'))  # Linear activation for regression

    return model

# Compile, train, and evaluate the model
def fine_tune_model(X_train, y_train, X_test, y_test):
    model = create_vggface_age_model()
    
    # Compile the model
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mean_squared_error', metrics=['mae'])

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    
    # Train the model
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                        validation_data=(X_test, y_test),
                        epochs=10, batch_size=32)

    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    return model, history

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load dataset
    X, y = load_imdb_wiki_data()

    # Preprocess images
    X = preprocess_images(X)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fine-tune the model
    model, history = fine_tune_model(X_train, y_train, X_test, y_test)

    # Plot the training history
    plot_training_history(history)

    # Save the model
    model.save('fine_tuned_vggface_age_model.h5')
