import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
import joblib

def train_cnn_model():
    """Main training function"""
    print("Starting CNN model training...")
    
    # Check if we have data
    data_dir = "dataset/collected_images"
    if not os.path.exists(data_dir):
        print("No data found. Please collect data first.")
        print("Run: python collect_data.py")
        return
    
    # Load and preprocess data
    X, y = [], []
    letters = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for letter in letters:
        letter_dir = os.path.join(data_dir, letter)
        images = [f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(letter_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0  # Normalize
            
            X.append(img)
            y.append(letter)
    
    X = np.array(X).reshape(-1, 64, 64, 1)
    y = np.array(y)
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Build CNN model
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(letters), activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    
    # Save
    model.save('cnn_model.h5')
    joblib.dump(encoder, 'label_encoder.pkl')
    
    print(f"Model saved! Test accuracy: {model.evaluate(X_test, y_test)[1]:.2f}")

if __name__ == "__main__":
    train_cnn_model()