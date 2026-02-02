import os
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_collected_data():
    """Load images from collected dataset"""
    data_dir = 'dataset/collected_images'
    
    images = []
    labels = []
    
    if not os.path.exists(data_dir):
        print("⚠️  No dataset found. Please collect data first.")
        return None, None
    
    letters = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for letter in letters:
        letter_dir = os.path.join(data_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
            
        image_files = [f for f in os.listdir(letter_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(letter_dir, img_file)
            try:
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convert to grayscale
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize
                img_resized = cv2.resize(img_gray, (64, 64))
                
                # Normalize
                img_normalized = img_resized / 255.0
                
                images.append(img_normalized)
                labels.append(letter)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    if len(images) == 0:
        print("❌ No images loaded. Please collect more data.")
        return None, None
    
    print(f"✅ Loaded {len(images)} images for {len(set(labels))} letters")
    return np.array(images), np.array(labels)

def create_cnn_model(input_shape=(64, 64, 1), num_classes=26):
    """Create CNN model for sign language recognition"""
    model = keras.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Third convolutional layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Train the CNN model"""
    print("📊 Loading training data...")
    X, y = load_collected_data()
    
    if X is None or y is None:
        print("❌ Cannot train without data. Please collect data first.")
        return None, None
    
    # Reshape X for CNN (add channel dimension)
    X = X.reshape(-1, 64, 64, 1)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"📈 Training samples: {X_train.shape[0]}")
    print(f"📊 Test samples: {X_test.shape[0]}")
    
    # Create model
    model = create_cnn_model(input_shape=(64, 64, 1), num_classes=len(np.unique(y_encoded)))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Train model
    print("🚀 Training CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n📊 Model Evaluation:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Save model
    model.save('cnn_model.h5')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    print(f"\n💾 Model saved as 'cnn_model.h5'")
    print(f"💾 Label encoder saved as 'label_encoder.pkl'")
    
    # Plot training history
    plot_training_history(history)
    
    return model, label_encoder

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def collect_data_from_webcam():
    """Helper function to collect data via webcam"""
    import cv2
    
    print("📸 Starting data collection...")
    print("Press 's' to save image, 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    letter = input("Enter letter to collect (A-Z): ").upper()
    if len(letter) != 1 or not letter.isalpha():
        print("❌ Invalid letter")
        return
    
    # Create directory
    os.makedirs(f'dataset/collected_images/{letter}', exist_ok=True)
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display frame
        cv2.putText(frame, f'Letter: {letter}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Collected: {count}', (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Press "s" to save, "q" to quit', (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand area
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 3
        cv2.rectangle(frame, 
                     (center_x - size, center_y - size),
                     (center_x + size, center_y + size),
                     (0, 255, 0), 2)
        
        cv2.imshow('Collect Data', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save image
            roi = frame[center_y-size:center_y+size, center_x-size:center_x+size]
            if roi.size > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = f'dataset/collected_images/{letter}/{timestamp}.jpg'
                cv2.imwrite(filename, roi)
                count += 1
                print(f"📁 Saved image {count}: {filename}")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Collected {count} images for letter {letter}")

if __name__ == '__main__':
    print("🤖 Sign Language CNN Model Training")
    print("=" * 50)
    print("1. Collect data from webcam")
    print("2. Train model with collected data")
    print("3. Test model")
    print("4. Exit")
    
    while True:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            collect_data_from_webcam()
        elif choice == '2':
            model, encoder = train_model()
            if model is not None:
                print("\n🎉 Model training completed!")
        elif choice == '3':
            # Test with sample image
            print("Testing functionality...")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")