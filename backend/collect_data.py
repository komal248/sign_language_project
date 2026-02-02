import cv2
import os
from datetime import datetime

def collect_training_data():
    """Collect training data using webcam"""
    
    # Create dataset directory
    os.makedirs("dataset/collected_images", exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("\n" + "="*60)
    print("📸 SIGN LANGUAGE DATA COLLECTION")
    print("="*60)
    print("\n📋 Instructions:")
    print("  1. Enter the letter (A-Z) you want to collect")
    print("  2. Show the sign language gesture in the green box")
    print("  3. Press SPACEBAR to capture image")
    print("  4. Press 'q' to change letter")
    print("  5. Press ESC to exit")
    print("="*60)
    
    current_letter = None
    count = 0
    
    try:
        while True:
            if not current_letter:
                letter = input("\n📝 Enter letter to collect (A-Z) or 'exit' to quit: ").upper().strip()
                
                if letter == 'EXIT':
                    break
                elif len(letter) == 1 and 'A' <= letter <= 'Z':
                    current_letter = letter
                    letter_dir = f"dataset/collected_images/{letter}"
                    os.makedirs(letter_dir, exist_ok=True)
                    count = 0
                    print(f"\n✅ Collecting images for letter: {letter}")
                    print("📸 Press SPACEBAR to capture | 'q' to change letter | ESC to exit")
                else:
                    print("❌ Invalid input. Please enter a single letter A-Z.")
                    continue
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Get frame dimensions
            h, w = frame.shape[:2]
            
            # Display instructions on frame
            cv2.putText(frame, f"LETTER: {current_letter}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"IMAGES: {count}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hand area box (center of screen)
            center_x, center_y = w // 2, h // 2
            box_size = min(w, h) // 3  # Box takes 1/3 of screen
            
            # Draw outer box
            cv2.rectangle(frame, 
                         (center_x - box_size, center_y - box_size),
                         (center_x + box_size, center_y + box_size),
                         (0, 255, 0), 3)
            
            # Draw inner guide
            cv2.rectangle(frame, 
                         (center_x - box_size + 10, center_y - box_size + 10),
                         (center_x + box_size - 10, center_y + box_size - 10),
                         (0, 200, 0), 1)
            
            # Add text guide
            cv2.putText(frame, "SHOW HAND HERE", 
                       (center_x - 100, center_y - box_size - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show controls
            cv2.putText(frame, "SPACE: Capture | Q: Change letter | ESC: Exit", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow("Sign Language Data Collection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC to exit
                break
            elif key == ord('q'):  # Change letter
                print(f"\n📁 Finished collecting for letter {current_letter}")
                print(f"📊 Total images for {current_letter}: {count}")
                current_letter = None
                continue
            elif key == 32:  # SPACE to capture
                # Crop hand area
                roi = frame[center_y-box_size:center_y+box_size, 
                           center_x-box_size:center_x+box_size]
                
                if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
                    # Resize to consistent size
                    roi_resized = cv2.resize(roi, (300, 300))
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = f"dataset/collected_images/{current_letter}/{timestamp}.jpg"
                    
                    # Save with good quality
                    cv2.imwrite(filename, roi_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    count += 1
                    
                    # Show confirmation
                    print(f"  ✅ Saved image {count}: {filename}")
                    
                    # Visual feedback (flash)
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
                    cv2.imshow("Sign Language Data Collection", flash)
                    cv2.waitKey(100)  # Flash for 100ms
                else:
                    print("  ❌ Invalid ROI - try again")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Data collection interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Show summary
        print("\n" + "="*60)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*60)
        
        total_images = 0
        if os.path.exists("dataset/collected_images"):
            letters = [d for d in os.listdir("dataset/collected_images") 
                      if os.path.isdir(os.path.join("dataset/collected_images", d))]
            
            for letter in sorted(letters):
                letter_dir = os.path.join("dataset/collected_images", letter)
                images = [f for f in os.listdir(letter_dir) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    print(f"  Letter {letter}: {len(images):3d} images")
                    total_images += len(images)
        
        print("-"*60)
        print(f"  TOTAL: {total_images} images")
        print("="*60)
        
        # Recommendations
        if total_images > 0:
            print("\n💡 NEXT STEPS:")
            print("  1. Run: python train_cnn.py (to train model)")
            print("  2. Run: python app.py (to start server)")
            print("  3. Open browser: http://localhost:5000")
            print("  4. Test your trained model!")

def check_existing_data():
    """Check existing collected data"""
    if not os.path.exists("dataset/collected_images"):
        print("❌ No dataset found. Starting fresh collection...")
        return 0
    
    print("\n📊 EXISTING DATASET:")
    print("="*40)
    
    total_images = 0
    letters = [d for d in os.listdir("dataset/collected_images") 
               if os.path.isdir(os.path.join("dataset/collected_images", d))]
    
    for letter in sorted(letters):
        letter_dir = os.path.join("dataset/collected_images", letter)
        images = [f for f in os.listdir(letter_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            print(f"  Letter {letter}: {len(images):3d} images")
            total_images += len(images)
    
    print("-"*40)
    print(f"  TOTAL: {total_images} images across {len(letters)} letters")
    print("="*40)
    
    return total_images

if __name__ == "__main__":
    # Check existing data first
    existing_count = check_existing_data()
    
    if existing_count > 0:
        choice = input("\nDo you want to: (1) Add more data, (2) Start fresh, (3) Exit? [1/2/3]: ")
        if choice == '3':
            exit()
        elif choice == '2':
            # Remove existing data
            import shutil
            if os.path.exists("dataset/collected_images"):
                shutil.rmtree("dataset/collected_images")
            print("🗑️  Old dataset removed. Starting fresh...")
    
    # Start data collection
    collect_training_data()