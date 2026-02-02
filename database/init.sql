-- Create database if not exists
CREATE DATABASE IF NOT EXISTS sign_language_db;
USE sign_language_db;

-- Table for storing users
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing gestures (A-Z)
CREATE TABLE IF NOT EXISTS gestures (
    gesture_id INT AUTO_INCREMENT PRIMARY KEY,
    sign_label CHAR(1) UNIQUE NOT NULL,
    meaning TEXT NOT NULL,
    sample_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert all letters A-Z
INSERT IGNORE INTO gestures (sign_label, meaning) VALUES
('A', 'Letter A - Fist with thumb on side'),
('B', 'Letter B - Flat hand, fingers together'),
('C', 'Letter C - Curved hand shape'),
('D', 'Letter D - Index finger pointing up'),
('E', 'Letter E - Fist with thumb across fingers'),
('F', 'Letter F - OK sign with thumb and index'),
('G', 'Letter G - Index finger pointing sideways'),
('H', 'Letter H - Index and middle finger pointing sideways'),
('I', 'Letter I - Pinky finger up'),
('J', 'Letter J - Pinky finger drawing J shape'),
('K', 'Letter K - Index and middle finger up, thumb on middle'),
('L', 'Letter L - Index and thumb forming L'),
('M', 'Letter M - Thumb under three fingers'),
('N', 'Letter N - Thumb under two fingers'),
('O', 'Letter O - Fingers making O shape'),
('P', 'Letter P - Index finger and thumb forming P'),
('Q', 'Letter Q - Index finger and thumb pointing down'),
('R', 'Letter R - Index and middle finger crossed'),
('S', 'Letter S - Fist with thumb across fingers'),
('T', 'Letter T - Fist with thumb between index and middle'),
('U', 'Letter U - Index and middle finger up together'),
('V', 'Letter V - Peace sign'),
('W', 'Letter W - Three fingers up'),
('X', 'Letter X - Index finger bent'),
('Y', 'Letter Y - Thumb and pinky out'),
('Z', 'Letter Z - Index finger drawing Z');

-- Table for predictions history
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    predicted_letter CHAR(1) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    input_type ENUM('webcam', 'image', 'video') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (predicted_letter) REFERENCES gestures(sign_label)
);

-- Table for collected training data
CREATE TABLE IF NOT EXISTS training_samples (
    sample_id INT AUTO_INCREMENT PRIMARY KEY,
    sign_label CHAR(1) NOT NULL,
    image_path VARCHAR(500),
    landmarks TEXT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sign_label) REFERENCES gestures(sign_label)
);

-- Table for word formations
CREATE TABLE IF NOT EXISTS words (
    word_id INT AUTO_INCREMENT PRIMARY KEY,
    word_text VARCHAR(100) NOT NULL,
    letters_sequence TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_training_samples_label ON training_samples(sign_label);
CREATE INDEX idx_words_text ON words(word_text);

-- Create a view for statistics
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    MAX(timestamp) as latest_prediction,
    input_type,
    COUNT(DISTINCT predicted_letter) as unique_letters
FROM predictions
GROUP BY input_type;