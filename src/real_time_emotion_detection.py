import numpy as np
import tensorflow as tf
import cv2

# Load the saved model
model = tf.keras.models.load_model('models/final_emotion_detection_model.keras')

# Function to preprocess and predict emotion from an image
def predict_emotion(image):
    # Preprocess the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    normalized_image = resized_image / 255.0
    reshaped_image = normalized_image.reshape(1, 48, 48, 1)
    
    # Predict the emotion
    predictions = model.predict(reshaped_image)
    emotion_index = np.argmax(predictions)
    
    # Map the index to the corresponding emotion
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotions[emotion_index]
    
    print(f"Predictions: {predictions}")
    print(f"Predicted emotion: {predicted_emotion} with confidence {predictions[0][emotion_index]}")
    
    return predicted_emotion

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the webcam feed.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Predict the emotion
    emotion = predict_emotion(frame)
    
    # Display the resulting frame with the predicted emotion
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()















































