import tensorflow as tf
import cv2
import numpy as np

# Load the SavedModel
model = tf.keras.layers.TFSMLayer('./saved_model', call_endpoint='serving_default')

# Load the labels
class_names = open("./labels.txt", "r").readlines()

# Open video file or webcam (use 0 for webcam, or replace with the video path)
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Resize the frame
    image = cv2.resize(frame, (224, 224))

    # Preprocess the image
    image_np = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1  # Normalize the image

    # Perform inference
    predictions = model(image_np)
    output_key = list(predictions.keys())[0]  # Adjust based on model output key
    predictions = predictions[output_key].numpy()
    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    # Display the result
    cv2.putText(frame, f"{class_name.strip()}: {confidence_score:.2%}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Video', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
