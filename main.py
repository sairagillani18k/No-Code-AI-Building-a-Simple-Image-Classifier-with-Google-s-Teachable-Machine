import tensorflow as tf
import cv2
import numpy as np

# Load the SavedModel using TFSMLayer
model = tf.keras.layers.TFSMLayer('/path/to/model.savedmodel', call_endpoint='serving_default')

# Load the labels
class_names = open("/path/to/labels.txt", "r").readlines()

# Open video file or webcam (use 0 for webcam, or replace with the video path)
camera = cv2.VideoCapture('path/to/(inference_video).mp4')

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Resize the frame to the same size as the model's input size (224x224 in this case)
    image = cv2.resize(frame, (224, 224))
    
    # Preprocess the image: convert to numpy array, reshape, and normalize
    image_np = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1  # Normalize the image

    # Perform inference
    predictions = model(image_np)
    
    # Check available output keys and print them
    print(predictions.keys())
    
    # Assuming the correct output key is found, adjust the following line
    output_key = list(predictions.keys())[0]  # You can replace with the exact key once known
    predictions = predictions[output_key].numpy()  
    
    # Continue with the prediction process
    index = np.argmax(predictions)
    class_name = class_names[index]
    confidence_score = predictions[0][index]

    # Display the prediction and confidence score on the frame
    cv2.putText(frame, f"{class_name.strip()}: {confidence_score:.2%}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()
