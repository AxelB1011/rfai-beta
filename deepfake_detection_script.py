import os
import cv2
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications import MobileNetV2 
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from pytube import YouTube

# Path to the folder containing video frames
VIDEO_FRAMES_FOLDER = 'video_frames'
import torch

# Load the PyTorch model
def load_pytorch_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return model

def load_video_frames(video_path):
    # # Create folder to store frames if it doesn't exist
    # os.makedirs(VIDEO_FRAMES_FOLDER, exist_ok=True)
    
    yt = YouTube(video_path)
    
    # Get the highest resolution video stream
    stream = yt.streams.get_highest_resolution()
    
    # Open video stream using OpenCV
    cap = cv2.VideoCapture(stream.url)

    # # Open the video file
    # cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Read and save each frame
    for i in range(frame_count):
        success, frame = cap.read()
        if success:#demo
            # frame_path = os.path.join(VIDEO_FRAMES_FOLDER, f"frame_{i}.jpg")
            # cv2.imwrite(frame_path, frame)
            # frames.append(frame_path)

            # Append the processed frame to the list
            frames.append(frame)
            # print(i)

    cap.release()
    return frames

# def preprocess_frame(frame_path):
#     # Load and preprocess frame
#     # img = tf.keras.preprocessing.image.load_img(frame_path, target_size=(224, 224, 3))

#     # print(frame_path, frame_path.shape)
#     img = tf.keras.preprocessing.image.smart_resize(frame_path, (224, 224))
#     # print(img.shape)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
#     return img_array

def preprocess_frame(frame):
    # Resize frame to (224, 224)
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    frame_normalized = frame_rgb / 255.0
    
    # Convert to torch tensor and add batch dimension
    frame_tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    
    return frame_tensor

# def create_model(num_classes):
#     # Load pre-trained MobileNetV2 model
#     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), alpha=1.4)
    
#     # Freeze all layers except the last one
#     for layer in base_model.layers:
#         layer.trainable = False
    
#     # Add custom classification layers
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
    
#     # Create model
#     model = Model(inputs=base_model.input, outputs=predictions)
#     return model

# def detect_deepfake(video_path):
#     # Load video frames
#     frames = load_video_frames(video_path)
    
#     # Preprocess frames
#     preprocessed_frames = [preprocess_frame(frame) for frame in frames]
#     preprocessed_frames = np.array(preprocessed_frames)
    
#     # Create deepfake detection model
#     num_classes = 2  # Real and fake
#     model = create_model(num_classes)
    
#     # # Load weights
#     # model.load_weights('deepfake_detection_weights.h5')
    
#     # Make predictions
#     predictions = model.predict(preprocessed_frames)
#     # print("Preds: ", predictions)
#     # Aggregate predictions
#     mean_prediction = np.mean(predictions, axis=0)
#     # print("Mean pred: ", mean_prediction)
#     # Get label
#     label = 'Real' if mean_prediction[0] > mean_prediction[1] else 'Fake'
    
#     return label

def detect_deepfake(video_path, pytorch_model):
    frames = load_video_frames(video_path)
    preprocessed_frames = [preprocess_frame(frame) for frame in frames]

    # Convert preprocessed frames to torch tensor
    preprocessed_frames = torch.tensor(preprocessed_frames, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions = pytorch_model(preprocessed_frames)
    
    # Convert predictions to numpy array
    predictions = predictions.numpy()
    
    # Aggregate predictions
    mean_prediction = np.mean(predictions, axis=0)
    
    # Get label
    label = 'Real' if mean_prediction[0] > mean_prediction[1] else 'Fake'
    
    return label


# if __name__ == "__main__":
#     # video_path = 'uploads/_mjzC5RJMEk.mp4'
#     video_path = 'https://www.youtube.com/watch?v=_mjzC5RJMEk'
#     detection_result = detect_deepfake(video_path)
#     # detection_result = detection_result[len(detection_result)-4:len(detection_result)]
#     print("Detection Result:", detection_result)

if __name__ == "__main__":
    video_path = 'https://www.youtube.com/watch?v=_mjzC5RJMEk'
    
    # Load PyTorch model
    model_path = '/Users/gopal/Desktop/rfai/RealForensics/stage2/weights/realforensics_allbutdf.pth'
    pytorch_model = load_pytorch_model(model_path)
    
    # Perform detection
    detection_result = detect_deepfake(video_path, pytorch_model)
    
    print("Detection Result:", detection_result)

