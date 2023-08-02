# cd C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\Capstone Projects\Streamlit
# conda prompt: streamlit run app.py

# --------------------------------------------------------Summary of the Model 1---------------------------------------------------------

# The objective of this streamlit file is to import and deploy a pre-trained machine learning model to 
# (i) detect 2 states - "Rest" and "Down"
# (ii) increase the counter by 1 when the state completes a cycle of Rest -> Down -> Rest

# --------------------------------------------------------Import Libraries---------------------------------------------------------
import os
import cv2
import mediapipe as mp
import streamlit as st
import pickle
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------Get the directory path of the current script----------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------Initialising All Variables------------------------------------------------------
class PoseDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Setting up relative path to the two files
        pkl_file_path = os.path.join(current_dir, '..', 'data')
        
        # Load the squat_2 trained model
        with open(os.path.join(pkl_file_path, 'squat_2.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        
        # Load the label encoder
        with open(os.path.join(pkl_file_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize counter and current_stage variables
        self.counter = 0
        self.current_stage = "Rest"

# --------------------------------------------------------Process Videos---------------------------------------------------------------
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")  # Convert frame to BGR format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for pose detection

        # Process the frame and get the pose landmarks
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(102, 204, 255), thickness=4, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4)
            )
            
            landmarks = self.get_landmarks(results.pose_landmarks)
            class_label, proba = self.predict(landmarks)
            self.draw_class_box(image, class_label)
            self.draw_proba_box(image, proba)
            self.draw_count_box(image, class_label, proba)

        return image

# --------------------------------------------------Getting Coordinates of Landmarks as Input----------------------------------------------
    def get_landmarks(self, pose_landmarks):
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks)

    
# --------------------------------------------------Predict Class and Probability---------------------------------------------------------
    def predict(self, landmarks):
        class_label_numerical = self.model.predict([landmarks])[0]
        class_label = self.label_encoder.inverse_transform([class_label_numerical])[0]
        proba = self.model.predict_proba([landmarks])[0]
        return class_label, proba

    
# --------------------------------------------------Display the Textboxes---------------------------------------------------------------
    def draw_class_box(self, image, class_label):
        class_text = f"Class: {class_label}"
        cv2.rectangle(image, (10, 10), (300, 70), (255, 255, 255), -1)
        cv2.putText(image, class_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    def draw_proba_box(self, image, proba):
        proba_text = "Proba: "
        for i, prob in enumerate(proba):
            class_label = self.label_encoder.classes_[i]
            prob_text = f"{class_label}: {prob:.2f}"
            proba_text += prob_text + " "
        cv2.rectangle(image, (10, 80), (620, 140), (255, 255, 255), -1)
        cv2.putText(image, proba_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def draw_count_box(self, image, class_label, proba):
        try: 
            if class_label == "Rest" and proba[proba.argmax()] > 0.7: 
                self.current_stage = "Rest" 
            elif self.current_stage == "Rest" and class_label == "Down" and proba[proba.argmax()] > 0.7:
                self.current_stage = "Down" 
                self.counter += 1 

        except Exception as e: 
            print(e)

        count_text = f"Count: {self.counter}"
        cv2.rectangle(image, (350, 10), (550, 70), (255, 255, 255), -1)
        cv2.putText(image, count_text, (360, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        
# --------------------------------------------------Display the Video---------------------------------------------------------------
    def on_video_frame(self, frame):
        processed_frame = self.transform(frame)
        self.results_writer.send(processed_frame)  # Send the processed frame for display


        
# --------------------------------------------------Display Streamlit Interface--------------------------------------------------------        
def main():
    st.title("Live AI Squat Counter")
    st.subheader("The system will analyze your body posture and keep track of the number of squats you perform. \n Instructions:\n 1. Press start to start the machine camera.\n 2. Move sufficiently far and side face camera.\n 3. Press stop when you finish doing squats.")

    webrtc_ctx = webrtc_streamer(
        key="pose-detection",
        video_transformer_factory=PoseDetectionTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )


if __name__ == "__main__":
    main()
