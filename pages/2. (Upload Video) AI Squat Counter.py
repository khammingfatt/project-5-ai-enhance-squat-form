# directory: C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\Capstone Projects\Streamlit\pages
# Conda Prompt for Streamlit: streamlit run upload_video.py

# --------------------------------------------------------Summary of the Model---------------------------------------------------------

# The objective of this streamlit file is to import and deploy a pre-trained machine learning model to 
# (i) detect 2 states - "Rest" and "Down"
# (ii) increase the counter by 1 when the state completes a cycle of Rest -> Down -> Rest

# --------------------------------------------------------Import Libraries---------------------------------------------------------

import cv2
import mediapipe as mp
import streamlit as st
import pickle
import numpy as np
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from sklearn.preprocessing import LabelEncoder
import os

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
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for pose detection

        # Process the frame and get the pose landmarks
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(102, 204, 255), thickness=4, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4)
            )

            landmarks = self.get_landmarks(results.pose_landmarks)
            class_label, proba = self.predict(landmarks)
            self.draw_class_box(frame, class_label)
            self.draw_proba_box(frame, proba)
            self.draw_count_box(frame, class_label, proba)

        return frame

# --------------------------------------------------Getting Coordinates of Landmarks as Input----------------------------------------------
    
    def get_landmarks(self, pose_landmarks):
        landmarks = []
        for landmark in pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(landmarks)

# --------------------------------------------------Predict Class and Probability--------------------------------------------------------

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
        cv2.rectangle(image, (10, 150), (620, 210), (255, 255, 255), -1)
        cv2.putText(image, proba_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

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
        cv2.rectangle(image, (10, 80), (300, 140), (255, 255, 255), -1)
        cv2.putText(image, count_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# --------------------------------------------------Display Streamlit Interface--------------------------------------------------------
def main():
    st.title("AI Squat Counter")
    st.subheader("The system will analyze your body posture and keep track of the number of squats you perform. \n Instructions:\n 1. Click on Browse File and Upload your video file. \n 2. The system only accepts mp4, mov and avi file. \n 3. Click on Download Video if you wish to download the video into your device.")

    output_video_file = 'squat_counter_output.mp4'
    
    if os.path.exists(output_video_file):
        os.remove(output_video_file)

# --------------------------------------------------Create Upload Button and define it---------------------------------------------------
    with st.form('Upload', clear_on_submit=True):
        up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
        uploaded = st.form_submit_button("Upload")


    frame_placeholder = st.empty()  # This will hold our video frames
    
    if up_file and uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up_file.read())

        vf = cv2.VideoCapture(tfile.name)

        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

        video_transformer = PoseDetectionTransformer()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # convert frame from BGR to RGB before processing it.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_frame = video_transformer.transform(frame_rgb)

            # Display the processed video frames in real-time
            frame_placeholder.image(out_frame, channels='RGB')

            video_output.write(cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

        vf.release()
        video_output.release()
        tfile.close()

# --------------------------------------------------Create Download Button and Define it--------------------------------------------------

        if os.path.exists(output_video_file):
            with open(output_video_file, 'rb') as op_vid:
                st.download_button('Download Video', data = op_vid, file_name='squat_counter_output.mp4')

if __name__ == "__main__":
    main()
