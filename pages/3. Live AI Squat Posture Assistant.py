# cd C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\Capstone Projects\Streamlit

# conda prompt: streamlit run app.py

import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder


BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_easy

st.title('AI Squat Training Assistant')
st.subheader("The system will analyze your body posture and keep track of the number of squats you perform. \n Instructions:\n 1. Press start to start the machine camera.\n 2. Move sufficiently far and side face camera.\n 3. Press stop when you finish doing squats.")


thresholds = get_thresholds_easy()

live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
# Initialize face mesh solution
pose = get_mediapipe_pose()


def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="rgb24")  # Decode and get RGB frame
    frame, _ = live_process_frame.process(frame, pose)  # Process frame
    return av.VideoFrame.from_ndarray(frame, format="rgb24")  # Encode and return BGR frame




output_video_file = f'output_live.mp4'

def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(output_video_file)
  
    
    
ctx = webrtc_streamer(
                        key="Squats-pose-analysis",
                        video_frame_callback=video_frame_callback,
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this config
                        media_stream_constraints={"video": {"width": {'min':600, 'ideal':600}}, "audio": False},
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
                        out_recorder_factory=out_recorder_factory
                    )


