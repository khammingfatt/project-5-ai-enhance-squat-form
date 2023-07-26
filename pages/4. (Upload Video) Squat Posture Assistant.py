# directory: C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\Capstone Projects\Streamlit\pages
# Conda Prompt for Streamlit: streamlit run upload_video.py

# --------------------------------------------------------Summary of the Model---------------------------------------------------------

# The objective of this streamlit file is to import and deploy a pre-trained machine learning model to 
# (i) detect 2 states - "Rest" and "Down"
# (ii) increase the counter by 1 when the state completes a cycle of Rest -> Down -> Rest

# --------------------------------------------------------Import Libraries---------------------------------------------------------

import av
import cv2
import os
import sys
import streamlit as st
import tempfile
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder

# --------------------------------------------------------Define File Path---------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

# --------------------------------------------------------Define Self Customised Libraries------------------------------------------

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_easy

# --------------------------------------------------------Interface of the Streamlit Page------------------------------------------

st.title('(Upload Video) AI Squat Training Assistant')
st.subheader("The system will analyze your body posture and give feedback to your squat postures. \n Instructions:\n 1. Click on Browse File and Upload your video file. \n 2. The system only accepts mp4, mov and avi file. \n 3. Click on Download Video if you wish to download the video into your device.")

thresholds = get_thresholds_easy()
upload_process_frame = ProcessFrame(thresholds=thresholds) # refer to the threshold document for angles
pose = get_mediapipe_pose() # Initialize face mesh solution


with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
    uploaded = st.form_submit_button("Upload")


stframe = st.empty()

ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'
warning_str = '<p style="font-family:Helvetica; font-weight: bold; color: Red; font-size: 17px;">Please Upload a Video first!!!</p>'

warn = st.empty()

download_button = st.empty()

# --------------------------------------------------------Removal of the original file to avoid conflict----------------------------

output_video_file = f'output_recorded.mp4'

if os.path.exists(output_video_file):
    os.remove(output_video_file)

# --------------------------------------------------------Interface of the Streamlit Page------------------------------------------

    
if up_file and uploaded:
    
    download_button.empty()
    tfile = tempfile.NamedTemporaryFile(delete=False)

    try:
        warn.empty()
        tfile.write(up_file.read())

        vf = cv2.VideoCapture(tfile.name)

        # ---------------------  Write the processed video frame. --------------------
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
        # -----------------------------------------------------------------------------

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break

            # convert frame from BGR to RGB before processing it.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_frame, _ = upload_process_frame.process(frame, pose)
            stframe.image(out_frame)
            video_output.write(out_frame[...,::-1])

        
        vf.release()
        video_output.release()
        stframe.empty()
        tfile.close()
    
    except AttributeError:
        warn.markdown(warning_str, unsafe_allow_html=True)
        
# --------------------------------------------------------Create a download button------------------------------------------        
        
download = None

if 'download' not in st.session_state:
    st.session_state['download'] = False


if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data = op_vid, file_name='output_recorded.mp4')
    
    if download:
        st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()