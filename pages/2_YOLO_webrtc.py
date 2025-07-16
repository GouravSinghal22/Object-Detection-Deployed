import streamlit as st 
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration
import av
import cv2
from yolo_predictions import YOLO_Pred

# load yolo model
yolo = YOLO_Pred('./models/best.onnx',
                 './models/data.yaml')


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Resize to speed up inference
    img_resized = cv2.resize(img, (416, 416))

    # YOLO prediction
    pred_img = yolo.predictions(img_resized)

    # Resize back to display size
    pred_img = cv2.resize(pred_img, (img.shape[1], img.shape[0]))

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video":True,"audio":False},
                rtc_configuration=RTC_CONFIGURATION)

