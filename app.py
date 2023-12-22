import streamlit as st
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model_architecture import LeNet5,LeNet5_BN

def make_prediction(img,model_version,label_dict):
    # print(img)
    # img=Image.open(img)
    transform_inference_PIL = transforms.Compose([transforms.Resize((32,32))])
    # Convert the PIL image to Torch tensor 
    img_tensor = transform_inference_PIL(img)
    img_tensor = transforms.functional.to_tensor(img_tensor)
    img_tensor=img_tensor.unsqueeze(0)

    model_version.eval()
    pred_lab=model_version(img_tensor)
    # print(pred_lab)
    pred_lab=torch.argmax(pred_lab).cpu().item()
    
    pred_label=label_dict[str(pred_lab+1)]
    return img_tensor,pred_label

st.title("Let's check who is in the image!")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick a model you'd like to use",
    ("Model 1 - LeNet5", 
     "Model 2 - LeNet5 with Batch Normalization") 
)

if choose_model=="Model 1 - LeNet5":
    model=LeNet5()
    model.load_state_dict(torch.load('LeNet5_model_20_12_2023_16_54_17.pth'))
else:
    model=LeNet5_BN()
    model.load_state_dict(torch.load('LeNet5_BN_model_20_12_2023_17_00_14.pth'))

label_name = {'1': 'airplane', '2':'bird', '3':'car', '4':'cat', '5':'deer', '6':'dog', '7':'horse', '8':'monkey', '9':'ship', '10':'truck'}
if st.checkbox("Show Classes"):
    st.write(label_name)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image you wish to classify",
                                 type=["png", "jpeg", "jpg"])

session_state = st.session_state
pred_button = st.button("Predict")

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = Image.open(uploaded_file)
    st.image(session_state.uploaded_image, use_column_width=True)
    # print(type(session_state.uploaded_image))
session_state.pred_button = False
if pred_button:
    session_state.pred_button = True

if session_state.pred_button:
    session_state.image,session_state.pred_class=make_prediction(session_state.uploaded_image,model,label_name)
    st.write("Prediction:",session_state.pred_class)
