import streamlit as st
import os
from os import listdir
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets

import pathlib
import datetime;

from os import listdir

class MLP(nn.Module):
    def __init__(self,input_size, output_size):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size,512,bias=True)
        self.layer2 = nn.Linear(512,512,bias=True)
        self.layer3 = nn.Linear(512,256,bias=True)
        self.layer4 = nn.Linear(256,256,bias=True)
        self.layer5 = nn.Linear(256,64,bias=True)
        self.layer6 = nn.Linear(64, output_size,bias=True)
        self.dropout1 = nn.Dropout(0.50)
        self.dropout2 = nn.Dropout(0.50)
        self.dropout3 = nn.Dropout(0.50)
        self.dropout4 = nn.Dropout(0.50)
        
    def forward(self,x):
        x = self.flatten(x)
        
        y = self.layer1(x)
        y = self.dropout1(y)
        y = torch.relu(y)
        
        y = self.layer2(y)
        y = self.dropout2(y)
        y = torch.relu(y)
        
        y = self.layer3(y)
        y = self.dropout3(y)
        y = torch.relu(y)
        
        y = self.layer4(y)
        y = self.dropout4(y)
        y = torch.relu(y)
        
        y = self.layer5(y)
        y = torch.relu(y)
        
        y = self.layer6(y)
        
        return y

current_path = os.path.abspath("./")
device = torch.device('cpu')
mlp_model_path = current_path + "/model/mlp.pt"
mlp_model = torch.load(mlp_model_path,map_location=device)
mlp_model = mlp_model.to(device)
class_names = ['angry', 'disgusted', 'happy', 'sad', 'surprised']
img_size = (32, 32)
train_mean = 0.4710924029350281
train_std = 0.26737314462661743
data_transforms = T.Compose([T.Resize(img_size),T.ToTensor()])

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {
                visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("UPLOAD IMAGE FILE", type=["png","jpg","svg"])
if uploaded_file is not None:
    print("0")
else:  
    def user_input_features():
        data_dir = pathlib.Path(current_path + '/model/images/')
        files = []
        for file in os.listdir(data_dir):
            files.append(current_path + "/model/images/" + file)
        island = st.sidebar.selectbox('Images',files)
        data = {'images': island}     
        return data
    image_location_and_name=user_input_features()
    st.image(str(image_location_and_name['images']))

    img = PIL.Image.open(str(image_location_and_name['images']))
    img_array = data_transforms(img)
    img_array = img_array.unsqueeze(0)
    img_array = img_array.to(device)
    
    score = mlp_model.forward( (img_array-train_mean)/train_std )
    score = torch.softmax(score, dim=1)
    result="""
    MLP model says that face is {}, with {:.2f} percent confidence.
    """.format(class_names[torch.argmax(score[0])], 100*score[0,torch.argmax(score[0])])
    st.write(result)

if uploaded_file is not None:
    ts = datetime.datetime.now().timestamp()
    file_name=str(ts)+'.png'
    image_location_and_name=current_path+ '/tempDir/'+str(ts)+'.png'
    image_file=pathlib.Path(image_location_and_name)
    with open(os.path.join(current_path+"/tempDir/",file_name),"wb") as f:
         f.write(uploaded_file.getbuffer())

    img = PIL.Image.open(str(image_location_and_name))
    
    img_array = data_transforms(img)
    img_array = img_array.unsqueeze(0)
    img_array = img_array.to(device)
    
    score = mlp_model.forward( (img_array-train_mean)/train_std )
    score = torch.softmax(score, dim=1)

    result="""
    MLP model says that face is {}, with {:.2f} percent confidence.
    """.format(class_names[torch.argmax(score[0])], 100*score[0,torch.argmax(score[0])])
    st.subheader(result)
    st.image(uploaded_file, channels="BGR")
    uploaded_file.seek(0)
    
else:
    st.write('Upload image file from the left to check. Currently using example images.')
    st.write()
