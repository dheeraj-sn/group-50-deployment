from streamlit_tensorboard import st_tensorboard
import os

def app():
    os.system('python -m tensorflow.tensorboard --logdir=./logs')
    st_tensorboard(logdir="./logs", port=6006, width=1080)