from streamlit_tensorboard import st_tensorboard
import tensorflow
import os

def app():
    os.system('python -m tensorboard --bind_all --host=0.0.0.0 --logdir=' + "./logs")
    st_tensorboard(logdir="./logs", port=6006, width=1080)