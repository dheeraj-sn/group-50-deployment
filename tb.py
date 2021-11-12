from streamlit_tensorboard import st_tensorboard

def app():
    st_tensorboard(logdir="./logs", port=6006, width=1080)