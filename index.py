import streamlit as st

from multipage import MultiPage
from inference import MLP, CNN
import inference
import mlp_graphs
import cnn_graphs
import ann_graphs
import resnet_graphs
#import tb
#streamlit_tensorboard
#tensorflow
#tensorboard
#tensorboardX
import train_acc
import test_acc

# Create an instance of the app 
st.set_page_config(
    page_title="Group-50 CS5242", layout="wide", page_icon="./images/icon.png"
)
app = MultiPage()

# Title of the main page
def introduction():
    st.title("**Welcome to CS5242 GROUP-50 Deployment 🧪**")
    st.subheader(
        """
        This is a place where you can get all our interactive plots and results.
        """
    )

    st.markdown(
        """
    - 🗂️ Choose an app from the sidebar in the left
    - ⚙️ Scroll through the app page and checkout the plots
    - 📉 Hover over the plots to get interactive data
    - 🩺 Change the size, precision and other aspects of the plots using options present beside each plot.
    -----
    """
    )
# Add all your applications (pages) here

introduction()
app.add_page("Using Model for Prediction", inference.app)
app.add_page("MLP Results", mlp_graphs.app)
app.add_page("CNN Results", cnn_graphs.app)
app.add_page("ANN Results", ann_graphs.app)
app.add_page("RESNET Results", resnet_graphs.app)
app.add_page("Training Accuracy and Loss Comparison", train_acc.app)
app.add_page("Test Accuracy and Loss Comparison", test_acc.app)
#app.add_page("Tensorboard",tb.app)

# The main app
app.run()