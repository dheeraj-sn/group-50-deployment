import streamlit as st

from multipage import MultiPage
from inference import MLP, CNN
import inference
import mlp_graphs
import tb
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
        This is a place where you can get all our interactive plot and results.
        """
    )

    st.markdown(
        """
    - 🗂️ Choose an app from the sidebar in the left
    - ⚙️ Scroll through the app page and checkout the plots
    - 📉 Hover over the plots to get interactive data
    - 🩺 Change the size, precision and other aspects of the plots using option present beside each plot.
    -----
    """
    )
# Add all your applications (pages) here

introduction()
app.add_page("Using Model for Prediction", inference.app)
app.add_page("MLP Results", mlp_graphs.app)
app.add_page("Training Accuracy Comparison", train_acc.app)
app.add_page("Test Accuracy Comparison", test_acc.app)
app.add_page("Tensorboard",tb.app)

# The main app
app.run()