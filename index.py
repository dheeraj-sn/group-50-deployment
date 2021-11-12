import streamlit as st

from multipage import MultiPage
from inference import MLP
import inference
import plotlytry

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("CS5242 GROUP-50 Deployment")

# Add all your applications (pages) here
app.add_page("Inference page", inference.app)
app.add_page("Plot Try Page", plotlytry.app)

# The main app
app.run()