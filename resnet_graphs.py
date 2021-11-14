import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def app():
    metrics = pd.read_csv("./csv/resnet.csv")   
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
        
    fig = px.line(metrics, x="epoch", y="loss", color='kind', title="Loss Plot")
    fig1 = px.line(metrics, x="epoch", y="accuracy", color='kind', title="Accuracy Plot")
    
    st.subheader(
        """
        This is a place where you can get the interactive loss plots for RESNET.
        """
    )

    col1, col2 = st.columns((1, 1))
    with col1:
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown(
        """
        -----
        
        - RESNET training and test loss plots are presented here. Hover on them to get the loss for each epoch.
        - RESNET training and test accuracy plots are presented here. Hover on them to get the accuracy for each epoch.
        - Choose options from the top right of each plot to get more information
        -----
        """
        )