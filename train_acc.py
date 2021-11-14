import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
def app():
    mlp_metrics = pd.read_csv("./csv/mlp.csv")
    cnn_metrics = pd.read_csv("./csv/cnn.csv")
    ann_metrics = pd.read_csv("./csv/ann.csv")
    resnet_metrics = pd.read_csv("./csv/resnet.csv")

    mlp_test_metrics = mlp_metrics[mlp_metrics["kind"]=="test"]
    cnn_test_metrics = cnn_metrics[cnn_metrics["kind"]=="test"]
    ann_test_metrics = ann_metrics[ann_metrics["kind"]=="test"]
    resnet_test_metrics = resnet_metrics[resnet_metrics["kind"]=="test"]

    mlp_train_metrics = mlp_metrics[mlp_metrics["kind"]=="train"]
    cnn_train_metrics = cnn_metrics[cnn_metrics["kind"]=="train"]
    ann_train_metrics = ann_metrics[ann_metrics["kind"]=="train"]
    resnet_train_metrics = resnet_metrics[resnet_metrics["kind"]=="train"]
    
    
    fig_train = make_subplots(rows=2, cols=2, specs=[[{'type' : 'indicator'}, {'type' : 'indicator'}], [{'type' : 'indicator'},{'type' : 'indicator'}]])
    fig_train.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={"text": f"Train Accuracy MLP"},
        value = max(mlp_train_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=1, col=1
    )
    fig_train.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={"text": f"Train Accuracy CNN"},
        value = max(cnn_train_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=1, col=2
    )
    fig_train.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={"text": f"Train Accuracy ANN"},
        value = max(ann_train_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=2, col=1
    )
    fig_train.add_trace(go.Indicator(
        mode="gauge+number+delta",
        title={"text": f"Train Accuracy RESNET"},
        value = max(resnet_train_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=2, col=2
    )

    fig_train.update_layout(height=800, width=800)
    
    st.subheader(
        """
        This is a place where we have presented the training accuracy comparison of different models.
        """
    )

    #st.markdown(
    #    """
    #- 🗂️ Choose another app from the sidebar in the left in case you want to switch
    #- ⚙️ Choose an example image from the left to check prediction
    #- 🩺 Upload you own image from the left if required. Make sure the image has only 1 person and is focussed on the face.
    #-----
    #"""
    #)
    
    
    col1, col2 = st.columns((1, 1))
    with col1:
        st.plotly_chart(fig_train, use_container_width=True)
    with col2:
        st.markdown(
            """
            -----
            
            - CNN and resnet have very high training accuracy
            - ANN has the lowest accuracy
            - MLP has an intermediate accuracy
        
            -----
            """
        )