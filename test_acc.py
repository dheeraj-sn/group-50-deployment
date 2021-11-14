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
    
    
    fig_test = make_subplots(rows=2, cols=2, specs=[[{'type' : 'indicator'}, {'type' : 'indicator'}], [{'type' : 'indicator'},{'type' : 'indicator'}]])
    fig_test.add_trace(go.Indicator(
        mode="gauge+number",
        title={"text": f"Test Accuracy MLP"},
        value = max(mlp_test_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=1, col=1
    )
    fig_test.add_trace(go.Indicator(
        mode="gauge+number",
        title={"text": f"Test Accuracy CNN"},
        value = max(cnn_test_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=1, col=2
    )
    fig_test.add_trace(go.Indicator(
        mode="gauge+number",
        title={"text": f"Test Accuracy ANN"},
        value = max(ann_test_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=2, col=1
    )
    fig_test.add_trace(go.Indicator(
        mode="gauge+number",
        title={"text": f"Test Accuracy RESNET"},
        value = max(resnet_test_metrics["accuracy"]),
        delta = {'reference': 0},
        gauge = {
            'axis': {'visible': True, 'range': [0,100]}},
        domain = {'row': 0, 'column': 0}),
        row=2, col=2
    )

    fig_test.update_layout(height=600, width=600)
    st.subheader(
        """
        This is a place where we have presented the test accuracy comparison of different models.
        """
    )
    
    col1, col2 = st.columns((1, 1))
    with col1:
        st.plotly_chart(fig_test, use_container_width=True)
    with col2:
        st.markdown(
            """
            -----
            
            - RESNET has a low test accuracy.
            - CNN, MLP, ANN have very similar test accuracies.
        
            -----
            """
        )