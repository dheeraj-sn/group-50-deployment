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
    metrics_test = [mlp_test_metrics,cnn_test_metrics,ann_test_metrics,resnet_test_metrics]
    
    
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

    fig_test.update_layout(height=800, width=800)
    st.subheader(
        """
        This is a place where we have presented the test accuracy comparison of different models.
        """
    )
    
    col1, col2 = st.columns((1, 1))
    with col1:
        st.plotly_chart(fig_test, use_container_width=True)
    with col2:
        f1 = go.Figure()
        
        colormap = plt.cm.gist_ncar 
        colors = [colormap(i) for i in np.linspace(0, 1,5)]
        l = ["MLP","CNN", "ANN", "RESNET"]
        for i in range(len(metrics_test)):
            f1.add_trace(go.Scatter(x=metrics_test[i]["epoch"], y=metrics_test[i]["accuracy"],mode='lines',name=l[i]))
        
        f1.update_layout(xaxis_range=[0,300], width=600, height=400, title="Accuracy Comparison",xaxis_title="Epoch", yaxis_title="Accuracy", legend_title="Model",
            font=dict(
                #family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            ))        
           
        st.plotly_chart(f1,use_container_width=True)
        
        
        
        f2 = go.Figure()
        for i in range(len(metrics_test)):
            f2.add_trace(go.Scatter(x=metrics_test[i]["epoch"], y=metrics_test[i]["loss"],mode='lines',name=l[i]))
        
        f2.update_layout(xaxis_range=[0,300], width=600, height=400, title="Test Loss Comparison",xaxis_title="Epoch", yaxis_title="Loss", legend_title="Model",
            font=dict(
                size=18,
                color="RebeccaPurple"
            ))        
           
        st.plotly_chart(f2,use_container_width=True)
    
    
    
    st.markdown(
        """
        -----
        
        - The RESNET model clearly seems to overfit our dataset, since it has the least accuracy on the test dataset and very high accuracy on the training dataset. This can be attributed to -
        - Small dataset size (We increased the number of images through augmentation)
        - Very deep architecture of the RESNET. For the emotion classification task having 5 classes this may be too deep.
        - The CNN Model performs the best on our dataset both in the training and the testing phase. A shallow network with fewer layers, batch normalization after each convolution operation and using the dropout technique ensured a better result for the model as compared to the others.
        - The MLP and the ANN model perform well on the test set with both the model's having a similar performance.
        - The ANN model has similar training and test accuracies and it would seem that it has not overfit our training data.
    
        -----
        """
    )