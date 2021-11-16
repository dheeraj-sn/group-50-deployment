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

    mlp_train_metrics = mlp_metrics[mlp_metrics["kind"]=="train"]
    cnn_train_metrics = cnn_metrics[cnn_metrics["kind"]=="train"]
    ann_train_metrics = ann_metrics[ann_metrics["kind"]=="train"]
    resnet_train_metrics = resnet_metrics[resnet_metrics["kind"]=="train"]
    metrics_train = [mlp_train_metrics,cnn_train_metrics,ann_train_metrics,resnet_train_metrics]
    
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
        This is a place where we have presented the training accuracy and loss comparison of different models.
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
        
        
        """
        colors = [colormap(i) for i in np.linspace(0, 1,5)]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.rcParams['legend.fontsize'] = 15
        l = ["MLP","CNN", "ANN", "RESNET"]
        fig, axs = plt.subplots(ncols=1,figsize=(10,8))
        for i in range(len(metrics_train)):
            sns.lineplot(x="epoch",y="loss",data=metrics_train[i],color=colors[i], label=l[i])
        axs.set_xlim([0,300])
        
        st.pyplot(fig=fig)
        """
        
    with col2:
        f1 = go.Figure()
        
        colormap = plt.cm.gist_ncar 
        colors = [colormap(i) for i in np.linspace(0, 1,5)]
        l = ["MLP","CNN", "ANN", "RESNET"]
        for i in range(len(metrics_train)):
            f1.add_trace(go.Scatter(x=metrics_train[i]["epoch"], y=metrics_train[i]["accuracy"],mode='lines',name=l[i]))
        
        f1.update_layout(xaxis_range=[0,300], width=600, height=400, title="Accuracy Comparison",xaxis_title="Epoch", yaxis_title="Accuracy", legend_title="Model",
            font=dict(
                size=18,
                color="RebeccaPurple"
            ))        
           
        st.plotly_chart(f1,use_container_width=True)
        
        
        
        f2 = go.Figure()
        for i in range(len(metrics_train)):
            f2.add_trace(go.Scatter(x=metrics_train[i]["epoch"], y=metrics_train[i]["loss"],mode='lines',name=l[i]))
        
        f2.update_layout(xaxis_range=[0,300], width=600, height=400, title="Training Loss Comparison",xaxis_title="Epoch", yaxis_title="Loss", legend_title="Model",
            font=dict(
                size=18,
                color="RebeccaPurple"
            ))        
           
        st.plotly_chart(f2,use_container_width=True)
        
        """
        colormap = plt.cm.gist_ncar 
        colors = [colormap(i) for i in np.linspace(0, 1,5)]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.rcParams['legend.fontsize'] = 15
        l = ["MLP","CNN", "ANN", "RESNET"]
        fig, axs = plt.subplots(ncols=1,figsize=(10,8))
        for i in range(len(metrics_train)):
            sns.lineplot(x="epoch",y="accuracy",data=metrics_train[i],color=colors[i], label=l[i])
        axs.set_xlim([0,300])
        
        st.pyplot(fig=fig)
        """
    st.markdown(
            """
            -----

            - The RESNET model being pretrained with a very deep architecture displays a strong increase in the accuracy of the model and can be seen to have a very good performance on the training data
            - Similar to RESNET, the CNN model also performs very well on the training data and has a very high accuracy.
            - The MLP model with only linear layers and the least number of parameters is not able to perform as good as the CNN network.
            - The ANN model does not seem to perform very well on the emotion's training data, which is apparent by its low accuracy. Our ANN model will need more data and longer training times to have a better accuracy.

            -----
            """
        )
    
        