import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import itertools


def app():
    metrics = pd.read_csv("./csv/cnn.csv")   
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
        
    f1 = px.line(metrics, x="epoch", y="loss", color='kind', title="Loss Plot")
    fig1 = px.line(metrics, x="epoch", y="accuracy", color='kind', title="Accuracy Plot")
    f1.update_layout(width=600, height=450,font=dict(
                size=18,
                color="RebeccaPurple"))
    
    fig1.update_layout(width=600, height=450,font=dict(
                size=18,
                color="RebeccaPurple"
            ))
            
    st.subheader(
        """
        This is a place where you can get the interactive plots for CNN.
        """
    )

    col1, col2 = st.columns((1, 1))
    with col1:
        st.plotly_chart(f1, use_container_width=True)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown(
        """
        -----
        
        - CNN training and test loss plots are presented here. Hover on them to get the loss for each epoch.
        - CNN training and test accuracy plots are presented here. Hover on them to get the accuracy for each epoch.
        - Choose options from the top right of each plot to get more information
        - The training loss of our model consistently decrease as the training progress. We noticed that the loss converges at around 50 epochs itself. 
        - The training accuracy also consistently increases and reaches near 100%. This means the model has learned the training data very well.
        - The loss plot for the test data is decreasing at first but increases hereafter. Also the variability of the test loss plot is very high across epochs. Just like MLP, even though the loss for the test data is increasing, the accuracy on the test data is also increasing and later it converges. Generally the test loss and test accuracy should have an inverse relationship, i.e. with increasing test accuracy, the test loss should decrease. But here both are increasing. 
        - Accuracy measures whether the model is making right predictions. Loss measure how confident the model is about the predictions.Since the loss and accuracy do not have have an inverse relationship in our case, this means that the model is making good predictions but it is less sure about the predictions it is making. 
        -----
        """
        )
        plt.rcParams['axes.grid'] = False
        cnn_cm = np.load("./npdata/cnn_cm.npy")
        cm = cnn_cm
        classes = ["angry","disgusted","happy","sad","surprised"]
        normalize=False
        cmap=plt.cm.Blues
        title='Confusion matrix'
        
        cm = cm.astype(int)
        fig, ax = plt.subplots()
        aximg = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title)
        fig.colorbar(aximg)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks, classes)
        ax.set_yticks(tick_marks, classes)
        #plt.setp(ax.get_xticklabels())
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        
        #ax.tight_layout()
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')

        #plt.figure(figsize=(10,10))
        st.pyplot(fig=fig)