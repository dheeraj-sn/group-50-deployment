import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import itertools

def app():
    metrics = pd.read_csv("./csv/resnet.csv")   
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
        This is a place where you can get the interactive plots for RESNET.
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
        
        - RESNET training and test loss plots are presented here. Hover on them to get the loss for each epoch.
        - RESNET training and test accuracy plots are presented here. Hover on them to get the accuracy for each epoch.
        - Choose options from the top right of each plot to get more information
        - The training loss of our model consistently decrease as the training progress. We noticed that the loss converges at around 50 epochs itself. 
        - The training accuracy also consistently increases and reaches near 100%. This means the model has learned the training data too well.
        - The loss plot for the test data seems to be increaseing. Also the variability of the test loss plot is very high across epochs. Unlike MLP and CNN, even though the loss for the test data is increasing, the accuracy on the test data is more or less stable around 32-35%. Generally the test loss and test accuracy should have an inverse relationship, i.e. with increasing test accuracy, the test loss should decrease.
        -----
        """
        )
        plt.rcParams['axes.grid'] = False
        resnet_cm = np.load("./npdata/resnet_cm.npy")
        cm = resnet_cm
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