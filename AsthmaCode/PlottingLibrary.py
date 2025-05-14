# from fhir.resources.patient import Patient
import pandas as pd
import numpy as np
import os 
import re
import matplotlib.pyplot as plt
import utils 
import json
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

from collections import Counter
import pickle


# Data over sampling , augmentation libraries
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN

import itertools

import seaborn as sns
from itertools import combinations
from collections import defaultdict
from sklearn.metrics import confusion_matrix



def joint_plot(df, x_col, y_col, title=None, x_label=None, y_label=None,log=None):
    """
    Create a scatter plot of two columns from a Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        title (str): Optional title for the plot.
        x_label (str): Optional label for the x-axis.
        y_label (str): Optional label for the y-axis.
    """
    #plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    # Joint plot
    g = sns.jointplot(x=x_col,y=y_col, data=df,
                  kind="reg", truncate=False,
                  #xlim=(0, 60), ylim=(0, 12),
                  logx=True,
                  color="m", height=7)
    
    # Set labels and title
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    
    if log =='True':
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
    # Show the plot
    plt.grid(True)
    plt.show()