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
#from imblearn.over_sampling import SMOTE
#from imblearn.over_sampling import BorderlineSMOTE
#from imblearn.over_sampling import ADASYN

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


def scatter_plot(df, x_col, y_col, title=None, x_label=None, y_label=None,log=None):
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
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    
    # Scatter plot
    #plt.scatter(df[x_col], df[y_col], marker='o', alpha=0.5)
    ax = sns.scatterplot(x=x_col,y=y_col,legend=False,data=df)
    
    # Set labels and title
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    
    if log =='True':
        plt.yscale('log')
        plt.xscale('log')
    # Show the plot
    plt.grid(True)
    plt.show()


def plot_3D(df, x_col, y_col,z_col,  title=None, x_label=None, y_label=None,z_label=None, log=None):
    """
    Create a scatter plot of two columns from a Pandas DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The name of the column to use for the x-axis.
        y_col (str): The name of the column to use for the y-axis.
        z_col (str):
        title (str): Optional title for the plot.
        x_label (str): Optional label for the x-axis.
        y_label (str): Optional label for the y-axis.
        z_label (str):
    """
    #plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.set_style ("darkgrid")
    plt.figure (figsize = (5, 4))
    seaborn_plot = plt.axes (projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    # 3D  plot
    
    if log == True:
        x=np.log10(df[x_col])
        y=np.log10(df[y_col])
        z=np.log10(df[y_col])
    else:
        x = df[x_col]
        y = df[y_col]
        z = df[z_col]

    print(y)
    
    #ax.scatter(x,y,z,marker='.')
    seaborn_plot.scatter3D (x,y,z)


    if title:
        plt.title(title)
    
    # Show the plot
    plt.grid(True)
    plt.show()



def read_json_to_df(location):
    """
    Gets location of the json file and converts it to a pd.DataFrame 
    :param:location:
    :type:str
    :Return:rtn: pd.DataFrame 
    """
    with open(location) as f:
        a=f.readline(0)
        print(a)
        df = pd.read_json(f,lines=True) 

    return df



def plot_category_distribution(data, category_column_name, categoryLabel, plotType='bar'):
    """
    Plot the percentage distribution of a category in a student population.

    Parameters:
    - data: Pandas containing  records.
    - category_column_name: String name of the column containing category data.
    """
   

     # Create a count plot
    category_counts = data[category_column_name].value_counts(normalize=True).reset_index()
    category_counts.columns = [category_column_name, 'percentage']
    plt.figure(figsize=(10, 6))  # Adjust to appropriate size
    ax = sns.barplot(x=category_column_name, y='percentage', data=category_counts, ci=None)

    # Annotate percentages on the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.1%'),  # Format the annotation to percentage
              (p.get_x() + p.get_width() / 2., p.get_height()),
              ha = 'center', va = 'center',
              xytext = (0, 9), 
              textcoords = 'offset points')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate labels to be vertical
    plt.title('Percentage of Each '+ categoryLabel+' in the Population')
    plt.xlabel(categoryLabel)
    plt.ylabel('Percentage')

     
    # Show the plot
    plt.show()



def plot_category_distribution(data, category_column_name, categoryLabel):
    """
    Plot the percentage distribution of a category in a student population.

    Parameters:
    - data: Pandas containing  records.
    - category_column_name: String name of the column containing category data.
    """
    # Calculate the counts of each race
    category_counts = data[category_column_name].value_counts().sort_index()
    
    # Calculate the percentage of each race
    category_percentages = ((category_counts / category_counts.sum()) * 100).round(2)
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot for counts
    axs[0].bar(category_counts.index, category_counts.values, color='skyblue')
    axs[0].set_title('Count of Data by '+categoryLabel)
    axs[0].set_xlabel(categoryLabel)
    axs[0].set_ylabel('Count')
    axs[0].set_xticklabels(category_counts.index, rotation=45,ha='right')
    
    # Add count labels above the bars
    for index, value in category_counts.items():
        axs[0].text(index, value, str(value), ha='center')

    # Bar plot for percentages
    axs[1].bar(category_percentages.index, category_percentages.values, color='lightgreen')
    axs[1].set_title('Percentage of Data by '+categoryLabel)
    axs[1].set_xlabel(categoryLabel)
    axs[1].set_ylabel('Percentage (%)')
    axs[1].set_xticklabels(category_percentages.index, rotation=45, ha='right')
    
    # Add percentage labels above the bars
    for index, value in category_percentages.items():
        axs[1].text(index, value, f'{value}%', ha='center')

    # Show the plots
    plt.show()


# In[ ]:


def categorize_age(age):
    if age < 20:
        return '<20'
    elif 20 <= age < 30:
        return '20-30'
    elif 30 <= age < 40:
        return '30-40'
    elif 40 <= age < 50:
        return '40-50'
    elif 50 <= age < 60:
        return '50-60'
    elif 60 <= age < 70:
        return '60-70'
    else:
        return '70+'
    

def plot_age_distribution(data):
    # Categorize age into groups
    data['age_group'] = data['AGE'].apply(categorize_age)
       
    # Calculate counts and percentages for age groups
    age_group_counts = data['age_group'].value_counts().sort_index()
    age_group_percentages = ((age_group_counts / age_group_counts.sum()) * 100).round(2)
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    
     # Function to add labels
    def add_labels(ax, rects, is_percentage=False):
        for rect in rects:
            height = rect.get_height()
            label = f'{height:.2f}' if is_percentage else f'{int(height)}'
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    
   
    # Bar plot for age group counts
    age_count_bars = axs[0].bar(age_group_counts.index, age_group_counts.values)
    axs[0].bar(age_group_counts.index, age_group_counts.values)
    axs[0].set_title('Count of Patients by Age Group')
    axs[0].set_xlabel('Age Group')
    axs[0].set_ylabel('Count')
    axs[0].set_xticklabels(age_group_percentages.index, rotation=45, ha='right')
    add_labels(axs[0], age_count_bars)



    # Bar plot for age group percentages
    age_pct_bars = axs[1].bar(age_group_percentages.index, age_group_percentages.values)
    axs[1].bar(age_group_percentages.index, age_group_percentages.values)
    axs[1].set_title('Percentage of Patients by Age Group')
    axs[1].set_xlabel('Age Group')
    axs[1].set_ylabel('Percentage')
    axs[1].set_xticklabels(age_group_percentages.index, rotation=45, ha='right')


    
 # Adding percentage labels to the bars
    for i, value in enumerate(age_group_percentages.values):
        axs[1].text(i, value, f"{value}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



# In[ ]:


def categorize_income(income):
    if income < 20000:
        return '<20,000'
    elif 20000 <= income < 40000:
        return '20,000-30,000'
    elif 40000 <= income < 60000:
        return '40,000-60,000'
    elif 60000 <= income < 80000:
        return '60,000-80,000'
    elif 80000 <= income < 100000:
        return '80,000-100,000'
    elif 100000 <= income < 120000:
        return '100,000-120,000'
    else:
        return '120,000+'
    

def plot_income_distribution(data):
    # Categorize income into groups
    data['income_group'] = data['MEDIANINCOME'].apply(categorize_income)
       
    # Calculate counts and percentages for age groups
    income_group_counts = data['income_group'].value_counts().sort_index()
    income_group_percentages = ((income_group_counts / income_group_counts.sum()) * 100).round(2)
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    
     # Function to add labels
    def add_labels(ax, rects, is_percentage=False):
        for rect in rects:
            height = rect.get_height()
            label = f'{height:.2f}' if is_percentage else f'{int(height)}'
            ax.annotate(label,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    
   
    # Bar plot for age group counts
    income_count_bars = axs[0].bar(income_group_counts.index, income_group_counts.values)
    axs[0].bar(income_group_counts.index, income_group_counts.values)
    axs[0].set_title('Count of Patients by Income Group')
    axs[0].set_xlabel('Income Group')
    axs[0].set_ylabel('Count')
    axs[0].set_xticklabels(income_group_percentages.index, rotation=45, ha='right')
    add_labels(axs[0], income_count_bars)



    # Bar plot for age group percentages
    income_pct_bars = axs[1].bar(income_group_percentages.index, income_group_percentages.values)
    axs[1].bar(income_group_percentages.index, income_group_percentages.values)
    axs[1].set_title('Percentage of Patients by Income Group')
    axs[1].set_xlabel('Income Group')
    axs[1].set_ylabel('Percentage')
    axs[1].set_xticklabels(income_group_percentages.index, rotation=45, ha='right')


    
 # Adding percentage labels to the bars
    for i, value in enumerate(income_group_percentages.values):
        axs[1].text(i, value, f"{value}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# In[ ]:


def plot_median_income_bycategory(df,category,income,label,function):
    # Set the aesthetic style of the plots
        # Calculate median income per ethnicity
    median_income_per_category = df.groupby(category)[income].median().reset_index()

    # Sort the dataframe by median income for better visualization
    median_income_per_category = median_income_per_category.sort_values(income, ascending=False)

    # Set the aesthetic style of the plots
    sns.set_style('whitegrid')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Bar plot with seaborn
    barplot = sns.barplot(x=category, y=income, data=median_income_per_category, ci=None)

    # Adding title and labels
    plt.title('Median Income by '+label)
    plt.xlabel(label)
    plt.ylabel('Median Income')
    
 
    # Add data labels
    for bar in barplot.patches:
        # The text annotation for each bar should be its height (the median income)
        barplot.annotate(format(bar.get_height(), '.2f'),
                         (bar.get_x() + bar.get_width() / 2, 
                          bar.get_height()), ha='center', va='bottom',
                         color='black', xytext=(0, 5),
                         textcoords='offset points')

    # Show the plot


    # Show the plot
    plt.xticks(rotation=45, ha ='right')  # Rotate x-axis labels if necessary
    plt.tight_layout()  # Adjust layout to fit well
    plt.show()
# Call the function with the dataframe


# In[ ]:


def remove_outliers_iqr(df, column,fences):
    """
    Remove outliers from a specific column in a dataframe using IQR.

    Parameters:
    - df: pandas DataFrame.
    - column: The name of the column to remove outliers from.

    Returns:
    - DataFrame with outliers removed from the specified column.
    """
    if column not in df.columns:
        raise ValueError(f"The column {column} does not exist in the DataFrame.")
        
    if not np.issubdtype(df[column].dtype, np.number):
        raise TypeError(f"The column {column} is not numeric and cannot be processed with IQR.")

    #Q1 = df[column].quantile(0.25)
    #Q3 = df[column].quantile(0.75)
    #IQR = Q3 - Q1
    #lower_bound = Q1 - 1.5 * IQR
    #upper_bound = Q3 + 1.5 * IQR
    
    
    #df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    #return df_filtered
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - fences * IQR    # The choice of 20 is conventional and can be lowered or increased based on how agressive 
    upper_bound = Q3 + fences * IQR    # Do we want to be on removing outliers
    
    
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]    
    #print(df_filtered)

    return(df_filtered)


def plot_percentage_groupby(df, group_col, category_col, title=None, x_label=None, y_label=None,isstacked=False):
    """
    Group a DataFrame by a column and plot the percentage of one category to another within each group.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The name of the column to use for grouping.
        category_col (str): The name of the column containing the categories to compare.
        title (str): Optional title for the plot.
        x_label (str): Optional label for the x-axis.
        y_label (str): Optional label for the y-axis.
    """
   # Group by 'Group' and 'Category' and calculate count
    grouped = df.groupby([group_col, category_col]).size().reset_index(name='count')
    grouped['percentage'] = grouped.groupby(group_col)['count'].transform(lambda x: x / x.sum() * 100)

    # Unique groups and categories
    groups = grouped[group_col].unique()
    categories = grouped[category_col].unique()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot counts
    bar_plot = sns.barplot(x=group_col, y='percentage', hue=category_col, data=grouped)

   

    # Set labels and title
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
     # Add data labels to the bars
    
    plt.xticks(rotation=45,ha='right')

    
    # Show the plot
    plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Example usage:
# plot_percentage_groupby(df, 'GroupColumn', 'CategoryColumn', title='Percentage Comparison', x_label='X-axis Label', y_label='Y-axis Label')


