import pandas as pd
import seaborn as sns

def plot_cm(ytrue, ypred):
    '''Plot a Confusion Matrix.'''
    cm = pd.crosstab(ytrue, ypred)
    sns.heatmap(cm, annot=True)
