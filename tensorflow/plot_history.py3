import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

def plot_history(model, start=0):
    '''
    Plotting training history (metrics on each epoch).
    Plot will start from `start` epoch.
    `model` is (sub)instance of `keras.models`.
    '''
    
    history = pd.DataFrame(model.history.history)
    for metric in (model.metrics):
        name = metric.name
        valname = 'val_'+name
        print('min', history[name].min(), 'max', history[name].max(), name)
        if valname in history.columns:
            print('min', history[valname].min(), 'max', history[valname].max(), valname)
            history[[name,valname]].iloc[start:].plot()
        else:
            history[[name]].iloc[start:].plot()
