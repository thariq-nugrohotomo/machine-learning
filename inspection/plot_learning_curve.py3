import numpy as np
import pandas as pd

def plot_learning_curve(lcurve, log10=0):
    '''
    Plot the output obtained from `sklearn.model_selection.learning_curve`.
    '''
    
    index = (lcurve[0])
    if log10:
        index = np.log10(index)
    lcurve = pd.DataFrame(dict(
      train=lcurve[1].mean(axis=1),
      val=lcurve[2].mean(axis=1)
    ),index=index)
    lcurve.plot()
