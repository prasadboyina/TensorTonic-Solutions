import numpy as np

def linear_regression_closed_form(X, y):
    X = np.array(X)
    y = np.array(y)
    XT = X.T
    XTX = XT @ X
    XTX_inv = np.linalg.inv(XTX)
    XTy = XT @ y
    w = XTX_inv @ XTy
    
    return w