import numpy as np


def create_lagged_features(x, exogenous, lag=3):
    # x is a (n_timestep, ) array
    # exogenous is an (n_timestep, n_variables) array
    X, y = [], []
    for i in range(len(x) - lag):
        # TODO: ensure that the data contains also all exogenous variables.
        # 'concatenate' the lagged feature and the exogenous features.
        lagged = x[i:i+lag]
        exog_features = exogenous[i]
        features = np.concatenate([lagged, exog_features])
        X.append(features)
        y.append(x[i + lag])

    return np.asarray(X), np.asarray(y)


def iterative_forecast(reg, last_known, last_known_exogenous, steps):
    # last_known_exogenous is the last known exogenous variable of the training data
    # last_known is the last_known lagged training sample
    forecast = []
    window = list(last_known)

    for _ in range(steps):
        # TODO: ensure the predict function also gets the exogenous variables
        # Remember that they should occupy the same position in the feature
        # vector
      
        #window_array = np.array(window)
        features = np.concatenate([window, last_known_exogenous])
        prediction = reg.predict(features.reshape(1, -1))[0]
        forecast.append(prediction)
        
        window.pop(0)
        window.append(prediction)

    return np.array(forecast)