import time
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor


def FQI(X, Y, X_next, possible_actions, iterations, verbose=True, gamma=0.95):
    """ FQI algorithm. Take as input a Learning set (X, Y, and X_next) and the name of the model to use"""

    model = ExtraTreesRegressor(50, max_features=3, bootstrap=False)
    Y_0 = Y

    start_time = time.time()


    for j in range(iterations):
        if verbose:
            remaining_iterations = iterations - j
            elapsed_time = time.time() - start_time
            remaining_time = 0
            if j > 0:
                remaining_time = elapsed_time / j * remaining_iterations
            print("Fit {}, elapsed time {:.0f}s, remaining time {:.0f}s".format(j, elapsed_time, remaining_time))

        model.fit(X, Y)

        # Update Y
        Y = []
        for i, x_next in enumerate(X_next):
            to_predict = np.array(list(map(lambda u: np.concatenate(([u], x_next)), possible_actions)))

            max_prediction = max(model.predict(to_predict))

            Y.append(Y_0[i] + gamma * max_prediction)
        
        j = j + 1

    return model
