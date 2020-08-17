import numpy as np

FOLDER = "results"
DATA_FOLDER="data"

class TrainingParameters:

    def __init__(self, batch_size=1, n_layers=1, lr=0.01, gamma=0.99, discrete=False, sigma=1, action_step=0):
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.lr = lr
        self.gamma = gamma
        self.discrete = discrete
        self.sigma = sigma
        self.action_step = action_step

    def get_model_name(self):
        return "discrete_{}_batch_{}_layers_{}_lr_{}_gamma_{}_sigma_{}_step_{}".format(self.discrete, self.batch_size, self.n_layers, self.lr, self.gamma, self.sigma, self.action_step)

    @classmethod
    def from_string(cls, string):
        arr = string.split("_")

        # For backward compatibility with previous format of data
        if(len(arr) < 13):
            arr.append(0)
            arr.append(0)

        return TrainingParameters(
            discrete=arr[1] == "True",
            batch_size=int(arr[3]),
            n_layers=int(arr[5]),
            lr=float(arr[7]),
            gamma=float(arr[9]),
            sigma=float(arr[11]),
            action_step=float(arr[13])
        )

def get_data_path(model_name):
    return "{}/{}/{}.csv".format(FOLDER, DATA_FOLDER, model_name)

def save_results(file_name, data):
    """Save data in a csv file"""

    path = get_data_path(file_name)
    np.savetxt(path, np.array(data))

def save_results_path(path, data):
    """Save data in a csv file"""

    np.savetxt(path, np.array(data))

def load_results(file_name):
    """Load data from a csv file"""

    path = get_data_path(file_name)
    rew = np.loadtxt(path)
    iterations = np.array(range(rew.shape[1]))
    return (iterations, rew[0], rew[1])

def get_cum_reward(rewards, gamma):
    """Returns the cumulative reward from an array of rewards"""

    cum = 0
    decay = 1

    for r in rewards:
        cum += r * decay
        decay *= gamma

    return cum

def build_moving_average(data, alpha=0.5):
    """ From an array, return the moving average for this array (average_t = average_{t-1} * (1 - alpha) + item_t) """
    r = []

    for d in data:
        if len(r) == 0:
            r.append(d)
        else:
            r.append(r[-1] * (1 - alpha) + alpha * d)

    return r
