import numpy as np

FOLDER = "results"
DATA_FOLDER="data"

class ModelParameters:

    def __init__(self, batch_size=1, n_layers=1, lr=0.01, gamma=0.99, discrete=False):
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.lr = lr
        self.gamma = gamma
        self.discrete = discrete

    def get_model_name(self):
        return "discrete_{}_batch_{}_layers_{}_lr_{}_gamma_{}".format(self.discrete, self.batch_size, self.n_layers, self.lr, self.gamma)

    @classmethod
    def from_string(cls, string):
        arr = string.split("_")
        return ModelParameters(
            discrete=bool(arr[1]),
            batch_size=int(arr[3]),
            n_layers=int(arr[5]),
            lr=float(arr[7]),
            gamma=float(arr[9])
        )

def save_results(model_name, cum_rew):
    file_name = "{}/{}/{}.csv".format(FOLDER, DATA_FOLDER, model_name)
    np.savetxt(file_name, np.array(cum_rew))

def read_results(model_name):
    file_name = "{}/{}/{}.csv".format(FOLDER, DATA_FOLDER, model_name)
    rew = np.loadtxt(file_name)
    iterations = np.array(range(rew.shape[1]))
    return (iterations, rew)

