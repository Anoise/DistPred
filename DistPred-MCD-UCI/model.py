import torch.nn as nn

# deterministic feed forward neural network
class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)


# early stopping scheme for hyperparameter tuning
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                           shall be a small positive value.
                           Default: 0
            best_score: value of the best metric on the validation set.
            best_epoch: epoch with the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):

        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0