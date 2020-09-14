import csv
import numpy as np


# %% read data: copy from ta's colab
def read_data():
    np.random.seed(0)
    X_train_fpath = './data/X_train'
    Y_train_fpath = './data/Y_train'
    X_test_fpath = './data/X_test'

    # Parse csv files to numpy array
    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float).transpose()
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float).transpose()
        Y_train = np.expand_dims(Y_train, axis=0)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float).transpose()

    return X_train, Y_train, X_test


def divide_train_test(X_all, Y, train_proportion=0.7):
    n_x, m = X_all.shape
    np.random.seed(1)
    idx = np.random.rand(m)
    train_idx = [i for i in range(m) if idx[i] < train_proportion]
    val_idx = list(set(range(m)).difference(set(train_idx)))

    X_train = X_all[:, train_idx]
    Y_train = Y[:, train_idx]
    X_val = X_all[:, val_idx]
    Y_val = Y[:, val_idx]

    return X_train, Y_train, X_val, Y_val


# ta's split function
# def _train_dev_split(X, Y, dev_ratio = 0.25):
#     # This function spilts data into training set and development set.
#     train_size = int(len(X) * (1 - dev_ratio))
#     # train_size is a scalar, the far left dimension
#     return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]
# use:
# X_train, Y_train, X_val, Y_val=_train_dev_split(X, Y, dev_ratio = 0.1)

# %% preprocessing
def get_normalization(X):
    mean = np.mean(X, axis=1, keepdims=True)
    sigma = np.mean(np.square(X - mean), axis=1, keepdims=True)
    # if sigma=0, all data are the same, so set them to 0, by minus mean
    for i in range(X.shape[0]):
        if sigma[i] == 0:
            sigma[i] = 1

    X_normalize = (X - mean) / sigma
    return X_normalize, mean, sigma


def normalize(X, mean, sigma):
    return (X - mean) / sigma


# %% functions for training
def sigmoid(z):
    # To avoid overflow, minimum/maximum output value is set.
    epsilon = 1e-8
    s = 1. / (1. + np.exp(-z))
    return np.clip(s, epsilon, 1 - epsilon)


def compute_cross_entro_cost(Y, Y_hat):
    return -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))


def forward(W, X, b):
    z = np.dot(W, X) + b
    Y_hat = sigmoid(z)
    return z, Y_hat


def forward_compute_cost(W, X, b, Y, lamb):
    z, Y_hat = forward(W, X, b)

    cross_entro_cost = np.mean(compute_cross_entro_cost(Y, Y_hat))
    regu_cost = lamb * np.sum(np.square(W)) / len(Y)
    total_cost = cross_entro_cost + regu_cost
    return z, Y_hat, total_cost


def gradient_descent_once_logistic_regression(X, Y, W, b, lamb):
    m = Y.shape[1]
    # forward
    z, y_hat, this_cost = forward_compute_cost(W, X, b, Y, lamb)
    # backward
    dz = sigmoid(z) * (1 - sigmoid(z)) * (y_hat - Y)
    dW = np.dot(dz, X.T) / m + 2 * lamb * W / m
    db = np.mean(dz)

    return dW, db, this_cost


def update_params(W, b, dW, db, learning_rate, dacay_param=1):
    W = W - learning_rate * dW / dacay_param
    b = b - learning_rate * db / dacay_param
    return W, b


def train(X, Y, X_val, Y_val, learning_rate=0.1, iteration=100, lamb=0.000000001, see_val_cost=False):
    # retrieve dimensions
    n_x, m = X.shape

    # initialize parameters
    W = np.random.rand(1, n_x) * np.sqrt(2. / n_x)
    b = 0

    train_cost = []  # record cost
    val_cost = []
    for i in range(iteration):
        dW, db, this_train_cost = gradient_descent_once_logistic_regression(X, Y, W, b, lamb=lamb)
        update_params(W, b, dW, db, learning_rate)
        train_cost.append(np.mean(this_train_cost))
        if see_val_cost:
            _, Y_val_hat, this_val_cost = forward_compute_cost(W, X_val, b, Y_val, lamb)
            val_cost.append(this_val_cost)

    return W, b, train_cost, val_cost


def shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(X.shape[1])  # same as np.random.randint(low=0, high=m, size=m)
    np.random.shuffle(randomize)
    return (X[:, randomize], Y[:, randomize])


def decay(time, start=60, power=0.5):
    time_step = np.maximum(time - start, 1)
    return np.power(np.minimum(time_step, 2000), power)


def train_mini_batches(X, Y, X_val, Y_val,
                       learning_rate=0.1, iteration=8, lamb=0.000000001, batch_size=8,
                       lr_decay=True):
    # retrieve dimensions
    n_x, m = X.shape

    # initialize parameters
    W = np.random.rand(1, n_x) * np.sqrt(2. / n_x)
    b = 0

    # record cost
    train_cost = []
    val_cost = []

    # set mini-batch params
    num_batches = int(m / batch_size)
    np.random.seed(1)
    time_step = 1

    # traverse training set iteration times
    for i in range(iteration):
        X_shuffle, Y_shuffle = shuffle(X, Y)

        for j in range(num_batches + 1):
            X_mini_train = X_shuffle[:, j * batch_size:np.minimum((j + 1) * batch_size, m)]
            Y_mini_train = Y_shuffle[:, j * batch_size:np.minimum((j + 1) * batch_size, m)]
            dW, db, this_train_cost = gradient_descent_once_logistic_regression(X_mini_train, Y_mini_train,
                                                                                W, b, lamb=lamb)
            lr_divide = decay(time_step, 1000, 0.6) if lr_decay == True else 1
            W, b = update_params(W, b, dW, db, learning_rate, lr_divide)
            time_step += 1

        _, Y_train_hat, this_train_cost = forward_compute_cost(W, X, b, Y, lamb)
        train_cost.append(this_train_cost)

    return W, b, train_cost, val_cost


# %% validation functions
def predict(Y_hat):
    # copy from ta's codes
    # 四舍五入the probability
    return np.round(Y_hat).astype(np.int)


def accuracy(Y, Y_hat):
    return 1 - np.mean(np.abs(predict(Y_hat) - Y))


def inspect_values(Y, Y_hat, num=10):
    for i in range(num):
        print("label " + str(int(Y[0, i])) + " has predicted probability " + str(Y_hat[0, i]))
    # inspect = compute_cross_entro_cost(Y[0, :num], Y_hat[0, :num])
    # print("some single cost in training:" + str(inspect))


# %% probabilistic model utils
def get_mu_sigma(X):
    mu = np.mean(X, axis=1, keepdims=True)

    n_x, m = X.shape
    X_cov = np.zeros((n_x, n_x))
    X_norm = X - mu
    for x in X_norm.T:  # each time select a row(a sample)
        X_cov += np.dot(x.T, x) / m
    return mu, X_cov


def my_inv(cov):
    # copy from ta's code
    # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
    # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
    u, s, v = np.linalg.svd(cov, full_matrices=False)
    inv = np.matmul(v.T * 1 / s, u.T)
    return inv


# %% other functions
def submit(Y_pred_test, W, b, file_name):
    output_fpath = './output_{}.csv'
    with open(output_fpath.format('generative'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(Y_pred_test):
            f.write('{},{}\n'.format(i, label))
            # print(i, label)

    # np.save('weight.npy', W)
    # np.save('bias.npy', b)
