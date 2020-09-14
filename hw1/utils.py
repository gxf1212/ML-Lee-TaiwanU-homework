import csv
import matplotlib.pyplot as plt
import numpy as np


# %% preprocess
def preprocess_list_train(data):
    data[data == 'NR'] = 0  # replace NR with 0
    np_data = data.to_numpy()  # convert to numpy array
    np_data = np_data[0:, 3:]  # delete texts
    np_data = np_data.astype(float)  # convert datatype to float
    LENGTH = 18  # define number of features
    HOUR = 9
    num_days = 20
    # print(np_data[0][0])

    # create training set
    set = []
    for month in range(12):
        # create a continuous-time dataset for each month
        conti_data = np.zeros((LENGTH, 24 * num_days))
        for day in range(num_days):
            conti_data[:, day * 24: (day + 1) * 24] = np_data[
                                                      (month * num_days + day) * LENGTH:
                                                      (month * num_days + day + 1) * LENGTH, :]

        # for each conti-data, create a list of examples
        for i in range(conti_data.shape[1] - HOUR):
            x_example = conti_data[:, i:i + HOUR]  # select 9 columns, 5 is also ok
            x_example = x_example.reshape(HOUR * LENGTH, 1)  # make it a vector
            y_example = conti_data[9, i + HOUR]  # select y
            example = (x_example, y_example)

            # append elements
            set.append(example)

    return set


def divide_train_test(set, train_proportion=0.7):
    training_set = []
    val_set = []
    np.random.seed(0)

    idx = np.random.rand(len(set))
    for i in range(len(set)):
        if idx[i] < train_proportion:
            training_set.append(set[i])
        else:
            val_set.append(set[i])

    return training_set, val_set


def set_to_matrix(set):
    X = []
    Y = []
    for x, y in set:
        X.append(x)
        Y.append(y)

    X = np.squeeze(np.transpose(np.array(X)))
    Y = np.squeeze(np.transpose(np.array(Y)))
    return X, Y


def preprocess_test(data):
    data[data == 'NR'] = 0  # replace NR with 0
    np_data = data.to_numpy()  # convert to numpy array
    np_data = np_data[:, 2:]  # delete texts
    np_data = np_data.astype(float)  # convert datatype to float
    LENGTH = 18  # define number of features
    m = int(np_data.shape[0] / LENGTH)

    test_X = []
    for i in range(m):
        test_X_example = np_data[i * LENGTH:(i + 1) * LENGTH, :].ravel()
        test_X.append(test_X_example)

    test_X = np.array(test_X).transpose()
    return test_X


def get_normalization(X_matrix):
    mean = np.mean(X_matrix, axis=1, keepdims=True)
    sigma = np.mean(np.square(X_matrix - mean), axis=1, keepdims=True)
    X_normalize = (X_matrix - mean) / sigma
    return X_normalize, mean, sigma


def normalize(X, mean, sigma):
    return (X - mean) / sigma


# %% train


def compute_cost(Y, Y_hat, w, lamb):
    return np.mean(np.square(Y - Y_hat)) + lamb * np.sum(np.square(w)) / len(Y)


def gradient_descent(x, y, w, b, learning_rate, lamb):
    # forward
    y_hat = np.dot(w, x) + b
    loss = compute_cost(y, y_hat, w, lamb)
    # backward
    dw = - 2 * np.dot(y - y_hat, x.T) / x.shape[1] + 2 * lamb * w / y.shape[1]
    db = - 2 * np.mean(y - y_hat)
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b, loss


def adagrad_once(x, y, w, b, learning_rate, lamb, grad_sum_w, grad_sum_b):
    # forward
    y_hat = np.dot(w, x) + b
    loss = compute_cost(y, y_hat, w, lamb)
    # backward
    dw = - 2 * np.dot(y - y_hat, x.T) / x.shape[1] + + 2 * lamb * w / len(y)
    db = - 2 * np.mean(y - y_hat)
    grad_sum_w += np.square(dw)
    grad_sum_b += np.square(db)
    w = w - learning_rate * dw / np.sqrt(grad_sum_w)
    b = b - learning_rate * db / np.sqrt(grad_sum_b)

    return w, b, loss, grad_sum_w, grad_sum_b


def train_sgd(training_set, learning_rate=0.001, iteration=1000):
    # retrieve dimensions
    x_sample, _ = training_set[0]
    n_x = x_sample.shape[0]
    m = len(training_set)

    # initialize parameters
    w = np.zeros((n_x,))
    b = 0

    # train with random order
    iter_list = np.random.randint(low=0, high=m, size=iteration)
    cost = []  # record cost
    for i in range(iteration):
        iter = iter_list[i]
        x, y = training_set[iter]
        w, b, loss = gradient_descent(x, y, w, b, learning_rate)
        cost.append(loss)

    return w, b, cost


def train_matrix(X, Y, learning_rate=0.1, iteration=100, lamb=0.0000001):
    # retrieve dimensions
    n_x, m = X.shape

    # initialize parameters
    w = np.zeros((1, n_x))
    b = 0

    cost = []  # record cost
    for i in range(iteration):
        w, b, loss = gradient_descent(X, Y, w, b, learning_rate, lamb=lamb)
        cost.append(np.mean(loss))

    return w, b, cost


def train_matrix_adagrad(X, Y, learning_rate=0.001, iteration=10, lamb=0.00000001):
    # retrieve dimensions
    n_x, m = X.shape

    # initialize parameters
    w = np.zeros((1, n_x))
    grad_sum_w = np.zeros((1, n_x))
    b = 0
    grad_sum_b = 0

    cost = []  # record cost
    for i in range(iteration):
        w, b, loss, grad_sum_w, grad_sum_b = adagrad_once(X, Y, w, b, learning_rate, lamb, grad_sum_w, grad_sum_b)
        cost.append(np.mean(loss))

    return w, b, cost


# %% error and plot
def plot_average_cost(cost, iteration, sep=10):
    cost_plt = np.zeros((int(iteration / sep),))
    for i in range(len(cost_plt)):
        cost_plt[i] = np.mean(cost[i * sep:(i + 1):sep])
    plt.plot(np.squeeze(cost))
    plt.ylabel('cost')
    plt.xlabel('iterations (average tens)')
    plt.show()


def submit(Y_pred_test, W):
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(Y_pred_test.shape[1]):
            row = ['id_' + str(i), Y_pred_test[0][i]]
            csv_writer.writerow(row)
            print(row)

    np.save('weight.npy', W)
