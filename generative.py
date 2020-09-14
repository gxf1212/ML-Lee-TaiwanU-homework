# probabilistic generative model
# 如果用同一个控制台，前两个cell可以不执行
from utils import *
X_all, Y_all, X_test = read_data()

# %% preprocessing
X_all_normalize, mean, sigma = get_normalization(X_all)
X_train, Y_train, X_val, Y_val = divide_train_test(X_all_normalize, Y_all, train_proportion=0.9)
X_test_normalize = normalize(X_test, mean, sigma)
# generative model 有可解析的最佳解，因此不必使用到 development set
# these parts below can be separated into functions

#%% compute params
# retrieve constants
n_x, m = X_all_normalize.shape

# divide samples with different labels
idx_0 = [i for i in range(m) if Y_all[:, i] == 0.]
idx_1 = [i for i in range(m) if Y_all[:, i] == 1.]
X_0 = X_all_normalize[:, idx_0]
X_1 = X_all_normalize[:, idx_1]

# compute mean and covariance matrix
X_0_mean, X_0_sigma = get_mu_sigma(X_0)
X_1_mean, X_1_sigma = get_mu_sigma(X_1)
sigma = (X_0_sigma * len(idx_0) + X_1_sigma * len(idx_1)) / m
sigma_inv = my_inv(sigma)

# estimate w and b
W_gen = np.dot((X_0_mean - X_1_mean).T, sigma_inv)
b_gen = -1. / 2. * (X_0_mean.T).dot(sigma_inv).dot(X_0_mean) \
        + 1. / 2. * (X_1_mean.T).dot(sigma_inv).dot(X_1_mean) \
        + np.log(len(idx_0)/len(idx_1))
#%% use ta's code

# retrieve constants
n_x, m = X_all_normalize.shape
# 分别计算类别0和类别1的均值
X_train_0 = np.array([x for x, y in zip(X_all_normalize.T, Y_all.T) if y == 0.])
X_train_1 = np.array([x for x, y in zip(X_all_normalize.T, Y_all.T) if y == 1.])

mean_0 = np.mean(X_train_0, axis = 0)
mean_1 = np.mean(X_train_1, axis = 0)

# 分别计算类别0和类别1的协方差
cov_0 = np.zeros((n_x, n_x))
cov_1 = np.zeros((n_x, n_x))
# 没有keepdims，能在后面的矩阵乘中灵活使用
for x in X_train_0:
    cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
for x in X_train_1:
    cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

# 共享协方差 = 独立的协方差的加权求和
cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

# 通过SVD矩阵分解，可以快速准确地获得方差矩阵的逆
u, s, v = np.linalg.svd(cov, full_matrices=False)
inv = np.matmul(v.T * 1 / s, u.T)

# 计算w和b
W_gen = np.dot(inv, mean_0 - mean_1)
b_gen = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
    + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

# 只有15%正确率！不知道为什么？！
# %% test accuracy
_, Y_train_hat_gen, train_cost_gen = forward_compute_cost(W_gen, X_all_normalize, b_gen, Y_all, lamb=0)
print("cost (train): " + str(train_cost_gen))
print("accuracy (train): " + str(accuracy(Y_train_hat_gen, Y_all)))
# inspect_values(Y_all, Y_train_hat_gen, num=10)

# %% predict
_, Y_test_hat_gen = forward(W_gen, X_test_normalize, b_gen)
Y_pred_test_gen = predict(Y_test_hat_gen)

# %% submit generative
submit(Y_pred_test_gen, W_gen, b_gen, "submit_generative")