# ml2020homework1
# pm2.5 prediction: using previous 9 days' data to predict the next day's pm2.5

import pandas as pd

from utils import *

# %% preprocessing
# no extra line of "1" added, which may merge w and b
train_data = pd.read_csv('./data/train.csv', encoding='big5')
all_set = preprocess_list_train(train_data)
training_set, val_set = divide_train_test(all_set, train_proportion=0.7)

test_data = pd.read_csv('data/test.csv', encoding='big5', header=None)  # 指定没有表头
X_test = preprocess_test(test_data)  # have been a matrix

# %% test
# test dimension
# x, y = training_set[1]
# print(x.shape)
# print(y.shape)
# print(len(training_set))

# test read data
X1, Y1 = all_set[0]
print(X1[81: 90])
# should be 26	39	36	35	31	28	25	20	19
print(Y1)
# should be 30

# %% train the model with stochastic GD
iteration = 100000
W, b, cost = train_sgd(training_set, learning_rate=0.000001, iteration=iteration)
plot_average_cost(cost[-1000:], iteration, sep=10)
# plt.plot(np.squeeze(cost[-100:]))
# plt.show()

# evaluate train error
print("Average error (SGD) of the latest 10 examples is " + str(np.mean(np.sqrt(cost[-10:]))))

# %% preprocess into a matrix
X, Y_train = set_to_matrix(training_set)
X_train, mean, sigma = get_normalization(X)  # normalization

# %% train the model with matrix GD
lamb = 0.00000001
W, b, cost = train_matrix(X_train, Y_train, learning_rate=0.0015,
                          iteration=2000, lamb=0.00000001)
plt.plot(np.squeeze(cost))
plt.show()
print("loss (matrix GD): " + str(cost[-1]))

# %% train the model with matrix adagrad
lamb = 0.0001
W, b, cost = train_matrix_adagrad(X_train, Y_train, learning_rate=5,
                                  iteration=10000, lamb=lamb)
Y_pred_train = np.dot(W, X_train) + b
plt.plot(np.squeeze(cost))
plt.show()
print("cost (matrix adagrad): " + str(cost[-1]))

# result:
# not much use to increase iteration: 200000, cost(train, val)=31, 36
# 

# %% compute error on val set
X_val, Y_val = set_to_matrix(val_set)
Y_pred_val = np.dot(W, normalize(X_val, mean, sigma)) + b
print("cost (val): " + str(np.mean(compute_cost(Y_val, Y_pred_val, W, lamb))))

# %% see the test set
# see the trend
idx = 3
plt.plot(X_test[81:90, idx])
plt.show()

# %% predict test set
Y_pred_test = np.dot(W, normalize(X_test, mean, sigma)) + b

# get average pm2.5 of the test set to "self check"
Y_ave_test = np.mean(X_test[81:90, :], axis=0)
plt.plot(np.squeeze(Y_pred_test - Y_ave_test))
plt.show()
print(np.mean(np.square(Y_pred_test - Y_ave_test)))

# %% save predictions
submit(Y_pred_test, W)

# %% advanced: just use pm2.5 data
X_train_pm = X[81:90, :]
X_train_pm, mean_pm, sigma_pm = get_normalization(X_train_pm)

# %% train
lamb = 0.01
W_pm, b_pm, cost_pm = train_matrix_adagrad(X_train_pm, Y_train,
                                           learning_rate=100, iteration=5000, lamb=lamb)
Y_pred_train_pm = np.dot(W_pm, X_train_pm) + b_pm
plt.plot(np.squeeze(cost_pm))
plt.show()
print("loss (matrix adagrad): " + str(cost_pm[-1]))

# %% val
X_pm_val = X_val[81:90, :]
Y_pm_pred_val = np.dot(W_pm, normalize(X_pm_val, mean_pm, sigma_pm)) + b_pm
print(np.mean(compute_cost(Y_val, Y_pm_pred_val, W, lamb)))

#%% test
X_pm_test = X_test[81:90, :]
Y_pm_pred_test = np.dot(W_pm, normalize(X_pm_test, mean_pm, sigma_pm)) + b_pm
plt.plot(np.squeeze(Y_pm_pred_test - Y_ave_test))
plt.show()
print(np.mean(np.square(Y_pm_pred_test - Y_ave_test)))

#%%
submit(Y_pm_pred_test, W_pm)

# %% conclusion
'''
结论
- tuple的list不方便，大多数可以整体操作的（如乘法）还是用矩阵
- 没有normalization时，lr必须设得极小，iteration也要很大
- 同样训练次数，调整学习率为最佳（），adagrad找到了更好的解
- 100多个特征是不是还会欠拟合？val误差不大，但为甚test可能的误差比较大？？
- 
'''
