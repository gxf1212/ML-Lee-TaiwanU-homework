# ml 2020 homework 2
# predict income >=50k?

# hat: model output value; no hat means true value
# hat在no hat之前
# gen means "generative model"
# train/val/test closely after X/Y
# W is a row vector, b is a real number

import matplotlib.pyplot as plt
from utils import *

X_all, Y_all, X_test = read_data()

# %% preprocessing
X_all_normalize, mean, sigma = get_normalization(X_all)
X_train, Y_train, X_val, Y_val = divide_train_test(X_all_normalize, Y_all, train_proportion=0.9)
X_test_normalize = normalize(X_test, mean, sigma)

# %% train with GD
# lamb = 0.00000000
# W, b, train_cost, val_cost = train(X_train, Y_train, X_val, Y_val,
#                              learning_rate=0.0001,
#                              iteration=2000,
#                              lamb=lamb, see_val_cost=False)
#
# plt.plot(np.squeeze(train_cost))
# plt.show()

'''
with only 60% accuracy
'''

#%% train with mini-batch
lamb = 0.00000000
batch_size = 8
lr = 0.004
max_iter = 200
W, b, train_cost, val_cost = train_mini_batches(X_train, Y_train, X_val, Y_val,
                                                learning_rate=lr,
                                                iteration=max_iter,
                                                lamb=lamb,
                                                batch_size=batch_size)
plt.plot(np.squeeze(train_cost))
plt.show()

# %% analysis
# gradient vanishing may occur due to sigmoid function, don't worry because it's just warning
# due to random training, results may differ
# the question is, fail to find the best solution
# or cannot update parameters later, have to decrease lr
# try different batch size or lr decay
# some good values: b_size, lr, iter
# 8,0.002,40, accu=0.82

# %% improvement: lr decay
# lamb = 0.00000000
# batch_size = 600
# lr = 0.01
# max_iter = 20000
# W, b, train_cost, val_cost = train_mini_batches(X_train, Y_train, X_val, Y_val,
#                                                 learning_rate=lr,
#                                                 iteration=max_iter,
#                                                 lamb=lamb,
#                                                 batch_size=batch_size,
#                                                 lr_decay=True)
# plt.plot(np.squeeze(train_cost))
# plt.show()

# %% analysis
# analysis cost
plt.plot(np.squeeze(train_cost[-300:]))
plt.show()
# which shows lr is too big!!

# some good values: b_size, lr, iter, power
# 8,0.007,250, accu=0.80
# 1000,0.01,3000 is what a net friend use
# 600,0.01,3000，0.4 accu=0.82,train>val
# 0.1 is too big; warning is because lr is too big
#TODO: adjust lr0 and iteration

# %% val
_, Y_train_hat, this_train_cost = forward_compute_cost(W, X_train, b, Y_train, lamb)
_, Y_val_hat, this_val_cost = forward_compute_cost(W, X_val, b, Y_val, lamb)

print("cost (train): " + str(this_train_cost))
print("cost (val): " + str(this_val_cost))
print("accuracy (train): " + str(accuracy(Y_train, Y_train_hat)))
print("accuracy (val): " + str(accuracy(Y_val, Y_val_hat)))
# inspect_values(Y_train, Y_train_hat, num=10)

# %% predict test
_, Y_test_hat = forward(W, X_test_normalize, b)
Y_pred_test = predict(Y_test_hat)

# %% submit logistic
submit(Y_pred_test, W, b, "submit_logistic")

# %% try code
