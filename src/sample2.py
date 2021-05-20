import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29
data = pd.read_csv('sat.trn',delimiter=' ', header=None)
data = data.loc[data[36] < 4].reset_index(drop=True)
data = data.to_numpy()
np.random.shuffle(data)
NUM_TRAIN = int(np.floor(data.shape[0] * 0.8))  #80% of total data
NUM_TRAIN_FOR_TRAIN = int(np.floor(NUM_TRAIN * 0.8))   #80% of train data


def load_spam_data(data):
    x = data[:, :-1].astype('float32')
    y = data[:, -1]
    x /= 255
    y -= 1
    x_train = x[:NUM_TRAIN]
    x_test = x[NUM_TRAIN:]
    y_train = y[:NUM_TRAIN]
    y_test = y[NUM_TRAIN:]
    return x_train, y_train, x_test, y_test


x_train_sub, y_train_sub, x_test, y_test = load_spam_data(data)

#train->train and validation
x_train = x_train_sub[:NUM_TRAIN_FOR_TRAIN]
x_val = x_train_sub[NUM_TRAIN_FOR_TRAIN:]
y_train = y_train_sub[:NUM_TRAIN_FOR_TRAIN]
y_val = y_train_sub[NUM_TRAIN_FOR_TRAIN:]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


class Forward:
    def dense_forward(x, w, b):  # w: 16, 256, 512 ....    b = 0 [0] * w.shape[1]
        return np.dot(x, w) + b

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        expX = np.exp(x - np.max(x))
        return expX / expX.sum(axis=1, keepdims=True)

    def cross_entropy(X, y):
        """
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        p = X.copy()
        #p = Forward.softmax(X)
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss


class Backward:
    def d_cross_entropy_softmax(score3, y, score2):
        """
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = y.shape[0]
        grad = score3.copy()
        grad[np.arange(m), y] -= 1 #.argmax(axis=1)
        grad /= m

        dW = score2.T.dot(grad)
        db = np.sum(grad, axis=0)

        return dW, db, grad

    def d_sigmod(x):
        return x * (1 - x)

    def dense_back(d_out, score1):
        dw = score1.T.dot(d_out)
        db = np.sum(d_out, axis=0)
        return dw, db


def accuracy(score3, y_batch):
    predicted_class = np.argmax(score3, axis=1)
    acc = np.mean(predicted_class == y_batch)
    return acc


def predict(X, W1, b1, W2, b2, W3, b3):
    dense1 = Forward.dense_forward(X, W1, b1)
    score1 = Forward.sigmoid(dense1)

    dense2 = Forward.dense_forward(score1, W2, b2)
    score2 = Forward.sigmoid(dense2)

    dense3 = Forward.dense_forward(score2, W3, b3)
    score3 = Forward.softmax(dense3)

    return score3


train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

W1 = 0.1 * np.random.normal(size=(x_train.shape[1],512))
b1 = np.zeros((1,512))
W2 = 0.1 * np.random.normal(size=(512, 256))
b2 = np.zeros((1,256))
W3 = 0.1 * np.random.normal(size=(256, 3))
b3 = np.zeros((1,3))

test_score = predict(x_test, W1, b1, W2, b2, W3, b3)
test_acc = accuracy(test_score, y_test)
print("random weights acc: ", test_acc)

epochs = 60
batch_size = 16
learning_rate = 0.09
for i in range(epochs):
    for batch_step in range(x_train.shape[0] // batch_size):
        X_batch = x_train[batch_step * batch_size: (batch_step+1) * batch_size, :]
        y_batch = y_train[batch_step * batch_size: (batch_step+1) * batch_size]

        #forward
        dense1 = Forward.dense_forward(X_batch, W1, b1)
        score1 = Forward.sigmoid(dense1)

        dense2 = Forward.dense_forward(score1, W2, b2)
        score2 = Forward.sigmoid(dense2)

        dense3 = Forward.dense_forward(score2, W3, b3)
        score3 = Forward.softmax(dense3)

        #backward
        dW3, db3, d_out = Backward.d_cross_entropy_softmax(score3, y_batch, score2)

        up_gradient = d_out.dot(W3.T)
        d_hiden2 = up_gradient * Backward.d_sigmod(score2)
        dw2, db2 = Backward.dense_back(d_hiden2, score1)

        up_gradient = d_hiden2.dot(W2.T)
        d_hiden1 = up_gradient * Backward.d_sigmod(score1)
        dw1, db1 = Backward.dense_back(d_hiden1, X_batch)

        #SGD
        if batch_step % 20 == 0:
            learning_rate *= 0.999
        W3 += -learning_rate * dW3
        b3 += -learning_rate * db3
        W2 += -learning_rate * dw2
        b2 += -learning_rate * db2
        W1 += -learning_rate * dw1
        b1 += -learning_rate * db1
    if i % 5 == 0:
        print('number of epochs: ', i)

    test_x = predict(x_train, W1, b1, W2, b2, W3, b3)
    test_score = Forward.cross_entropy(test_x, y_train)
    train_loss_list.append(test_score)
    train_acc = accuracy(test_x, y_train)
    train_acc_list.append(train_acc)

    val_x = predict(x_val, W1, b1, W2, b2, W3, b3)
    val_score = Forward.cross_entropy(val_x, y_val)
    val_loss_list.append(val_score)
    val_acc = accuracy(val_x, y_val)
    val_acc_list.append(val_acc)


final_score = predict(x_test, W1, b1, W2, b2, W3, b3)
final_acc = accuracy(final_score, y_test)
final_loss = Forward.cross_entropy(final_score, y_test)
print("final accuracy: ", final_acc)
print("final loss: ", final_loss)

train_loss_value = train_loss_list
val_loss_values = val_loss_list
epochs = range(1, epochs + 1)

plt.clf()
plt.plot(epochs, train_loss_value, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('S2_loss')
plt.show()

plt.clf()
train_accuracy = train_acc_list
val_accuracy = val_acc_list

plt.plot(epochs, train_accuracy, 'bo', label = 'Training acc')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('S2_accuracy')
plt.show()
