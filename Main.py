import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NeuronNetwork import NeuronNetwork
from Sigmoid import Sigmoid

class Main():
    def evaluate_accuracy(NN, X, y):
        predictions = [NN.predict(x) for x in X]
        predicted_classes = [np.argmax(p) for p in predictions]
        true_classes = [np.argmax(y_true) for y_true in y]
        accuracy = accuracy_score(true_classes, predicted_classes)
        return accuracy

    def train(NN, X_train, y_train, X_test, y_test, method, epoch=100):
        train_accuracy = []
        test_accuracy = []
        loss = []
        for i in range(epoch):
            e = 0
            a_train = 0
            a_test = 0

            if method == "batch":
                for xe, ye in zip(X_train, y_train):
                    p = NN.predict(xe)
                    e += (ye - p)
                    if np.argmax(p) == np.argmax(ye):
                        a_train += 1
                    e /= len(X_train)
                    NN.fit(e)
                    loss.append(e)

            elif method == "online":
                for xe, ye in zip(X_train, y_train):
                    p = NN.predict(xe)
                    e = ye - p
                    loss_e = e
                    NN.fit(e)
                    if np.argmax(p) == np.argmax(ye):
                        a_train += 1
                    train_acc = a_train / len(y_train)
                    test_acc = Main.evaluate_accuracy(NN, X_test, y_test)

                    train_accuracy.append(train_acc)
                    test_accuracy.append(test_acc)


        print(f"Trained model using {method}")

        return train_accuracy, test_accuracy, loss

x = np.linspace(-3, 3, 60)
y = np.random.uniform(-30, 30, 60)
data = np.column_stack((x, y))

X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

NN = NeuronNetwork([4, 10, 3], Sigmoid(), 0.01)
batch_train_acc, batch_test_acc, batch_loss = Main.train(NN, X_train, y_train, X_test, y_test, method="batch", epoch=1000)

NN = NeuronNetwork([4, 10, 3], Sigmoid(), 0.01)
online_train_acc, online_test_acc, online_loss = Main.train(NN, X_train, y_train, X_test, y_test, method="online", epoch=1000)

combined_train_acc = np.concatenate((batch_train_acc, online_train_acc))
combined_test_acc = np.concatenate((batch_test_acc, online_test_acc))

plt.plot(combined_train_acc, label='Train Accuracy (Batch)', linestyle='--')
plt.plot(combined_test_acc, label='Test Accuracy (Batch)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.plot(online_train_acc, label='Train Accuracy (Online)')
plt.plot(online_test_acc, label='Test Accuracy (Online)')
plt.ylim(0.8, 1.0)
plt.legend()
plt.show()

batch_loss = np.array(batch_loss).ravel()
online_loss = np.array(online_loss).ravel()
combined_loss = np.concatenate((batch_loss, online_loss))

plt.plot(combined_loss, label='Loss (Batch)', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(online_loss, label='Loss (Online)')
plt.legend()
plt.show()