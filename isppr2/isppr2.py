import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# активації

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_deriv(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# втрата

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

# MLP 

class MLP(object):

    def __init__(self, layer_sizes, activation="relu", learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.activation_name = activation

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def activation(self, x):
        if self.activation_name == "relu":
            return relu(x)
        elif self.activation_name == "leaky_relu":
            return leaky_relu(x)
        elif self.activation_name == "tanh":
            return tanh(x)
        elif self.activation_name == "elu":
            return elu(x)

    def activation_deriv(self, x):
        if self.activation_name == "relu":
            return relu_deriv(x)
        elif self.activation_name == "leaky_relu":
            return leaky_relu_deriv(x)
        elif self.activation_name == "tanh":
            return tanh_deriv(x)
        elif self.activation_name == "elu":
            return elu_deriv(x)

    def forward(self, X):
        self.a = [X]
        self.z = []

        for i in range(len(self.weights) - 1):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            a = self.activation(z)
            self.a.append(a)

        z = self.a[-1] @ self.weights[-1] + self.biases[-1]
        self.z.append(z)
        a = softmax(z)
        self.a.append(a)

        return a

    def backward(self, X, y):
        m = X.shape[0]
        delta = self.a[-1] - y

        for i in reversed(range(len(self.weights))):
            dw = self.a[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # gradient clipping 
            dw = np.clip(dw, -5, 5)
            db = np.clip(db, -5, 5)

            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

            if i != 0:
                delta = (delta @ self.weights[i].T) * self.activation_deriv(self.z[i-1])

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X, y_true_labels):
        preds = self.predict(X)
        return np.mean(preds == y_true_labels)

# завантаження даних набору

data = pd.read_csv("A_Z Handwritten Data.csv").values

X = data[:, 1:] / 255.0
y = data[:, 0]

num_classes = 26
y_onehot = np.eye(num_classes)[y]

indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]
y_onehot = y_onehot[indices]

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y_onehot[:split], y_onehot[split:]
y_train_labels, y_val_labels = y[:split], y[split:]

# навчання

def train_model(model, epochs=40, batch_size=128):
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    n = len(X_train)

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, n, batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            model.forward(X_batch)
            model.backward(X_batch, y_batch)

        train_pred = model.forward(X_train)
        val_pred = model.forward(X_val)

        train_loss.append(cross_entropy(y_train, train_pred))
        val_loss.append(cross_entropy(y_val, val_pred))

        train_acc.append(model.accuracy(X_train, y_train_labels))
        val_acc.append(model.accuracy(X_val, y_val_labels))

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Loss: {train_loss[-1]:.4f} | "
              f"Val Acc: {val_acc[-1]:.4f}")

    return train_loss, val_loss, train_acc, val_acc

def plot_history(train_loss, val_loss, train_acc, val_acc, title):

    epochs = range(1, len(train_loss) + 1)

    # втрата
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # точність
    plt.figure()
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# базова модель

print("\nБазова модель з 1 скритим шаром")
model1 = MLP([784, 128, 26], activation="relu", learning_rate=0.005)
loss1, valloss1, acc1, valacc1 = train_model(model1, epochs=60)
plot_history(loss1, valloss1, acc1, valacc1,
             "Базова модель (1 скритий шар)")

# глибша модель

print("\nГлибша модель з 2 скритими шарами")
model2 = MLP([784, 256, 128, 26], activation="relu", learning_rate=0.005)
loss2, valloss2, acc2, valacc2 = train_model(model2, epochs=60)
plot_history(loss2, valloss2, acc2, valacc2,
             "Глибша модель (2 скритих шари)")

# помилкові класифікації

preds = model1.predict(X_val)
wrong = np.where(preds != y_val_labels)[0][:9]

plt.figure(figsize=(8,8))
for i, idx in enumerate(wrong):
    plt.subplot(3,3,i+1)
    plt.imshow(X_val[idx].reshape(28,28), cmap="gray")
    plt.title(f"P:{preds[idx]} T:{y_val_labels[idx]}")
    plt.axis("off")
plt.show()

# порівняння активацій

def measure_prediction_time(model, X, repeats=20):
    start = time.time()
    for _ in range(repeats):
        model.predict(X)
    return (time.time() - start) / repeats

def run_experiment(activation_name):

    model = MLP([784, 128, 26],
                activation=activation_name,
                learning_rate=0.01)

    start_train = time.time()
    train_loss, val_loss, train_acc, val_acc = train_model(model, epochs=40)
    train_time = time.time() - start_train

    pred_time = measure_prediction_time(model, X_val)

    # будуємо графіки
    plot_history(train_loss, val_loss, train_acc, val_acc,
                 f"Activation: {activation_name}")

    return {
        "Activation": activation_name,
        "Train Time (s)": round(train_time,2),
        "Val Accuracy": round(val_acc[-1],4),
        "Prediction Time (s)": round(pred_time,6)
    }

results = []
for act in ["relu", "leaky_relu", "elu", "tanh"]:
    print(f"\n{act.upper()}")
    results.append(run_experiment(act))

results_df = pd.DataFrame(results)

print("\nРезультати")
print(results_df)