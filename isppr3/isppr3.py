
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from sklearn.model_selection import train_test_split

# завантаження та генерація початкових даних
np.random.seed(42)
tf.random.set_seed(42)

X_data = np.linspace(-1, 1, 100).astype(np.float32)

num_coef = 8
coef = [1, 20, 3, 4, 6, 7, 300, 2]

y_data = np.zeros_like(X_data, dtype=np.float32)
for i in range(num_coef):
    y_data += coef[i] * np.power(X_data, i)

y_data += np.random.randn(*X_data.shape).astype(np.float32) * 20.5

# дані подані графічно

plt.figure(figsize=(10, 6))
plt.scatter(X_data, y_data, s=25, alpha=0.8, label="Початкові дані")
plt.title("Початкові дані")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

# розбиття на train/test набори

X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

X_train = X_train.reshape(-1, 1).astype(np.float32)
X_val = X_val.reshape(-1, 1).astype(np.float32)
y_train = y_train.reshape(-1, 1).astype(np.float32)
y_val = y_val.reshape(-1, 1).astype(np.float32)


# tf.data.Dataset

def make_dataset(X, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# клас моделі tf.keras.Model, поліноміальна регресія

class PolynomialRegressionModel(tf.keras.Model):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        # вектор параметрів моделі
        self.w = tf.Variable(
            tf.random.normal(shape=(degree + 1, 1), stddev=0.1),
            trainable=True,
            name="weights"
        )

    def call(self, x):
        # x shape (batch, 1)
        powers = [tf.pow(x, i) for i in range(self.degree + 1)]
        x_poly = tf.concat(powers, axis=1)   # shape (batch, degree+1)
        return tf.matmul(x_poly, self.w)     # shape (batch, 1)

# функція втрат MSE і L2 регуляризація

@tf.function
def compute_loss(model, x, y, l2_lambda):
    y_pred = model(x)
    mse = tf.reduce_mean(tf.square(y - y_pred))
    l2_reg = tf.reduce_sum(tf.square(model.w))
    total_loss = mse + l2_lambda * l2_reg
    return total_loss, mse, l2_reg

# крок навчання і крок перевірки

def train_step(model, optimizer, x_batch, y_batch, l2_lambda):
    with tf.GradientTape() as tape:
        total_loss, mse, l2_reg = compute_loss(model, x_batch, y_batch, l2_lambda)

    grads = tape.gradient(total_loss, [model.w])
    optimizer.apply_gradients(zip(grads, [model.w]))
    return total_loss, mse, l2_reg

@tf.function
def val_step(model, x_batch, y_batch, l2_lambda):
    total_loss, mse, l2_reg = compute_loss(model, x_batch, y_batch, l2_lambda)
    return total_loss, mse, l2_reg

# навчання моделі

def train_model(
    learning_rate=0.001,
    l2_lambda=1e-4,
    epochs=100,
    batch_size=16,
    momentum=0.9,
    checkpoint_dir="checkpoints_poly",
    restore=True
):
    train_ds = make_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = make_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)

    model = PolynomialRegressionModel(degree=7)
    _ = model(tf.zeros((1, 1), dtype=tf.float32))   # будуємо модель
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

    # змінна для збереження номера епохи
    epoch_var = tf.Variable(0, dtype=tf.int64)

    ckpt = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch_var
    )
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # відновлення останнього checkpoint
    if restore and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"Відновлено з checkpoint: {manager.latest_checkpoint}")
        start_epoch = int(epoch_var.numpy())
    else:
        print("Checkpoint не знайдено. Навчання починається з нуля.")
        start_epoch = 0

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, epochs):
        # навчання
        epoch_train_losses = []
        for x_batch, y_batch in train_ds:
            total_loss, mse, l2_reg = train_step(model, optimizer, x_batch, y_batch, l2_lambda)
            epoch_train_losses.append(total_loss.numpy())

        # пкревірка
        epoch_val_losses = []
        for x_batch, y_batch in val_ds:
            total_loss, mse, l2_reg = val_step(model, x_batch, y_batch, l2_lambda)
            epoch_val_losses.append(total_loss.numpy())

        train_loss_mean = float(np.mean(epoch_train_losses))
        val_loss_mean = float(np.mean(epoch_val_losses))

        train_losses.append(train_loss_mean)
        val_losses.append(val_loss_mean)

        epoch_var.assign(epoch + 1)

        # вивід кожні 10 епох
        if (epoch + 1) % 10 == 0:
            print(
                f"Епоха {epoch + 1:3d}/{epochs} | "
                f"train_loss = {train_loss_mean:.4f} | "
                f"val_loss = {val_loss_mean:.4f}"
            )

        # checkpoint кожні 10 епох
        if (epoch + 1) % 10 == 0:
            save_path = manager.save()
            print(f"Checkpoint збережено: {save_path}")

    # фінальне збереження
    final_model_dir = "final_model_poly"
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_weights(os.path.join(final_model_dir, "model_weights.weights.h5"))
    print(f"Фінальна модель збережена у: {final_model_dir}")

    return model, train_losses, val_losses

# дослідження різних learning rate, параметрів регуляризації

def run_hyperparameter_search():
    learning_rates = [0.0001, 0.001, 0.01]
    l2_lambdas = [0.0, 1e-5, 1e-4, 1e-3]

    results = []

    print("\nПочаток підбору гіперпараметрів\n")

    for lr in learning_rates:
        for l2_lambda in l2_lambdas:
            print("-" * 70)
            print(f"Тестування: learning_rate = {lr}, l2_lambda = {l2_lambda}")

            model = PolynomialRegressionModel(degree=7)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

            train_ds = make_dataset(X_train, y_train, batch_size=16, shuffle=True)
            val_ds = make_dataset(X_val, y_val, batch_size=16, shuffle=False)

            train_losses = []
            val_losses = []

            for epoch in range(100):
                batch_train_losses = []
                for x_batch, y_batch in train_ds:
                    total_loss, _, _ = train_step(model, optimizer, x_batch, y_batch, l2_lambda)
                    batch_train_losses.append(total_loss.numpy())

                batch_val_losses = []
                for x_batch, y_batch in val_ds:
                    total_loss, _, _ = val_step(model, x_batch, y_batch, l2_lambda)
                    batch_val_losses.append(total_loss.numpy())

                train_losses.append(float(np.mean(batch_train_losses)))
                val_losses.append(float(np.mean(batch_val_losses)))

            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]

            results.append({
                "learning_rate": lr,
                "l2_lambda": l2_lambda,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "train_curve": train_losses,
                "val_curve": val_losses
            })

            print(
                f"Завершено: final_train_loss = {final_train_loss:.4f}, "
                f"final_val_loss = {final_val_loss:.4f}"
            )

    # вибір найкращої комбінації за найменшим val_loss
    best_result = min(results, key=lambda x: x["final_val_loss"])

    print("\n" + "=" * 70)
    print("Найкращі гіперпараметри:")
    print(f"learning_rate = {best_result['learning_rate']}")
    print(f"l2_lambda     = {best_result['l2_lambda']}")
    print(f"val_loss       = {best_result['final_val_loss']:.4f}")

    return results, best_result


results, best_result = run_hyperparameter_search()

# крива навчання для найкращих параметрів

plt.figure(figsize=(10, 6))
plt.plot(best_result["train_curve"], label="Train Loss")
plt.plot(best_result["val_curve"], label="Validation Loss")
plt.title(
    f"Крива навчання\nlearning_rate={best_result['learning_rate']}, "
    f"L2={best_result['l2_lambda']}"
)
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# навчання фін. моделі

best_lr = best_result["learning_rate"]
best_l2 = best_result["l2_lambda"]

model, train_losses, val_losses = train_model(
    learning_rate=best_lr,
    l2_lambda=best_l2,
    epochs=100,
    batch_size=16,
    momentum=0.9,
    checkpoint_dir="checkpoints_best_model",
    restore=True
)

# графік початкових даних та лінії регресії

X_plot = np.linspace(-1, 1, 300).reshape(-1, 1).astype(np.float32)
y_plot_pred = model(X_plot).numpy()

X_true = np.linspace(-1, 1, 300).astype(np.float32)
y_true = np.zeros_like(X_true)
for i in range(num_coef):
    y_true += coef[i] * np.power(X_true, i)

plt.figure(figsize=(11, 7))
plt.scatter(X_data, y_data, s=25, alpha=0.7, label="Початкові дані")
plt.plot(X_true, y_true, linewidth=2, label="Справжня крива (без шуму)")
plt.plot(X_plot, y_plot_pred, linewidth=2, label="Поліноміальна регресія")
plt.title("Початкові дані та лінія регресії")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()


print("\nНавчені параметри моделі:")
for i, w_i in enumerate(model.w.numpy().flatten()):
    print(f"w{i} = {w_i:.6f}")