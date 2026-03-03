import numpy as np
import matplotlib.pyplot as plt

a = 0.4
b = 0.8
c = 0.5
eta = 1.0

T = 50  # кількість ітерацій

# матриці
r = np.array([a, b])
R = np.array([[1.0, c],
              [c, 1.0]])

# функція вартості
def cost(w):
    return - r.T @ w + 0.5 * w.T @ R @ w

# градієнт
def grad(w):
    return R @ w - r

def rmsprop(w0, eta, beta=0.9, eps=1e-8, T=50):
    w = w0.copy()
    v = np.zeros_like(w)
    
    trajectory = [w.copy()]
    costs = [cost(w)]
    
    for t in range(T):
        g = grad(w)
        v = beta * v + (1 - beta) * (g ** 2)
        w = w - eta * g / (np.sqrt(v) + eps)
        
        trajectory.append(w.copy())
        costs.append(cost(w))
    
    return np.array(trajectory), np.array(costs)

# початкова т.
w0 = np.array([2.0, -2.0])
traj, costs = rmsprop(w0, eta=eta, T=T)

w_star = np.linalg.inv(R) @ r
print("Аналітичний мінімум w* =", w_star)
print("Останнє значення w(T) =", traj[-1])

# 3D графік поверхні і траєкторія

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)

Z = np.zeros_like(W1)
for i in range(len(w1)):
    for j in range(len(w2)):
        w_temp = np.array([W1[i,j], W2[i,j]])
        Z[i,j] = cost(w_temp)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(W1, W2, Z, alpha=0.4)
ax.plot(traj[:,0], traj[:,1], costs, 'r.-', label="RMSProp trajectory")

ax.set_xlabel("w1")
ax.set_ylabel("w2")
ax.set_zlabel("E(w)")
ax.set_title("RMSProp: траєкторія в 3D")
plt.show()

# контурний графік

plt.figure(figsize=(8,6))
plt.contour(W1, W2, Z, levels=30)
plt.plot(traj[:,0], traj[:,1], 'r.-')
plt.scatter(w_star[0], w_star[1], color='black', s=100, label="w*")
plt.title("RMSProp: траєкторія в площині W")
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend()
plt.grid()
plt.show()

# графік збіжності

plt.figure(figsize=(8,5))
plt.plot(costs)
plt.title("Збіжність RMSProp")
plt.xlabel("Ітерація")
plt.ylabel("E(w)")
plt.grid()
plt.show()