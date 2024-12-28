import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
g = 9.81  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in meters
L2 = 1.0  # length of pendulum 2 in meters
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg

def equations(t, y):
    theta1, z1, theta2, z2 = y
    delta = theta2 - theta1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
    denominator2 = (L2 / L1) * denominator1

    theta1_dot = z1
    z1_dot = (m2 * L1 * z1 * z1 * np.sin(delta) * np.cos(delta) +
              m2 * g * np.sin(theta2) * np.cos(delta) +
              m2 * L2 * z2 * z2 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta1)) / denominator1

    theta2_dot = z2
    z2_dot = (-m2 * L2 * z2 * z2 * np.sin(delta) * np.cos(delta) +
              (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
              (m1 + m2) * L1 * z1 * z1 * np.sin(delta) -
              (m1 + m2) * g * np.sin(theta2)) / denominator2

    return [theta1_dot, z1_dot, theta2_dot, z2_dot]

# Initial conditions: [theta1, theta1_dot, theta2, theta2_dot]
y0 = [np.pi / 2, 0, np.pi / 2, 0]

# Time span
t_span = (0, 20)  # 20 seconds
t = np.linspace(0, 20, 1000)

# Solve the system of differential equations
sol = solve_ivp(equations, t_span, y0, t_eval=t, method='RK45')
theta1, theta2 = sol.y[0], sol.y[2]

# Convert polar coordinates to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    thisx = [0, x1[frame], x2[frame]]
    thisy = [0, y1[frame], y2[frame]]
    line.set_data(thisx, thisy)
    return line,

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
plt.show()
