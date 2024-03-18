# SDRE_Solver

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# Define system dynamics
def f(t, x, A, B, Q, R):
    u = -np.dot(np.linalg.inv(R), np.dot(B.T, P.dot(x)))
    return np.dot(A, x) + np.dot(B, u)

# System parameters
A = np.array([[0, 1],
              [-1, -1]])
B = np.array([[0],
              [1]])
Q = np.array([[1, 0],
              [0, 1]])
R = np.array([[1]])

# Solve the Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Initial condition
x0 = np.array([1, 1])

# Time span
t_span = (0, 5)

# Solve the differential equation
sol = solve_ivp(f, t_span, x0, args=(A, B, Q, R), t_eval=np.linspace(t_span[0], t_span[1], 100))

# Print solution
print(sol.y)

