import cvxpy as cp
import numpy as np

def RobustH2_YALMIP(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act, Price_Sens, A, B, D, C, num_states, num_actuators, num_outputs, num_sensors, num_prcocessn, W, V):
    # Define CVXPY variables
    X = cp.Variable((num_states, num_states), symmetric=True)
    Y = cp.Variable((num_states, num_states), symmetric=True)
    L = cp.Variable((num_actuators, num_states))
    F = cp.Variable((num_states, num_sensors))
    Q = cp.Variable((num_states, num_states), symmetric=True)
    Ybar = cp.Variable((num_outputs, num_outputs), symmetric=True)

    Sens_Prec = V**-1

    Wcl_inv = cp.bmat([[W**-1, np.zeros((num_prcocessn, num_sensors))],
                       [np.zeros((num_sensors, num_prcocessn)), Sens_Prec]])
    Dcl = cp.bmat([[np.zeros((num_outputs, num_prcocessn)), np.eye(num_sensors)]])

    # Optimization problem
    objective = cp.Minimize(cp.trace(Ybar))
    constraints = [
        cp.bmat([[Ybar, C @ X, C, Dcl],
                 [X @ C.T, X, np.eye(num_states), np.zeros((num_states, num_prcocessn + num_sensors))],
                 [C.T, np.eye(num_states), Y, np.zeros((num_states, num_prcocessn + num_sensors))],
                 [Dcl.T, np.zeros((num_states, num_prcocessn + num_sensors)).T, np.zeros((num_states, num_prcocessn + num_sensors)).T, Wcl_inv]]) >= 0,
        cp.bmat([[X, np.eye(num_states), A @ X + B @ L, A, D, np.zeros((num_states, num_sensors))],
                 [np.eye(num_states).T, Y, Q, Y @ A + F @ C, Y @ B, Y @ D],
                 [A.T, Y @ A.T + F @ C.T, Q.T, X, np.zeros((num_states, num_prcocessn)), np.zeros((num_states, num_sensors))],
                 [B.T, Y @ B.T, np.zeros((num_prcocessn, num_states)), np.zeros((num_prcocessn, num_states)).T, W**-1, np.zeros((num_prcocessn, num_sensors))],
                 [D.T, Y @ D.T, np.zeros((num_prcocessn, num_states)), np.zeros((num_prcocessn, num_states)).T, np.zeros((num_prcocessn, num_sensors)).T, Sens_Prec]]) >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    # Get the solution values
    XX_inv = np.linalg.inv(X.value)
    Ac = (cp.bmat([[np.eye(num_states), Y],
                   [np.zeros((num_states, num_states)), np.eye(num_states)]]).value @ X.value).value
    Bc = (cp.bmat([[np.zeros((num_states, num_actuators))],
                   [L]]).value @ X.value).value
    Cc = (cp.bmat([[Q - Y @ A @ X, F],
                   [L, np.zeros((num_actuators, num_sensors))]]).value @ X.value).value

    return XX_inv, Ac, Bc, Cc, X.value, Ybar.value
