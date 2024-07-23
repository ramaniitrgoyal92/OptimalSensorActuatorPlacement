import cvxpy as cp
import numpy as np

def RobustH2(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act, Price_Sens, A, B, D, C, num_states, num_actuators, num_outputs, num_sensors, num_prcocessn, W, V):
    # Define CVXPY variables
    Act_Prec = cp.Variable((num_actuators, num_actuators), symmetric=True, PSD=True)
    Sens_Prec = cp.Variable((num_sensors, num_sensors), symmetric=True, PSD=True)
    X = cp.Variable((num_states, num_states), symmetric=True)
    Y = cp.Variable((num_states, num_states), symmetric=True)
    L = cp.Variable((num_actuators, num_states))
    F = cp.Variable((num_states, num_sensors))
    Q = cp.Variable((num_states, num_states), symmetric=True)

    # Defining Variables used in LMIs
    Act_Prec_Vec = cp.diag(Act_Prec)  # Extracting the diagonal elements
    Sens_Prec_Vec = cp.diag(Sens_Prec)  # Extracting the diagonal elements

    Sens_Prec = V**-1

    Wcl_inv = cp.bmat([[Act_Prec, np.zeros((num_actuators, num_prcocessn)), np.zeros((num_actuators, num_sensors))],
                       [np.zeros((num_prcocessn, num_actuators)), W**-1, np.zeros((num_prcocessn, num_sensors))],
                       [np.zeros((num_sensors, num_actuators)), np.zeros((num_sensors, num_prcocessn)), Sens_Prec]])
    Dcl = cp.bmat([[np.zeros((num_outputs, num_actuators)), np.zeros((num_outputs, num_prcocessn)), np.eye(num_sensors)]])

    # LMIs
    objective = cp.Minimize(cp.sum(cp.diag(Ubar)))  # Change this line if mv == 2
    constraints = [
        Price_Act.T @ Act_Prec_Vec + Price_Sens.T @ Sens_Prec_Vec <= Dollar_Bar,
        Act_Prec <= Act_Prec_Bar,
        Sens_Prec <= Sens_Prec_Bar,
        cp.bmat([[Ybar, C @ X, C, Dcl],
                 [X @ C.T, X, np.eye(num_states), np.zeros((num_states, num_actuators + num_prcocessn + num_sensors))],
                 [C.T, np.eye(num_states), Y, np.zeros((num_states, num_actuators + num_prcocessn + num_sensors))],
                 [Dcl.T, np.zeros((num_states, num_actuators + num_prcocessn + num_sensors)).T, np.zeros((num_states, num_actuators + num_prcocessn + num_sensors)).T, Wcl_inv]]) >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=False)

    # Check the status of the problem
    if problem.status == 'infeasible' or problem.status == 'unbounded':
        print('SYSTEM IS INFEASIBLE OR UNBOUNDED')

    # Get the solution values
    Act_Prec_sol = Act_Prec.value
    Sens_Prec_sol = Sens_Prec.value
    XX_inv = X.value**-1
    Ac = (cp.bmat([[np.eye(num_states), Y],
                   [np.zeros((num_states, num_states)), np.eye(num_states)]]).value @ X.value).value
    Bc = (cp.bmat([[np.zeros((num_states, num_actuators))],
                   [L]]).value @ X.value).value
    Cc = (cp.bmat([[Q - Y @ A @ X, F],
                   [L, np.zeros((num_actuators, num_sensors))]]).value @ X.value).value

    return XX_inv, Ac, Bc, Cc, Act_Prec_sol, Sens_Prec_sol, X.value
