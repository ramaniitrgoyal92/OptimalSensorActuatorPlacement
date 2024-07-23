import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize_scalar
from RobustH2_YALMIP import RobustH2_YALMIP 

# DEFINING THE SYSTEM MATRICES, A, B, C (ACC paper system)
A = np.array([[0.96, 0, 0], [0.04, 0.97, 0], [-0.04, 0, 0.9]])
B = np.array([[8.8, -2.3, 0, 0], [0.2, 2.2, 4.9, 0], [-0.21, -2.2, 1.9, 21]])
D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Number of states, outputs, etc.
num_states = A.shape[0]
num_actuators = B.shape[1]
num_sensors = C.shape[0]
num_outputs = C.shape[0]
num_processn = D.shape[1]

# Rank check for controllability
RankCheck = np.linalg.matrix_rank(np.hstack((B, A @ B, A**2 @ B, A**3 @ B, A**4 @ B, A**5 @ B, A**6 @ B, A**7 @ B)))
if RankCheck != num_states:
    print('SYSTEM IS UNCONTROLLABLE')

# Noise Specs and Matrices
W = 1e-3 * np.eye(num_processn)  # Process Noise covariance
V = 1e-3 * np.eye(num_sensors)  # Sensor Noise covariance

# Input Values
Price_Act = np.ones((num_actuators, 1))
Price_Sens = np.ones((num_sensors, 1))

Ubar = 5 * np.eye(num_actuators)
Ybar = 10 * np.eye(num_outputs)

Dollar_Bar = 10
Act_Prec_Bar = 1e3 * np.eye(num_actuators)
Sens_Prec_Bar = 1e3 * np.eye(num_sensors)

# Solution for the Robust control problem
XX_inv, Ac, Bc, Cc, Act_Prec_sol, Sens_Prec_sol, XX, Ybar0 = RobustH2_YALMIP(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act,
                                                                            Price_Sens, A, B, D, C, num_states, num_actuators, num_outputs,
                                                                            num_sensors, num_processn, W, V)

E_cl = np.hstack((np.eye(num_actuators), np.zeros((num_actuators, num_states)))) @ \
       np.vstack((np.zeros((num_actuators, num_sensors)), Cc, Bc, Ac)) @ \
       np.vstack((C, np.zeros((num_states, num_states))))
Control = np.trace(E_cl @ XX @ E_cl.T)
Output = np.trace(np.hstack((C, np.zeros((num_outputs, num_states)))) @ XX @ np.hstack((C, np.zeros((num_outputs, num_states)))).T)
Dollar = Price_Act.T @ np.diag(Act_Prec_sol) + Price_Sens.T @ np.diag(Sens_Prec_sol)
print(Act_Prec_sol)
print(Sens_Prec_sol)

# Different Cases
# NO H20
XX_inv, Ac, Bc, Cc, XX, Ybar0 = RobustH2_YALMIP(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act, Price_Sens, A, B, D, C,
                                                num_states, num_actuators, num_outputs, num_sensors, num_processn, W, V)
Act_Prec = 1e1 * np.eye(num_actuators)

# OptWater
# load('H2_OptH20.mat','Act_Prec','Ac', 'Bc', 'Cc')
# Act_Prec = .605*Act_Prec;

# OptWaterAcIter
# load('H2_OptH20ContrIter.mat','Act_Prec','Ac', 'Bc', 'Cc')
# Act_Prec = .675*Act_Prec;

# OptWaterAc
# load('H2_OptH20Contr.mat','Act_Prec','Ac', 'Bc', 'Cc')
# Act_Prec = .78*Act_Prec;

#%% Solution for Performance Loss
Acl = np.vstack((np.hstack((A, B @ Cc)), np.hstack((Bc @ C, Ac))))
Bcl = np.vstack((np.hstack((B, D, np.zeros((num_states, num_sensors)))), np.hstack((np.zeros((num_states, num_actuators + num_processn)), Bc))))
Ccl = np.hstack((C, np.zeros((num_outputs, num_states))))
Dcl = np.hstack((np.zeros((num_outputs, num_actuators + num_processn)), np.eye(num_outputs)))

U = Act_Prec_sol
Wcl = np.vstack((np.hstack((U, np.zeros((num_actuators, num_processn)), np.zeros((num_actuators, num_sensors)))),
                 np.hstack((np.zeros((num_processn, num_actuators)), W, np.zeros((num_processn, num_sensors)))),
                 np.hstack((np.zeros((num_sensors, num_actuators)), np.zeros((num_sensors, num_processn)), V / 1e-10))))

Xcl = solve_discrete_are(Acl, Bcl, Ccl.T @ Ccl, Dcl.T @ Dcl, inv(Wcl))
Ybar = np.zeros((num_outputs, num_outputs))
CON = Xcl - Acl.T @ Xcl @ Acl + (Acl.T @ Xcl @ Ccl).T @ np.linalg.inv(Ccl @ Xcl @ Ccl.T + Dcl @ Wcl @ Dcl.T) @ (Acl.T @ Xcl @ Ccl) >= 0

options = {'disp': True}
diagnostics = minimize_scalar(lambda trace_Ybar: -trace_Ybar, bounds=(0, None), constraints=CON, method='trust-constr', options=options)
Ybar = diagnostics.x
Perfrm_Ybar = np.trace(Ybar)

#%% Solution for the Detection
Acl = cp.bmat([[A, B @ Cc],
                   [Bc @ C, Ac]])
Bcl = cp.bmat([[D, np.zeros((num_states, num_sensors))],
               [np.zeros((num_states, num_prcocessn)), Bc]])
Ccl = cp.bmat([[C, np.zeros((num_outputs, num_states))]])
Dcl = cp.eye(num_outputs + num_states)
epsilon = 1e-10
Wcl = cp.bmat([[Act_Prec**-1, np.zeros((num_actuators, num_prcocessn)), np.zeros((num_actuators, num_sensors))],
               [np.zeros((num_prcocessn, num_actuators)), W, np.zeros((num_prcocessn, num_sensors))],
               [np.zeros((num_sensors, num_actuators)), np.zeros((num_sensors, num_prcocessn)), epsilon * np.eye(num_sensors)]])
Wcl_inv = cp.bmat([[Act_Prec, np.zeros((num_actuators, num_prcocessn)), np.zeros((num_actuators, num_sensors))],
                  [np.zeros((num_prcocessn, num_actuators)), W**-1, np.zeros((num_prcocessn, num_sensors))],
                  [np.zeros((num_sensors, num_actuators)), np.zeros((num_sensors, num_prcocessn)), 1 / epsilon * np.eye(num_sensors)]])

Acl_tlde = cp.bmat([[A, B @ Cc],
                    [np.zeros((num_states, num_states)), Ac]])
Pe_cl = cp.Variable((2 * num_states, 2 * num_states), symmetric=True)
Ycl = cp.Variable((2 * num_states, num_outputs))
Ve = V

# Optimization problem
objective = cp.Minimize(-cp.trace(Pe_cl))
constraints = [
    cp.bmat([[Pe_cl, Pe_cl @ Acl_tlde - Ycl @ Ccl, Ycl, Pe_cl @ Bcl],
             [Pe_cl @ Acl_tlde - Ycl @ Ccl, Ycl, Pe_cl @ Bcl, cp.bmat([[Pe_cl, np.zeros((2 * num_states, num_actuators))],
                                                                         [np.zeros((num_actuators, 2 * num_states)), Ve**-1, Wcl_inv]])]]) >= 0
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS, verbose=False)

# Get the solution values
Pe_cl = Pe_cl.value
Xe_cl = np.linalg.inv(Pe_cl)
Ycl = Ycl.value
L = Xe_cl @ Ycl

Xe_cl = cp.Variable((2 * num_states, 2 * num_states), symmetric=True)
constraints = [
    cp.bmat([[Xe_cl, Acl @ Xe_cl, Acl @ Xe_cl @ Ccl, Bcl],
             [Acl @ Xe_cl, Acl @ Xe_cl @ Ccl, Bcl, cp.bmat([[Xe_cl, np.zeros((2 * num_states, num_actuators))],
                                                           [np.zeros((num_actuators, 2 * num_states)), Ccl @ Xe_cl @ Ccl.T + Ve, Wcl]])]]) >= 0
]
problem = cp.Problem(cp.Minimize(cp.trace(Xe_cl)), constraints)
problem.solve(solver=cp.SCS, verbose=False)
Xe_cl = Xe_cl.value
L = Xe_cl @ Ccl.T @ np.linalg.inv(Ccl @ Xe_cl @ Ccl.T + Ve)

U_cal = cp.Variable((num_states, num_states), symmetric=True)
constraints = [
    A_cal @ U_cal @ A_cal.T + B @ U_cal @ B.T == U_cal
]
problem = cp.Problem(cp.Minimize(-cp.trace(U_cal)), constraints)
problem.solve(solver=cp.SCS, verbose=False)
U_cal = U_cal.value

Ybar_H20 = Ybar.value
Detection = cp.trace(C.T @ np.linalg.inv(X_cal) @ C @ U_cal)

                           
