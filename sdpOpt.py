#%%
import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag
import sys
import csv
import pandas as pd
import matplotlib.pyplot as plt

#%%
# N = 10
# dim = 3

# m1 = m2 = m3 = 100
# k1 = k2 = k3 = 500
# d1 = d2 = d3 = 20

# K = np.vstack((([k1+k2, -k2, 0]),([-k2, k2+k3, -k3]),([0, -k3, k3])))  # stiffness
# C = np.vstack(([d1+d2, -d2, 0],[-d2, d2+d3, -d3],[0, -d3, d3]))  # damping

# M = np.diag([m1, m2, m3])  # mass

def blk_diag(A,B):
    k1 = A
    k2 = B
    dims_a = np.shape(A)
    dims_b = np.shape(B)
    A = cp.hstack((A, np.zeros((dims_a[0], dims_b[1]))))
    B = cp.hstack((np.zeros((dims_b[0], dims_a[1])), B))
    return cp.vstack((A,B))

#########System Description######################


# A = np.hstack([np.zeros((dim,dim)), np.eye(dim)])
# A = np.vstack([A, np.hstack([-np.linalg.inv(M)@K, -np.linalg.inv(M)@C])])



# B = np.zeros((6, dim))
# B[dim:,:]=np.eye(dim)
# B[dim,:] = [-1, 1, 0]
# B[dim+1,:] = [0, -1, 1]
# B[dim+2,:] = [0, 0, -1]

#%%
N = 5
A_full = pd.read_csv("A.csv").to_numpy().reshape((25,25,200))
B_full = pd.read_csv("B.csv").to_numpy().reshape((25,4,200))
# A = A.to_numpy()

A = A_full[:,:,0]
B = B_full[:,:,0]
#%%
D = np.eye(np.shape(A)[0])
E = np.copy(B)
Nx = np.shape(A)[0]
Nu = np.shape(B)[1]
Nw = np.shape(D)[1]

C = np.eye(Nx)
Ny = np.shape(C)[0]
F = np.eye(Ny)

Qm = 0*np.diag([4,4,4,.05,.05,.05])
Qcov = 0*Qm
Qbarm = block_diag(np.kron(np.eye(N),Qm),np.zeros(np.shape(Qm)))
Qbarcov = block_diag(np.kron(np.eye(N),Qcov),np.zeros(np.shape(Qcov)))
Rm = 20*np.eye(Nu)
Rcov = 200*np.eye(Nu)
Rbarm = np.kron(np.eye(N),Rm)
Rbarcov = np.kron(np.eye(N),Rcov)
eps = 1e-3

Ak = np.zeros((np.shape(A)[0],np.shape(A)[1], N))
Bk = np.zeros((np.shape(B)[0],np.shape(B)[1], N))
Dk = np.zeros((np.shape(D)[0],np.shape(D)[1], N))
for i in range(N):
    Ak[:, :, i] = A
    Bk[:, :, i] = B
    Dk[:, :, i] = D

PjFail = eps

########## Creating Matrix Structure ###############
Abar = np.zeros((np.shape(Ak)[0],np.shape(Ak)[1], N+1))
Bbar = np.zeros((np.shape(Bk)[0],np.shape(Bk)[1], N+1))

for j in range(1):#(N-1):
    Atemp = Ak[:,:,j]
    for i in range(N-j):
        Abar[:,:,j*N+i] = Atemp
        Atemp = Ak[:, :,j+i]@Atemp

    Abar[:, :, j*N+i+1] = Atemp


# Abar[:,:,j*N+1] = Ak[:,:,N]


Bbar = np.zeros((Nx,(N*Nu),N))
Btemp = Bk[:,:,0]


for i in range(N-1):
    Bbar[:,:,i] = np.hstack([Btemp, np.zeros((Nx,(N-i-1)*Nu))])
    Btemp = np.hstack([Ak[:,:,i]@Btemp, Bk[:,:,i]])

Bbar[:,:,N-1]=Btemp
Ebar = np.copy(Bbar)

Dbar = np.zeros((Nx,(N*Nw),N))
Dtemp = Dk[:,:,0]

for i in range(N-1):
    Dbar[:,:,i] = np.hstack([Dtemp, np.zeros((Nx,(N-i-1)*Nw))])
    Dtemp = np.hstack([Ak[:,:,i]@Dtemp, Dk[:,:,i]])

Dbar[:,:,N-1]=Dtemp

Acal = np.eye(Nx)
Bcal = np.zeros((Nx,N*Nu))
Dcal = np.zeros((Nx,N*Nw))


for i in range(N):

    Acal = np.vstack([Acal, Abar[:,:,i]])
    Bcal = np.vstack([Bcal, Bbar[:, :, i]])
    Dcal = np.vstack([Dcal, Dbar[:, :, i]])


Ecal = np.copy(Bcal)
Ccal = np.kron(np.eye(N+1),C)
Fcal = np.kron(np.eye(N+1),F)

## Creating more matrices
Ik = np.zeros((Nx, N*Nx, N))
for i in range(N):
    Ik[:, i*Nx:(i+1)*Nx,i] = np.eye(Nx)
IN = np.hstack([np.zeros((Nx, N*Nx)), np.eye(Nx)])

Iku = np.zeros((Nu, N*Nu, N))
for i in range(N):
    Iku[:,i*Nu:(i+1)*Nu,i] = np.eye(Nu)

## Convex
 # alpha(:,1) = [1 -5 -5 0 0 0]'; beta(:,1) = 1;
 # alpha(:,2) = [1 5 5 0 0 0]'; beta(:,2) = 1;
 # N_reg = 2;

##  Non-convex
#  alpha(:,1) = [1 0 0 0]'; beta(:,1) = -6;
#  alpha(:,2) = [-1 0 0 0]'; beta(:,2) = 5;
#  alpha(:,3) = [0 1 0 0]'; beta(:,3) = -1;
#  alpha(:,4) = [0 -1 0 0]'; beta(:,4) = 3;
#  N_reg = 4;

## Boundary Conditions

mu0 = np.array([.1,0.1,0.1,0,0,0]).reshape((6,1))
Sigma0 = np.diag([.1,.1,.1,.01,.01,.01])
muN = 2*mu0#np.array([0,0,0,0,0,0]).reshape((6,1))
SigmaN = 10*np.diag([.01,.01,.01,.001,.001,0.001])


#%%
####### Optimization Solver ##################
DollarBar = 100*100
PrcAct = 1 * np.ones((Nu, 1))
PrcSen = 1 * np.ones((Ny, 1))
V = cp.Variable((N * Nu, 1))

K1 = cp.Variable((Nu, Ny))
K2 = cp.Variable((Nu, Ny))
K3 = cp.Variable((Nu, Ny))
K4 = cp.Variable((Nu, Ny))
K5 = cp.Variable((Nu, Ny))
K6 = cp.Variable((Nu, Ny))
K7 = cp.Variable((Nu, Ny))
K8 = cp.Variable((Nu, Ny))
K9 = cp.Variable((Nu, Ny))
K10 = cp.Variable((Nu, Ny))

k = [K1, K2, K3, K4, K5, K6, K7, K8, K9, K10]
for i in range(N):
    if i==0:
        K = k[i]
    else:
        K = blk_diag(K, k[i])

K = cp.hstack([K, np.zeros((N * Nu, Ny))])


Jx = cp.Variable((N * Nx + Nx, 1))
Ju = cp.Variable((N * Nu, 1))
ElxMat = cp.Variable((N * Nx + Nx, 1))
EluMat = cp.Variable((N * Nu, 1))

ActPrec = cp.Variable((Nu, 1))
SenPrec = cp.Variable((Ny, 1))
Scale_SigmaN = cp.Variable((1, 1))

constraints = []
constraints += [PrcAct.T @ ActPrec + PrcSen.T @ SenPrec <= DollarBar]
constraints += [ActPrec <= 1e5]
constraints += [SenPrec <= 1e5]


constraints += [muN <= IN@(Acal@mu0 + Bcal@V)+1e-3*np.ones((6,1))]
constraints += [muN >= IN@(Acal@mu0 + Bcal@V)-1e-3*np.ones((6,1))]
# constraints += [muN == IN@(Acal@mu0 + Bcal@V)]
Wcov = np.eye(np.shape(Dcal)[1])

ActPrecCov = cp.kron(np.eye(N), cp.diag(ActPrec))
SenPrecCov = cp.kron(np.eye(N + 1), cp.diag(SenPrec))

Gcal = cp.hstack([((np.eye(np.shape(Bcal)[0])+Bcal@K@Ccal)@Acal), ((np.eye(np.shape(Bcal)[0])+Bcal@K@Ccal)@Dcal), ((np.eye(np.shape(Bcal)[0])+Bcal@K@Ccal)@Ecal), Bcal@K@Fcal])
Hcal = cp.hstack([K@Ccal@Acal, K@Ccal@Dcal, K@Ccal@Ecal, K@Fcal])


# Convex Ellipse
rd_ellps = 5
h_ellps = 10
Elalp = np.array([0, 0, h_ellps - .1, 0, 0, 0]).reshape((6,1))

ElP = np.diag([(1/rd_ellps)**2,(1/rd_ellps)**2,(1/h_ellps)**2,0,0,0])

Elbeta = 1

Mnum = 0e4
Elalpbar = np.kron(np.ones((N+1, 1)), Elalp)
Elbetbar = np.kron(np.ones((Nx*(N+1), 1)), Elbeta)
ElPbar = np.kron(np.eye(N+1), ElP)
block_mat = blk_diag(blk_diag(np.linalg.inv(Sigma0), np.linalg.inv(Wcov)), blk_diag(ActPrecCov, SenPrecCov))


c1 = cp.hstack((cp.diag(Jx), (cp.sqrt(Qbarm)@(Acal@mu0+Bcal@V)+Mnum*cp.sqrt(ElPbar)@((Acal@mu0+Bcal@V)-Elalpbar)), (cp.sqrt(Qbarcov)+Mnum*cp.sqrt(ElPbar))@Gcal))
c2 = cp.hstack(((np.sqrt(Qbarm)@(Acal@mu0+Bcal@V)+Mnum*cp.sqrt(ElPbar)@((Acal@mu0+Bcal@V)-Elalpbar)).T, np.eye(np.shape(V)[1]), np.zeros((np.shape(V)[1], np.shape(Gcal)[1]))))
c3 = cp.hstack((((np.sqrt(Qbarcov)+Mnum*cp.sqrt(ElPbar))@Gcal).T, np.zeros((np.shape(Gcal)[1], np.shape(V)[1])), block_mat))



# constraints += [cp.diag(Jx)>=0]
constraints += [cp.vstack((c1,c2,c3)) >= 0]

c1 = cp.hstack((cp.diag(Ju), (cp.sqrt(Rbarm)@V), (cp.sqrt(Rbarcov))@Hcal))
c2 = cp.hstack(((np.sqrt(Rbarm)@V).T, np.eye(np.shape(V)[1]),np.zeros((np.shape(V)[1], np.shape(Hcal)[1]))))
c3 = cp.hstack(((np.sqrt(Rbarcov)@Hcal).T, np.zeros((np.shape(Hcal)[1], np.shape(V)[1])), block_mat))
constraints += [cp.vstack((c1,c2,c3)) >= 0]


c1 = cp.hstack((cp.multiply(Scale_SigmaN, SigmaN), (IN@Gcal)))
c2 = cp.hstack(((IN@Gcal).T, block_mat))
# print(np.shape(IN@Gcal))
# print(np.shape(c2))
# constraints += [cp.vstack((c1,c2)) >= 0]
constraints +=[cp.multiply(Scale_SigmaN, SigmaN)>=0]
constraints += [block_mat>=0]
#### Other constraints

##### Solve SDP
# diagnostics = optimize(CON, Scale_SigmaN)
# diagnostics = cp.Problem(cp.Minimize(Scale_SigmaN), constraints)
cost = cp.vstack((Jx, Ju))
diagnostics = cp.Problem(cp.Minimize(cp.trace(cp.diag(cost))), constraints=constraints)
# diagnostics = cp.Problem(cp.Minimize(cp.trace(cp.diag(Ju))), constraints=constraints)
diagnostics.solve(verbose=True)

##### Results

V = V.value
K = K.value
ActPrec = ActPrec.value
SenPrec = SenPrec.value
ActPrecCov = ActPrecCov.value
SenPrecCov = SenPrecCov.value
Gcal = Gcal.value
Jx = Jx.value
Ju = Ju.value
cost = np.sum(np.vstack((Jx, Ju)))
Scale_SigmaN = Scale_SigmaN.value

print('V : ', V)
print('K : ', K)
print('ActPrec : ', ActPrec)
print('SenPrec : ', SenPrec)
Dollar = PrcAct.T @ ActPrec + PrcSen.T @ SenPrec
print('Dollar = ', Dollar)
print('Cost : ', cost)

block_mat2 = blk_diag(blk_diag(Sigma0, Wcov), blk_diag(np.linalg.inv(ActPrecCov), np.linalg.inv(SenPrecCov)))
FinalCov = (IN @ Gcal) @ block_mat2 @ (IN @ Gcal).T

# print('FinalCov : ', FinalCov)

sys.exit()

#%%
n_sim = 100
x_sim = np.zeros((Nx, N+1))
y_sim = np.zeros((Nx, N+1))
u_bar = V.reshape((Nu, N))
u_sim = np.copy(u_bar)
posx = np.zeros((n_sim, N+1))
posy = np.zeros((n_sim, N+1))
posz = np.zeros((n_sim, N+1))
# print(np.linalg.cholesky(Sigma0))
# print(np.random.normal(np.zeros((Nx,1))))
# sys.exit()
# ActPrec += 1e-3
# SenPrec += 1e-3
for j in range(n_sim):
    x_sim[:, 0] = (mu0 + 0.9*np.linalg.cholesky(Sigma0)@np.random.normal(np.zeros((Nx, 1)))).flatten()
    y_sim[:,0] = (x_sim[:,0] - mu0.reshape(np.shape(x_sim[:,0]))).flatten()
    wsim = np.random.normal(np.zeros((Nw, N)))
    wasim = np.linalg.cholesky(np.diag(np.divide(1, ActPrec).flatten())) @ np.random.normal(np.zeros((Nu, N)))
    wssim = np.linalg.cholesky(np.diag(np.divide(1, SenPrec).flatten())) @ np.random.normal(np.zeros((Ny, N)))
    for i in range(N):

        y_sim[:,i+1] = Ak[:,:,i]@y_sim[:,i] + Dk[:,:,i]@wsim[:,i]+ Bk[:,:,i]@wasim[:,i]
        u_sim[:,i] = u_bar[:,i] + K[Nu*i:Nu*(i+1), Ny*i:Ny*(i+1)]@(C @ y_sim[:,i] + F @ wssim[:,i])#.reshape(6,1)
        x_sim[:,i+1] = Ak[:,:,i]@x_sim[:,i] + Bk[:,:,i]@u_sim[:,i] + Dk[:,:,i]@wsim[:,i]+ Bk[:,:,i]@wasim[:,i]

    posx[j] = x_sim[0, :]
    posy[j] = x_sim[1, :]
    posz[j] = x_sim[2, :]

# print(np.shape(np.mean(posx, axis=0)))
mean_x = np.mean(posx, axis=0)
mean_y = np.mean(posy, axis=0)
mean_z = np.mean(posz, axis=0)
var_x = np.var(posx, axis=0)
var_y = np.var(posy, axis=0)
var_z = np.var(posz, axis=0)
print(mu0[:dim])
print(muN[:dim])
print('meanx : ', mean_x)
print('meany : ', mean_y)
print('meanz : ', mean_z)
print('varx : ', var_x)
print('vary : ', var_y)
print('varz : ', var_z)
# plt.plot(np.arange(N+1))