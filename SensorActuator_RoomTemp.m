clear all; clc;
% close all;

N = 25;
dim = 3;

m1 = 100; m2 = 100; m3 = 100;
k1 = 500; k2 = 500; k3 = 500;
d1 = 20; d2 = 20; d3 = 20;

K = [k1+k2 -k2 0;-k2 k2+k3 -k3; 0 -k3 k3];  %stiffness
C = [d1+d2 -d2 0;-d2 d2+d3 -d3; 0 -d3 d3];  %damping
M = diag([m1;m2;m3]); %mass


%% DEFINING THE SYSTEM MATRICES

A = [zeros(dim,dim) eye(dim);-inv(M)*K -inv(M)*C];
B = [zeros(dim,dim);[-1 1 0;0 -1 1;0 0 -1]];
D = eye(size(A,1));
E = B;
C = eye(Nx); 
Nx = size(A,1); Nu = size(B,2); Nw = size(D,2);
Ny = size(C,1); 
F = eye(Ny);

Qm = diag([4;4;4;.05;.05;.05]); Qcov = 0*Qm;
Qbarm = blkdiag(kron(eye(N),Qm),zeros(size(Qm)));  Qbarcov = blkdiag(kron(eye(N),Qcov),zeros(size(Qcov)));
Rm = 20*eye(Nu); Rcov = 200*eye(Nu);
Rbarm = kron(eye(N),Rm); Rbarcov = kron(eye(N),Rcov);
eps = 1e-3;
Ak = repmat(A,[1 1 N]); Bk = repmat(B,[1 1 N]); Dk = repmat(D,[1 1 N]);
PjFail = eps;

%% Creating Matrix Structure

for j = 1:1%N-1
    Atemp = Ak(:,:,j);
    for i = 1:N-j
        Abar(:,:,(j-1)*N+i) = Atemp;
        Atemp = Ak(:,:,j+i)*Atemp;
    end
    Abar(:,:,(j-1)*N+i+1) = Atemp;
end
% Abar(:,:,j*N+1) = Ak(:,:,N);

% Bbar(:,:,1) = zeros(nx,(N)*nu);
Btemp = Bk(:,:,1);
for i = 1:N-1
    Bbar(:,:,i) = [Btemp zeros(Nx,(N-i)*Nu)];
    Btemp = [Ak(:,:,i)*Btemp Bk(:,:,i)];
end
Bbar(:,:,N) = Btemp;
Ebar = Bbar;

% Dbar(:,:,1) = zeros(nx,(N)*nw);
Dtemp = Dk(:,:,1);
for i = 1:N-1
    Dbar(:,:,i) = [Dtemp zeros(Nx,(N-i)*Nw)];
    Dtemp = [Ak(:,:,i)*Dtemp Dk(:,:,i)];
end
Dbar(:,:,N) = Dtemp;

Acal = eye(Nx);
Bcal = zeros(Nx,N*Nu);
Dcal = zeros(Nx,N*Nw);
for i = 1:N
    Acal = [Acal;Abar(:,:,i)];
    Bcal = [Bcal;Bbar(:,:,i)];
    Dcal = [Dcal;Dbar(:,:,i)];
end
Ecal = Bcal;
Ccal = kron(eye(N+1),C);
Fcal = kron(eye(N+1),F);

%% Creating more matrices

for i = 1:N
    Ik(:,:,i) = [zeros(Nx,(i-1)*Nx) eye(Nx) zeros(Nx,(N-i+1)*Nx)];
end
IN = [zeros(Nx,N*Nx) eye(Nx)];

for i = 1:N
    Iku(:,:,i) = [zeros(Nu,(i-1)*Nu) eye(Nu) zeros(Nu,(N-i)*Nu)];
end

% Convex
% alpha(:,1) = [1 -5 -5 0 0 0]'; beta(:,1) = 1;
% alpha(:,2) = [1 5 5 0 0 0]'; beta(:,2) = 1;
% N_reg = 2;

%Non-convex
% alpha(:,1) = [1 0 0 0]'; beta(:,1) = -6;
% alpha(:,2) = [-1 0 0 0]'; beta(:,2) = 5;
% alpha(:,3) = [0 1 0 0]'; beta(:,3) = -1;
% alpha(:,4) = [0 -1 0 0]'; beta(:,4) = 3;
% N_reg = 4;

%% Boundary Conditions

mu0 = [-10;0.1;0.1;0;0;0]; Sigma0 = diag([.1;.1;.1;.01;.01;.01]);
muN = [0;0;0;0;0;0]; SigmaN = 1*diag([.01;.01;.01;1e0*.001;.001;0.001]);

%% Convex Optimization --------- YALMIP -----------------------------------
DollarBar = 100;
PrcAct = 1*ones(Nu,1);
PrcSen = 1*ones(Ny,1);
V = sdpvar(N*Nu,1);

K1 = sdpvar(Nu,Ny); K2 = sdpvar(Nu,Ny); K3 = sdpvar(Nu,Ny); K4 = sdpvar(Nu,Ny); K5 = sdpvar(Nu,Ny);
K6 = sdpvar(Nu,Ny); K7 = sdpvar(Nu,Ny); K8 = sdpvar(Nu,Ny); K9 = sdpvar(Nu,Ny); K10 = sdpvar(Nu,Ny);
K = blkdiag(K1,K2,K3,K4,K5,K6,K7,K8,K9,K10);
K = [K zeros(N*Nu,Ny)];

% J = sdpvar(1,1);
Jx = sdpvar(N*Nx+Nx,1);
Ju = sdpvar(N*Nu,1);
% DollarBar = sdpvar(1,1);
ElxMat = sdpvar(N*Nx+Nx,1);
EluMat = sdpvar(N*Nu,1);

ActPrec = sdpvar(Nu,1);
SenPrec = sdpvar(Ny,1);
Scale_SigmaN = sdpvar(1,1);
% SigmaN = diag(SigmaN);

CON = [];
CON = PrcAct'*ActPrec+PrcSen'*SenPrec <= DollarBar;
% CON = [CON, V <= 10];
% CON = [CON, V >= -10];
% CON = [CON, K(:) <= 100];
% CON = [CON, K(:) >= -100];
CON = [CON, ActPrec <= 1e5];
CON = [CON, SenPrec <= 1e5];
CON = [CON, muN == IN*(Acal*mu0+Bcal*V)];
Wcov = eye(size(Dcal,2));
ActPrecCov = kron(eye(N),diag(ActPrec));
SenPrecCov = kron(eye(N+1),diag(SenPrec));

% SigmaY = Acal*Sigma0*Acal'+Dcal*Wcov*Dcal';%+Ecal*ActPrecCov*Ecal';

Gcal = [((eye(size(Bcal,1))+Bcal*K*Ccal)*Acal) ((eye(size(Bcal,1))+Bcal*K*Ccal)*Dcal) ((eye(size(Bcal,1))+Bcal*K*Ccal)*Ecal) Bcal*K*Fcal];
Hcal = [K*Ccal*Acal K*Ccal*Dcal K*Ccal*Ecal K*Fcal];

% CON = [CON, [diag(Jx) Qbarm^(1/2)*(Acal*mu0+Bcal*V) Qbarcov^(1/2)*Gcal;
%             (Qbarm^(1/2)*(Acal*mu0+Bcal*V))' eye(size(V,2)) zeros(size(V,2),size(Gcal,2));
%             (Qbarcov^(1/2)*Gcal)' zeros(size(Gcal,2),size(V,2)) blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >=0];

% Convex Ellipse
rd_ellps = 5;
h_ellps = 10; 
Elalp = [0 0 h_ellps-.1 0 0 0]'; ElP = diag([1/rd_ellps^2;1/rd_ellps^2;1/h_ellps^2;0;0;0]); Elbeta = 1;


Mnum = 0e4;
Elalpbar = kron(ones(N+1,1),Elalp);
Elbetbar = kron(ones(Nx*(N+1),1),Elbeta);
ElPbar = kron(eye(N+1),ElP);
CON = [CON, [diag(Jx) (Qbarm^(1/2)*(Acal*mu0+Bcal*V)+Mnum*ElPbar^(1/2)*((Acal*mu0+Bcal*V)-Elalpbar)) (Qbarcov^(1/2)+Mnum*ElPbar^(1/2))*Gcal;
            (Qbarm^(1/2)*(Acal*mu0+Bcal*V)+Mnum*ElPbar^(1/2)*((Acal*mu0+Bcal*V)-Elalpbar))' eye(size(V,2)) zeros(size(V,2),size(Gcal,2));
            ((Qbarcov^(1/2)+Mnum*ElPbar^(1/2))*Gcal)' zeros(size(Gcal,2),size(V,2)) blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >=0];


CON = [CON, [diag(Ju) Rbarm^(1/2)*V Rbarcov^(1/2)*Hcal;
            (Rbarm^(1/2)*V)' eye(size(V,2)) zeros(size(V,2),size(Hcal,2));
            (Rbarcov^(1/2)*Hcal)' zeros(size(Hcal,2),size(V,2)) blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >=0];

CON = [CON, [Scale_SigmaN*SigmaN (IN*Gcal); (IN*Gcal)' blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >= 0];



%% Other constraints

% for i = 1:N
%     CON = [CON, [Elbeta-sum(ElxMat((i-1)*Nx+1:i*Nx,1)) ((Acal*mu0+Bcal*V)-Elalpbar)'*Ik(:,:,i)'*ElP^(1/2);
%           (((Acal*mu0+Bcal*V)-Elalpbar)'*Ik(:,:,i)'*ElP^(1/2))' eye(size(ElP,1))] >= 0];
%     CON = [CON, [diag(ElxMat((i-1)*Nx+1:i*Nx,1))  Ik(:,:,i)*ElPbar^(1/2)*Gcal;
%           (Ik(:,:,i)*ElPbar^(1/2)*Gcal)'   blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >= 0];  
% end
% CON = [CON, [Elbeta-sum(ElxMat((i)*Nx+1:end,1)) ((Acal*mu0+Bcal*V)-Elalpbar)'*IN'*ElP^(1/2);
%     (((Acal*mu0+Bcal*V)-Elalpbar)'*IN'*ElP^(1/2))' eye(size(ElP,1))] >= 0];
% CON = [CON, [diag(ElxMat((i)*Nx+1:end,1))  IN*ElPbar^(1/2)*Gcal;
%     (IN*ElPbar^(1/2)*Gcal)'   blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >= 0];
%  
% ElbetaU = 400;
% for i = 1:N
%     CON = [CON, [ElbetaU-sum(EluMat((i-1)*Nu+1:i*Nu,1)) V'*Iku(1:Nu,:,i)';
%           Iku(1:Nu,:,i)*V eye(Nu)] >= 0];
%     CON = [CON, [diag(EluMat((i-1)*Nu+1:i*Nu,1))  Iku(1:Nu,:,i)*Hcal;
%           (Iku(1:Nu,:,i)*Hcal)'   blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >= 0];  
% end

% CON = [CON, sum(Ju)+sum(Jx) <= 1.0e5];
    

% options = sdpsettings('solver','mosek','verbose',1,'debug',1);
% options = sdpsettings('verbose',1,'debug',1);
% diagnostics = optimize(CON,(sum(Ju)+sum(Jx)))
% diagnostics = optimize(CON,(sum(diag(SigmaN))))
% diagnostics = optimize(CON,DollarBar)
diagnostics = optimize(CON,Scale_SigmaN)

% diagnostics = optimize(CON,J,options)

%% Results

V = value(V);
K = value(K);
ActPrec = value(ActPrec)
SenPrec = value(SenPrec)
ActPrecCov = value(ActPrecCov);
SenPrecCov = value(SenPrecCov);
Gcal = value(Gcal);
Jx = value(Jx);
Ju = value(Ju);
Scale_SigmaN = value(Scale_SigmaN);
Dollar = PrcAct'*ActPrec+PrcSen'*SenPrec;
cost = sum([Jx;Ju]);
FinalCov = (IN*Gcal)*blkdiag(Sigma0,Wcov,ActPrecCov^(-1),SenPrecCov^(-1))*(IN*Gcal)';


%% Simulation Results
% N_sim = 100;
% for j = 1:N_sim
%     xsim(:,1) = mu0 + .9*chol(Sigma0)*randn(Nx,1);
%     ysim(:,1) = xsim(:,1) - mu0;
%     ubar = reshape(V,[],N);
%     wsim = randn(Nw,N);
%     wasim = chol(1*diag(1./ActPrec))*randn(Nu,N);
%     wssim = chol(1*diag(1./SenPrec))*randn(Ny,N);
%     for i = 1:N
%         ysim(:,i+1) = Ak(:,:,i)*ysim(:,i) + Dk(:,:,i)*wsim(:,i)+ Bk(:,:,i)*wasim(:,i);
%         usim(:,i) = ubar(:,i) + K(Nu*(i-1)+1:Nu*i,Ny*(i-1)+1:Ny*i)*(C*ysim(:,i)+ F*wssim(:,i));
%         xsim(:,i+1) = Ak(:,:,i)*xsim(:,i) + Bk(:,:,i)*usim(:,i) + Dk(:,:,i)*wsim(:,i)+ Bk(:,:,i)*wasim(:,i);
%     end
%     posx(j,:) = xsim(1,1:end-1); posy(j,:) = xsim(2,1:end-1); posz(j,:) = xsim(3,1:end-1);
% end





