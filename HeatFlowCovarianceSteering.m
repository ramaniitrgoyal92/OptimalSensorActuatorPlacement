clear all; clc;
% close all;

%% SMD

dim = 25; dt = 0.2; N = 10;

m1 = 100; m2 = 100; m3 = 100;
k1 = 500; k2 = 500; k3 = 500;
d1 = 20; d2 = 20; d3 = 20;

K = [k1+k2 -k2 0;-k2 k2+k3 -k3; 0 -k3 k3];  %stiffness
C = [d1+d2 -d2 0;-d2 d2+d3 -d3; 0 -d3 d3];  %damping
M = diag([m1;m2;m3]); %mass


% DEFINING THE SYSTEM MATRICES

% Ac = [zeros(dim,dim) eye(dim);-inv(M)*K -inv(M)*C];
% Bc = [zeros(dim,dim);[-1 1 0;0 -1 1;0 0 -1]];
% sysc = ss(Ac,Bc,[],[]);
% sysd = c2d(sysc,dt);
% A = sysd.A;
% B = sysd.B;
% dt = 1e-4;
A = reshape(csvread('A.csv'),[25,25,200]);
% A = A(:,:,1)
B0 = reshape(csvread('B.csv'),[25,4,200]);
b = B0(:,1,1);
B = b;
% for i=1:1:24
%     B = [B circshift(b,1)];
%     b=circshift(b,1);
% end
B = 0.0834*eye(25) % All actuators

% sysc = ss(Ac,Bc,[],[]);
% sysd = c2d(sysc, dt);
% A = sysd.A;
% B=sysd.B;
%%

% A = [1,0,0,dt,0,0;0,1,0,0,dt,0;0,0,1,0,0,dt;0,0,0,1,0,0;0,0,0,0,1,0;0,0,0,0,0,1]; 
% B = [dt^2/2 0 0;0 dt^2/2 0;0 0 dt^2/2;dt 0 0;0 dt 0;0 0 dt]; 
% w1 = 0; w2 = 0; w3 = 0; %.1047;%2*pi/60;
% Sw = [0 -w3 w2;w3 0 -w1;-w2 w1 0];
% Ac = [zeros(3,3) eye(3);-Sw^2 -2*Sw];
% Bc = [zeros(3,3);eye(3)];
% syms t
% A = expm(Ac*dt);
% B = int(expm(Ac*t)*Bc);
% B = vpa(subs(B,t,dt),4); B = double(B);

%%
D = .01*eye(size(A,1));
Nx = size(A,1); Nu = size(B,2); Nw = size(D,2);
C = eye(Nx); 
% C(4,:) = [];
Ny = size(C,1); F = 1*eye(Ny);
Qm = 0.1*eye(Nx);%diag([4;4;4;.05;.05;.05]); 
Qcov = 0*Qm;
Qbarm = blkdiag(kron(eye(N),Qm),zeros(size(Qm)));  Qbarcov = blkdiag(kron(eye(N),Qcov),zeros(size(Qcov)));
Rm = 20*eye(Nu); Rcov = 200*eye(Nu);
Rbarm = kron(eye(N),Rm); Rbarcov = kron(eye(N),Rcov);
eps = 1e-3;
% Ak = repmat(A,[1 1 N]); Bk = repmat(B,[1 1 N]); Dk = repmat(D,[1 1 N]);
Ak = A(:,:,1:N);
Bk = repmat(B,[1,1,N]);Dk = repmat(D,[1 1 N]);
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

% mu0 = [-10;0.1;0.1;0;0;0]; Sigma0 = diag([.1;.1;.1;.01;.01;.01]);
% muN = [0;1;1;0;0;0]; SigmaN = diag([.01;.01;.01;1e2*.001;.001;0.001]);

% mu0 = [5;5;5;1*2;1*2;1*2]; Sigma0 = diag([.1;.1;.1;.01;.01;.01]);
% mu0 = [3;3;8;-1*2;1*2;-1*2]; Sigma0 = diag([.1;.1;.1;.01;.01;.01]);
% muN = [0;0;0;0;0;0]; SigmaN = 1*diag([.01;.01;.01;1e0*.001;.001;0.001]);

mu0 = 53*ones(25,1);%[1;1;1;0;0;0];%[3;3;8;-1*2;1*2;-1*2]; 
Sigma0 = 0.1*eye(25);%diag([.1;.1;.1;.01;.01;.01]);
muN = 53*ones(25,1);%[0;0;0;0;0;0]; 
SigmaN = 0.1*Sigma0;%.1*diag([.01;.01;.01;1e0*.001;.001;0.001]);


%% CVX solver
cvx_begin sdp
% cvx_begin sdp quiet
cvx_precision high
cvx_solver mosek

 
DollarBar = 100000;
PrcAct = .1e0*ones(Nu,1);
PrcSen = .1e0*ones(Ny,1);

variable V(N*Nu,1);
variable K1(Nu,Ny);
variable K2(Nu,Ny);
variable K3(Nu,Ny);
variable K4(Nu,Ny);
variable K5(Nu,Ny);
variable K6(Nu,Ny);
variable K7(Nu,Ny);
variable K8(Nu,Ny);
variable K9(Nu,Ny);
variable K10(Nu,Ny);

K = blkdiag(K1,K2,K3,K4,K5,K6,K7,K8,K9,K10);
K = [K zeros(N*Nu,Ny)];

variable Jx(N*Nx+Nx,1);
variable Ju(N*Nu,1);
variable ElxMat(N*Nx+Nx,1);
variable EluMat(N*Nu,1);
variable ActPrec(Nu,1) nonnegative;
variable SenPrec(Ny,1) nonnegative;
variable Scale_SigmaN(1,1);

% Additional Variables

Wcov = eye(size(Dcal,2));
ActPrecCov = kron(eye(N),diag(ActPrec));
SenPrecCov = kron(eye(N+1),diag(SenPrec));
Gcal = [((eye(size(Bcal,1))+Bcal*K*Ccal)*Acal) ((eye(size(Bcal,1))+Bcal*K*Ccal)*Dcal) ((eye(size(Bcal,1))+Bcal*K*Ccal)*Ecal) Bcal*K*Fcal];
Hcal = [K*Ccal*Acal K*Ccal*Dcal K*Ccal*Ecal K*Fcal];

% Convex Ellipse
rd_ellps = 5;
h_ellps = 10; 
Elalp = zeros(25,1)%[0 0 h_ellps-.1 0 0 0, zeros(1,19)]';
ElP = zeros(25,25)%diag([1/rd_ellps^2;1/rd_ellps^2;1/h_ellps^2;0;0;0;zeros(19,1)]);
Elbeta = 1;


Mnum = 0e4;
Elalpbar = kron(ones(N+1,1),Elalp);
Elbetbar = kron(ones(Nx*(N+1),1),Elbeta);
ElPbar = kron(eye(N+1),ElP);
mu_n_bar = repmat(muN, [N+1,1,1]);

% LMIs

% minimize(Scale_SigmaN);
minimize(trace(diag(Ju))+trace(diag(Ju)))

% subject to
    PrcAct'*ActPrec+PrcSen'*SenPrec <= DollarBar;
    ActPrec <= 1e5;
    SenPrec <= 1e5;
    muN == IN*(Acal*mu0+Bcal*V);
    [diag(Jx) (Qbarm^(1/2)*(Acal*mu0+Bcal*V-mu_n_bar)+Mnum*ElPbar^(1/2)*((Acal*mu0+Bcal*V)-Elalpbar)) (Qbarcov^(1/2)+Mnum*ElPbar^(1/2))*Gcal;
       (Qbarm^(1/2)*(Acal*mu0+Bcal*V-mu_n_bar)+Mnum*ElPbar^(1/2)*((Acal*mu0+Bcal*V)-Elalpbar))' eye(size(V,2)) zeros(size(V,2),size(Gcal,2));
        ((Qbarcov^(1/2)+Mnum*ElPbar^(1/2))*Gcal)' zeros(size(Gcal,2),size(V,2)) blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >=0;
    
    [diag(Ju) Rbarm^(1/2)*V Rbarcov^(1/2)*Hcal;
        (Rbarm^(1/2)*V)' eye(size(V,2)) zeros(size(V,2),size(Hcal,2));
        (Rbarcov^(1/2)*Hcal)' zeros(size(Hcal,2),size(V,2)) blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >=0;
    
    [Scale_SigmaN*SigmaN (IN*Gcal); (IN*Gcal)' blkdiag(Sigma0^(-1),Wcov^(-1),ActPrecCov,SenPrecCov)] >= 0;
       
cvx_end

cvx_status


%% Simulation Results
N_sim = 1000;
for j = 1:N_sim
    xsim(:,1) = mu0 + .9*chol(Sigma0)*randn(Nx,1);
    ysim(:,1) = xsim(:,1) - mu0;
    ubar = reshape(V,[],N);
    wsim = randn(Nw,N);
    wasim = chol(1*diag(1./ActPrec))*randn(Nu,N);
    wssim = chol(1*diag(1./SenPrec))*randn(Ny,N);
    for i = 1:N
        ysim(:,i+1) = Ak(:,:,i)*ysim(:,i) + Dk(:,:,i)*wsim(:,i)+ Bk(:,:,i)*wasim(:,i);
        usim(:,i) = ubar(:,i) + K(Nu*(i-1)+1:Nu*i,Ny*(i-1)+1:Ny*i)*(C*ysim(:,i)+ F*wssim(:,i));
        xsim(:,i+1) = Ak(:,:,i)*xsim(:,i) + Bk(:,:,i)*usim(:,i) + Dk(:,:,i)*wsim(:,i)+ Bk(:,:,i)*wasim(:,i);
    end
    posx(j,:) = xsim(1,1:end-1); posy(j,:) = xsim(2,1:end-1); posz(j,:) = xsim(3,1:end-1);
end



%%
% figure()
% plot3(posx(1:N_sim,:)',posy(1:N_sim,:)',posz(1:N_sim,:)')
% hold on
% 
% % Draw Ellipse
% rd_ellps = 5;
% h_ellps = 10; 
% Elalp = [0 0 h_ellps-.1 0 0 0]'; ElP = diag([1/rd_ellps^2;1/rd_ellps^2;1/h_ellps^2;0;0;0]); Elbeta = 1;
% 
% % figure()
% cnstrntX = -rd_ellps:.1:rd_ellps;
% ElcnstrntZ2 = Elalp(3,1)-sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(3,3));
% th = linspace(0,2*pi,40);
% for i = 1:size(th,2)
% Ellpse = [cos(th(i)) sin(th(i)) 0;-sin(th(i)) cos(th(i)) 0;0 0 1]*[cnstrntX;zeros(size(cnstrntX));ElcnstrntZ2];
% plot3(Ellpse(1,:)',Ellpse(2,:)',Ellpse(3,:)',':','linewidth',1,'color','r')
% end
% 
% % ElcnstrntY1 = Elalp(2,1)+sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(2,2));
% % ElcnstrntY2 = Elalp(2,1)-sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(2,2));
% % ElcnstrntZ1 = Elalp(3,1)+sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(3,3));
% % plot3(cnstrntX',ElcnstrntY1',ElcnstrntZ1','linewidth',3,'color','k')
% % plot3(cnstrntX',ElcnstrntY2',ElcnstrntZ2','linewidth',3,'color','k')
% % plot3(cnstrntX',-ElcnstrntZ1',-ElcnstrntY1','linewidth',3,'color','k')
% % plot3(cnstrntX',-ElcnstrntZ2',-ElcnstrntY2','linewidth',3,'color','k')
% hold off
% xlim([-6 6])
% ylim([-6 6])
% zlim([0 12])
% % view([-152,12])
% view([44,12])
% % axis equal
% xlabel('$x$','Interpreter','latex')
% ylabel('$y$','Interpreter','latex')
% zlabel('$z$','Interpreter','latex')
% set(gca,'FontName','Times New Roman','fontsize', 20,'linewidth',1.15)
% set(gca,'XMinorTick','on','YMinorTick','on')
% set(gca,'ticklength',1.4*get(gca,'ticklength'))
% grid on

%%
MeanX0 = mean(posx(:,1))
MeanY0 = mean(posy(:,1))
MeanZ0 = mean(posz(:,1))

VarX0 = var(posx(:,1))
VarY0 = var(posy(:,1))
VarZ0 = var(posz(:,1))

MeanXN = mean(posx(:,end))
MeanYN = mean(posy(:,end))
MeanZN = mean(posz(:,end))

VarXN = var(posx(:,end))
VarYN = var(posy(:,end))
VarZN = var(posz(:,end))



%%
figure()
plot3(posx(1:N_sim,:)',posy(1:N_sim,:)',posz(1:N_sim,:)')
hold on

% Elalp = [0 0 5 0 0 0]'; ElP = diag([1/4;1/25;1/25;0;0;0]); Elbeta = 1;
% 
% 
% 
% 
% % Draw constraint line
% % figure()
% % cnstrntX = mu0(1,1)+2:-.1:muN(1,1);
% cnstrntX = -5:.1:5;
% % cnstrntY1 = (beta(:,1)-alpha(1,1).*cnstrntX)/alpha(2,1);
% % cnstrntY2 = (beta(:,2)-alpha(1,2).*cnstrntX)/alpha(2,2);
% % cnstrntZ1 = (beta(:,1)-alpha(1,1).*cnstrntX)/alpha(2,1);
% % cnstrntZ2 = (beta(:,2)-alpha(1,2).*cnstrntX)/alpha(2,2);
% % plot(cnstrntX',cnstrntY1','linewidth',3,'color','k')
% % plot(cnstrntX',cnstrntY2','linewidth',3,'color','k')
% ElcnstrntY1 = Elalp(2,1)+sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(2,2));
% ElcnstrntY2 = Elalp(2,1)-sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(2,2));
% ElcnstrntZ1 = Elalp(3,1)+sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(3,3));
% ElcnstrntZ2 = Elalp(3,1)-sqrt((Elbeta - ElP(1,1)*(cnstrntX-Elalp(1,1)).^2)./ElP(3,3));
% plot3(cnstrntX',ElcnstrntY1',ElcnstrntZ1','linewidth',3,'color','k')
% plot3(cnstrntX',ElcnstrntY2',ElcnstrntZ2','linewidth',3,'color','k')
% plot3(cnstrntX',-ElcnstrntZ1',-ElcnstrntY1','linewidth',3,'color','k')
% plot3(cnstrntX',-ElcnstrntZ2',-ElcnstrntY2','linewidth',3,'color','k')
% hold off
xlim([-2 2])
ylim([-2 2])
zlim([0 2])
% axis equal
xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')
zlabel('$z$','Interpreter','latex')
set(gca,'FontName','Times New Roman','fontsize', 20,'linewidth',1.15)
set(gca,'XMinorTick','on','YMinorTick','on')
set(gca,'ticklength',1.4*get(gca,'ticklength'))
grid on


