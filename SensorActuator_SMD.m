clc; close all;
clear variables

% cvx_setup
clc

%%
dim = 3;

%Nominal Structure Parameters
m1 = 100; m2 = 100; m3 = 100;
k1 = 500; k2 = 500; k3 = 500;
d1 = 20; d2 = 20; d3 = 20;

K = [k1+k2 -k2 0;-k2 k2+k3 -k3; 0 -k3 k3];  %stiffness
C = [d1+d2 -d2 0;-d2 d2+d3 -d3; 0 -d3 d3];  %damping
M = diag([m1;m2;m3]); %mass


%% DEFINING THE SYSTEM MATRICES, Ap, Bp and Ep

Ap = [zeros(dim,dim) eye(dim);-inv(M)*K -inv(M)*C];
Bp = [zeros(dim,dim);[-1 1 0;0 -1 1;0 0 -1]];
Ep = eye(2*dim);

%% Noise Specs and Matrices

Wp = 1;
Da = Bp;
Dp = [zeros(dim,1);m1;0;0];
num_process = 1;

%% Output and Measurement Matrices

num_states = size(Ap,1);
num_actuators = size(Bp,2);

Cy =[[1 0 0;-1 1 0;0 -1 1] zeros(dim,dim); zeros(dim,dim) [1 0 0;-1 1 0;0 -1 1]];
num_outputs = size(Cy,1);

Cz = eye(2*dim);
num_sensors = size(Cz,1);

Ds = eye(num_sensors);

%% Choose which strings are actuators and sensors
RankCheck = rank([Bp Ap*Bp Ap^2*Bp Ap^3*Bp Ap^4*Bp Ap^5*Bp Ap^6*Bp Ap^7*Bp]);

if RankCheck ~= num_states
    warning('SYSTEM IS UNCONTROLLABLE')
end


%% ------------------------ Input Values ----------------------------------

Price_Act = 20*ones(num_actuators,1);
Price_Sens = 20*ones(num_sensors,1);

Ubar = 5*eye(num_actuators);
Ybar = 2*eye(num_outputs);

Dollar_Bar = 100;

Act_Prec_Bar = 1*eye(num_actuators);
Sens_Prec_Bar = 1*eye(num_sensors);

%% Initializing the Constant Matrix G Used in the Convexifying Algorithm

[G, Q, Ac, Bc, Cc, Act_Prec, Sens_Prec, XX] = IA(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act, Price_Sens, Ap, Bp, Ep, Dp, Da, Ds, Cy, Cz, num_states, num_actuators, num_outputs, num_sensors, Wp);
E_cl = [eye(num_actuators) zeros(num_actuators,num_states)]*[zeros(num_actuators,num_sensors) Cc;Bc Ac]*[Cz zeros(num_sensors,num_states);zeros(num_states,num_states) eye(num_states)];
Control(iter1,iter2) = trace(E_cl*XX*E_cl');
Output(iter1,iter2) = trace([Cy zeros(num_outputs,num_states)]*XX*[Cy zeros(num_outputs,num_states)]');
Dollar(iter1,iter2) = Price_Act'*diag(Act_Prec)+Price_Sens'*diag(Sens_Prec);
Act_Prec;
Sens_Prec;



%%
figure()
surf(Ubar_iter,Ybar_iter,Dollar)
xlabel('Control');
ylabel('Output');
zlabel('Dollar');
