function [G1, Q_init,  Ac_init, Bc_init, Cc_init, Act_Prec_init, Sens_Prec_init, XX] = IA(Ubar, Ybar, Dollar_Bar, Act_Prec_Bar, Sens_Prec_Bar, Price_Act, Price_Sens, Ap, Bp, Ep, Dp, Da, Ds, Cy, Cz, num_states, num_actuators, num_outputs, num_sensors,Wp)

cvx_begin sdp quiet

cvx_precision high

variable Act_Prec(num_actuators,num_actuators) diagonal nonnegative
variable Sens_Prec(num_sensors,num_sensors) diagonal nonnegative
variable X(num_states,num_states) symmetric 
variable Y(num_states,num_states) symmetric
variable L(num_actuators, num_states)
variable F(num_states, num_sensors)
variable Q(num_states,num_states)


%Defining Variables used in LMIs

Act_Prec_Vec = Act_Prec*ones(num_actuators,1);
Sens_Prec_Vec = Sens_Prec*ones(num_sensors,1);


colsDp = size(Dp,2);

W_inv = [Wp^(-1) zeros(colsDp, num_actuators) zeros(colsDp, num_sensors);...
                zeros(num_actuators, colsDp) Act_Prec zeros(num_actuators, num_sensors);...
                zeros(num_sensors, colsDp) zeros(num_sensors, num_actuators) Sens_Prec];
            

Ap = Ep^(-1)*Ap;
Bp =  Ep^(-1)*Bp; 
Dp =  Ep^(-1)*Dp; 
Da =  Ep^(-1)*Da;


% LMIs 

minimize(Price_Act'*Act_Prec_Vec + Price_Sens'*Sens_Prec_Vec);
% minimize(Price_Act'*Act_Prec*ones(num_actuators,1) + Price_Sens'*Sens_Prec*ones(num_sensors,1));
% maximize(Ybar);

% subject to
    Price_Act'*Act_Prec_Vec + Price_Sens'*Sens_Prec_Vec <= Dollar_Bar;
    Act_Prec <= Act_Prec_Bar;
    Sens_Prec <= Sens_Prec_Bar;
    [Ybar Cy*X Cy;X*Cy' X eye(num_states);Cy' eye(num_states) Y] >= 0;
    [Ubar L zeros(num_actuators,num_states); L' X eye(num_states);zeros(num_states, num_actuators) eye(num_states) Y] >= 0;
    Phi11 = [Ap*X+Bp*L Ap; Q Y*Ap+F*Cz]+[Ap*X+Bp*L Ap; Q Y*Ap+F*Cz]';
    Phi12 = [Dp Da zeros(num_states, num_sensors); Y*Dp Y*Da F*Ds];
    [Phi11 Phi12; Phi12' -W_inv] <= 0;

cvx_end
cvx_status

V = Y;
U = Y^(-1)-X;

Control_Matrix = [V^(-1) -V^(-1)*Y*Bp; zeros(num_actuators, num_states) eye(num_actuators)]*[Q-Y*Ap*X F;L zeros(num_actuators, num_sensors)]*...
                                [U^(-1) zeros(num_states, num_sensors); -Cz*X*U^(-1) eye(num_sensors)];
                            
Ac_init = Control_Matrix(1:num_states, 1:num_states);    
Bc_init = Control_Matrix(1:num_states, num_states+1:end);
Cc_init = Control_Matrix(num_states+1:end,1:num_states);

Ap = Ep*Ap;
Bp =  Ep*Bp; 

Acl = [Ap Bp*Cc_init; Bc_init*Cz Ac_init];

Ecl = [Ep zeros(num_states); zeros(num_states) eye(num_states)];

%The following calculation of the full Covariance matrix XX comes from the
%dissertation of Faming Li.


T = [eye(num_states) Y; zeros(num_states) V'];

TTXT = [X eye(num_states);eye(num_states) Y];

XX = (T'^(-1))*TTXT*(T^(-1));


G1 = (Acl-Ecl)*XX;


Q_init = XX^(-1);

Act_Prec_init = Act_Prec;
Sens_Prec_init= Sens_Prec;