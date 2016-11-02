# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 19:14:39 2016

@author: roman
"""

from sympy import *

# q: quaternion describing rotation from frame 1 to frame 2
# returns a rotation matrix derived form q which describes the same
# rotation
def quat2Rot(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    
    Rot = Matrix([[q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                  [2*(q1*q2 + q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
                   [2*(q1*q3-q0*q2), 2*(q2*q3 + q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
                   
    return Rot

# take an expression calculated by the cse() method and write the expression
# into a text file in C format
def write_simplified(P_touple, filename, out_name):
    subs = P_touple[0]
    P = Matrix(P_touple[1])
    fd = open(filename, 'a')
    
    is_vector = P.shape[0] == 1 or P.shape[1] == 1 
    
    # write sub expressions
    for index, item in enumerate(subs):
        fd.write('float ' + str(item[0]) + ' = ' + str(item[1]) + ';\n')
        
    # write actual matrix values
    fd.write('\n')
    
    if not is_vector:
        iterator = range(0,sqrt(len(P)), 1)
        for row in iterator:
            for column in iterator:
                fd.write(out_name + '[' + str(row) + '][' + str(column) + '] = ' + str(P[row, column]) + ';\n')
    else:
        iterator = range(0, len(P), 1)
        
        for item in iterator:
            fd.write(out_name + '[' + str(item) + '] = ' + str(P[item]) + ';\n')

    fd.write('\n\n')  
    fd.close()

########## Symbolic variable definition #######################################

# model state
w_n = Symbol("w_n", real=True)  # wind in north direction
w_e = Symbol("w_e", real=True)  # wind in east direction
k_tas = Symbol("k_tas", real=True) # true airspeed scale factor
state = Matrix([w_n, w_e, k_tas])

# process noise
q_w = Symbol("q_w", real=True) # process noise for wind states
q_k_tas = Symbol("q_k_tas", real=True) # process noise for airspeed scale state

# airspeed measurement noise
r_tas = Symbol("r_tas", real=True)

# sideslip measurement noise
r_beta = Symbol("r_beta", real=True)

#################### time varying parameters ##################################

# vehicle velocity
v_n = Symbol("v_n", real=True)  # north velocity in earth fixed frame
v_e = Symbol("v_e", real=True)  # east velocity in earth fixed frame
v_d = Symbol("v_d", real=True)  # down velocity in earth fixed frame

# unit quaternion describing vehicle attitude, qw is real part
qw = Symbol("qw", real=True)
qx = Symbol("qx", real=True)
qy = Symbol("qy", real=True)
qz = Symbol("qz", real=True)
q_att = Matrix([qw, qx, qy, qz])

# sampling time in seconds
dt = Symbol("dt", real=True)

######################## State and covariance prediction ######################

# state transition matrix is zero because we are using a stationary
# process model. We only need to provide formula for covariance prediction

# create process noise matrix for covariance prediction
# the process noise appears linearly in the state equation so we can
# compute it directly
state_new = state + Matrix([q_w, q_w, q_k_tas]) * dt
Q = diag(q_w, q_k_tas)
L = state_new.jacobian([q_w, q_k_tas])
Q = L * Q * Transpose(L)

# define symbolic covariance matrix
p00 = Symbol('_P[0][0]', real=True)
p01 = Symbol('_P[0][1]', real=True)
p02 = Symbol('_P[0][2]', real=True)
p12 = Symbol('_P[1][2]', real=True)
p11 = Symbol('_P[1][1]', real=True)
p22 = Symbol('_P[2][2]', real=True)
P = Matrix([[p00, p01, p02], [p01, p11, p12], [p02, p12, p22]])

# covariance prediction equation
P_next = P + Q

# simplify the result and write it to a text file in C format
PP_simple = cse(P_next, symbols('SPP0:30'))
P_pred = Matrix(PP_simple[1])
write_simplified(PP_simple, "cov_pred.txt", 'P_next')


############################ Measurement update ###############################

# airspeed fusion

tas_pred = Matrix([((v_n - w_n)**2 + (v_e - w_e)**2 + v_d**2)**0.5]) * k_tas
# compute true airspeed observation matrix
H_tas = tas_pred.jacobian(state)
# simplify the result and write it to a text file in C format
H_tas_simple = cse(H_tas, symbols('HH0:30'))
write_simplified(H_tas_simple, "airspeed_fusion.txt", 'H_tas')
K = P * Transpose(H_tas) * (H_tas * P * Transpose(H_tas) + Matrix([r_tas])).inv()

K_simple = cse(K, symbols('KTAS0:30'))
write_simplified(K_simple, "airspeed_fusion.txt", "K")

P_m = P - K*H_tas*P
P_m_simple = cse(P_m, symbols('PM0:50'))
write_simplified(P_m_simple, "airspeed_fusion.txt", "_P")
 

# sideslip fusion

# compute relative wind vector in vehicle body frame
relative_wind_earth = Matrix([v_n - w_n, v_e - w_e, v_d])
R_body_to_earth = quat2Rot(q_att)
relative_wind_body = Transpose(R_body_to_earth) * relative_wind_earth
# small angle approximation of side slip model
beta_pred = relative_wind_body[1] / relative_wind_body[0]
# compute side slip observation matrix
H_beta = Matrix([beta_pred]).jacobian(state)
# simplify the result and write it to a text file in C format
H_beta_simple = cse(H_beta, symbols('HB0:30'))
write_simplified(H_beta_simple, "beta_fusion.txt", 'H_beta')
K = P * Transpose(H_beta)
denom = H_beta * P * Transpose(H_beta) + Matrix([r_beta])
denom = 1/denom.values()[0]
K = K*denom
K_simple = cse(K, symbols('KB0:30'))
write_simplified(K_simple, "beta_fusion.txt", 'K')

P_m = P - K*H_beta*P
P_m_simple = cse(P_m, symbols('PM0:50'))
write_simplified(P_m_simple, "beta_fusion.txt", "_P")


