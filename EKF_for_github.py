import numpy as np
import pandas as pd
from IPython import get_ipython
from timeit import default_timer as timer
get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt_1
import matplotlib.pyplot as plt_2
import matplotlib.pyplot as plt_3
import matplotlib.pyplot as plt_4
import matplotlib.pyplot as plt_5
import matplotlib.pyplot as plt_6
import matplotlib.pyplot as plt_7
import matplotlib.pyplot as plt_8
import matplotlib.pyplot as plt_9
import matplotlib.pyplot as plt_10
import matplotlib.pyplot as plt_11
import matplotlib.pyplot as plt_12
import matplotlib.pyplot as plt_13
import matplotlib.pyplot as plt_14
import matplotlib.pyplot as plt_15
import matplotlib.pyplot as plt_16
import matplotlib.pyplot as plt_17
plt_1.rcParams['figure.figsize'] = (15, 10)
plt_2.rcParams['figure.figsize'] = (15, 10)
plt_3.rcParams['figure.figsize'] = (15, 10)
plt_4.rcParams['figure.figsize'] = (15, 10)
plt_5.rcParams['figure.figsize'] = (15, 10)
plt_6.rcParams['figure.figsize'] = (15, 10)
plt_7.rcParams['figure.figsize'] = (15, 10)
plt_8.rcParams['figure.figsize'] = (15, 10)
plt_9.rcParams['figure.figsize'] = (15, 10)
plt_10.rcParams['figure.figsize'] = (15, 10)
plt_11.rcParams['figure.figsize'] = (15, 10)
plt_12.rcParams['figure.figsize'] = (15, 10)
plt_13.rcParams['figure.figsize'] = (15, 10)
plt_14.rcParams['figure.figsize'] = (15, 10)
plt_15.rcParams['figure.figsize'] = (15, 10)
plt_16.rcParams['figure.figsize'] = (15, 10)
plt_17.rcParams['figure.figsize'] = (15, 10)
import math
import random
from numpy.linalg import norm, inv
from scipy.sparse.linalg import lsmr

def GPS_noise(mu, sigma_x, sigma_y, samples):
    
        
    white_x = np.random.normal(mu, sigma_x, size=samples) 
    white_y = np.random.normal(mu, sigma_y, size=samples)
    
    '''
    temp_pink_x = np.fft.fft(white_x)
    temp_pink_y = np.fft.fft(white_y)

    for k in range(int(temp_pink_x.size/2)+1):
        if (k!=0):
            temp_pink_x[k]/=math.sqrt(k)

    pink_x = np.fft.ifft(np.concatenate((temp_pink_x[:int(temp_pink_x.size/2 + 1)], np.conj(temp_pink_x[int(temp_pink_x.size/2 + 1)-1:1:-1])))).real


    for k in range(int(temp_pink_y.size/2)):
        if (k!=0):
            temp_pink_y[k]/=math.sqrt(k)

    pink_y = np.fft.ifft(np.concatenate((temp_pink_y[:int(temp_pink_y.size/2 + 1)], np.conj(temp_pink_y[int(temp_pink_y.size/2 + 1)-1:1:-1])))).real

   
    noise_x = white_x + pink_x
    noise_y = white_y + pink_y
    '''
    return white_x, white_y


def CalculateAzimuthAngle (x_observer, x_target, y_observer, y_target):
  
    if (x_observer == x_target):
        
        if (y_observer < y_target):
            
            return math.radians(0)
        
        else:
            
            return math.radians(180)
        
    if (y_observer == y_target):
        
        if (x_observer < x_target):
            
            return math.radians(90)
        
        else:
            
            return math.radians(270)
        
    if (x_target > x_observer and y_target > y_observer):
        
        a = (math.atan((x_target-x_observer)/(y_target-y_observer)))
        
    elif (x_target > x_observer and y_target < y_observer):
        
        a = math.radians(90) + (math.atan((y_observer-y_target)/(x_target-x_observer)))
        
        
    elif (x_target < x_observer and y_target < y_observer):
        
        a = math.radians(180) + (math.atan((x_observer-x_target)/(y_observer-y_target)))
        
        
    elif (x_target < x_observer and y_target > y_observer):
        
        a = math.radians(270) + (math.atan((y_target-y_observer)/(x_observer-x_target)))
        
    
    return a

def CalculateAoA (x_observer, x_target, y_observer, y_target):
    
    if (x_observer == x_target or y_observer == y_target):
        return 0
    
    a = (math.atan((y_target-y_observer)/(x_target-x_observer)))
    
    return a

def Solve_The_System (true_Points, L, anchors_index, anchors, delta_X, delta_Y):
    
    L_bar = np.zeros((L[:,0].size, anchors[:,0].size))

    anchors_matrix = np.zeros((anchors[:,0].size, delta_X.size))

    for i in range(anchors_index.size):
        anchors_matrix[i,int(anchors_index[i])] = 1
    
   
    L_bar = np.concatenate([L, anchors_matrix])

    b = np.zeros((delta_X.size + anchors_index.size))
    q = np.zeros((delta_Y.size + anchors_index.size))

    for i in range(delta_X.size):
        
        b[i] = delta_X[i]
        q[i] = delta_Y[i]

    k = delta_X.size
    l = delta_Y.size

    for i in range(anchors[:,0].size):
        b[k] = anchors[i,0]
        q[l] = anchors[i,1]
        k += 1
        l += 1
   
    X = np.zeros((true_Points[:,0].size))
    Y = np.zeros((true_Points[:,0].size))

    
    X = lsmr(L_bar,b)[0]
    Y = lsmr(L_bar,q)[0]
    
    lapl_Points = np.zeros((X.size, 3))

    lapl_Points[:,0] = X
    lapl_Points[:,1] = Y

    reconstr_error = np.zeros((X.size))

    for i in range(reconstr_error.size):
        
        reconstr_error[i] = norm(np.array([true_Points[i,0] - X[i], true_Points[i,1] - Y[i]]),2)
        
    return lapl_Points, reconstr_error, L_bar    

def Create_Clusters(true_X, A, Deg):
    
    clusters = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
    clusters_size = np.zeros((true_X.shape[0], true_X.shape[1]))
    clusters_index = np.zeros((true_X.shape[0], true_X.shape[1]))
    
    for l in range (true_X.shape[1]):
        
        print (l)
        is_checked = np.zeros(true_X.shape[0])
        
        for i in range (true_X.shape[0]):
            
            if (is_checked[i] == 0):
                
                is_checked[i] = 1
                
                clusters[i,l] = []
                
                clusters[i,l].append(i)
                
                clusters_index[i,l] = i
                
                if (Deg[l,i,i] > 0):
                    
                    neighbours = []
                    
                    neighbours = (np.argwhere(1 == A[l,i,:]))
                    
                    for j in neighbours:
                        
                        if (is_checked[j] == 0):
                            
                            is_checked[j] = 1
                            
                            clusters[i,l].append(j)
                            
                            clusters_index[j,l] = i
                            
                    while (len(neighbours) > 0):
                        
                        temp_neighbours = []
                        
                        for u in neighbours:
                            
                            next_round_neighbours = []
                            
                            next_round_neighbours = (np.argwhere(1 == A[l,u[0],:]))
                            
                            for k in next_round_neighbours:
                               
                                if (is_checked[k] == 0):
                                    
                                    is_checked[k] = 1
                                    
                                    clusters[i,l].append(k.item())
                                    
                                    clusters_index[k.item(),l] = i
                                    
                                    temp_neighbours.append(k)
                                    
                        neighbours = []
                        
                        neighbours = temp_neighbours
                        
                else:
                        
                    clusters[i,l] = []
                
                    clusters[i,l].append(0)
                    
                    clusters_index[i,l] = 5000
                        
            else:
                
                clusters[i,l] = []
                
                clusters[i,l].append(0)
                    
        
        for i in range(clusters.shape[0]):
        
            if (sum(clusters[i,l]) > 0):
                
                for j in range (len(clusters[i,l])):
                    
                    if (np.isscalar(clusters[i,l][j]) == 0):
                        
                        clusters[i,l][j] = clusters[i,l][j].item()
                
                clusters[i,l].sort()
                clusters_size[i,l] = len(clusters[i,l])
                
            else:
                
                clusters_size[i,l] = 0
                
      
    return clusters, clusters_size, clusters_index


def EKF_with_GPS(flag_for_EKF_alone, mhi_alone_KF, mhi_alone_hat_KF, Sigma_alone_KF, Sigma_alone_hat_KF, Deg, true_X, true_Y, idx_time, idx_test_cluster, KM_comp_x, KM_comp_y, sigma_x, sigma_y):
   
    if (idx_time == 0):
                
        mhi_alone_KF[idx_test_cluster, idx_time][0] = noise_X[idx_test_cluster,idx_time]
        mhi_alone_KF[idx_test_cluster, idx_time][1] = noise_Y[idx_test_cluster,idx_time]
                   
        Sigma_alone_KF[idx_test_cluster, idx_time] = np.eye(2)
                            
        EKF_recon_X = mhi_alone_KF[idx_test_cluster, idx_time][0]
        EKF_recon_Y = mhi_alone_KF[idx_test_cluster, idx_time][1]
                                
                
        ekf_error = norm(np.array([EKF_recon_X - true_X[idx_test_cluster,idx_time], EKF_recon_Y - true_Y[idx_test_cluster,idx_time]]))
                
    else:
         
        flag_for_EKF_alone[idx_test_cluster, idx_time] = 1
        
        state = 2
                
        if (Deg[idx_time-1, idx_test_cluster, idx_test_cluster] == 0):
                    
                    
            G_alone = np.eye(state)
                    
            mhi_alone_hat_KF[idx_test_cluster, idx_time][0] = mhi_alone_KF[idx_test_cluster, idx_time-1].T[0] + KM_comp_x[idx_test_cluster, idx_time]
                                    
            mhi_alone_hat_KF[idx_test_cluster, idx_time][1] = mhi_alone_KF[idx_test_cluster, idx_time-1].T[1] + KM_comp_y[idx_test_cluster, idx_time]
                        
            Sigma_alone_hat_KF[idx_test_cluster, idx_time] = G_alone@Sigma_alone_KF[idx_test_cluster, idx_time-1]@G_alone.T + (1**2)*np.eye(state)
                                
            Kalman_gain_alone = np.zeros((state, size_of_measurement_model))
            
            H_alone = np.eye(state)
                     
            Q_alone = np.eye(state)
                    
            Q_alone[0,0] *= sigma_x**2
                    
            Q_alone[1,1] *= sigma_y**2
                    
            Kalman_gain_alone = Sigma_alone_hat_KF[idx_test_cluster, idx_time] @ H_alone.T @ (inv(H_alone @ Sigma_alone_hat_KF[idx_test_cluster, idx_time] @ H_alone.T + Q_alone))
                                
            z_alone, h_alone = np.zeros((H_alone.shape[0],1)), np.zeros((H_alone.shape[0],1))
                    
            z_alone = np.array([noise_X[idx_test_cluster,idx_time], noise_Y[idx_test_cluster,idx_time]])                        
                    
            h_alone = np.array([mhi_alone_hat_KF[idx_test_cluster, idx_time][0], mhi_alone_hat_KF[idx_test_cluster, idx_time][1]])
           
            measurement_mat_alone = np.zeros((H_alone.shape[0],1))
                                            
            measurement_mat_alone = np.reshape(z_alone - h_alone, (H_alone.shape[0],1))
                                            
            aggreg_vec_alone = Kalman_gain_alone @ measurement_mat_alone
                                            
            aggreg_matrix_alone = Kalman_gain_alone@H_alone
                                
            mhi_alone_KF[idx_test_cluster, idx_time] = mhi_alone_hat_KF[idx_test_cluster, idx_time] + aggreg_vec_alone.T
                                    
            Sigma_alone_KF[idx_test_cluster, idx_time] = Sigma_alone_hat_KF[idx_test_cluster, idx_time] - aggreg_matrix_alone@Sigma_alone_hat_KF[idx_test_cluster, idx_time]
                    
            EKF_recon_X = mhi_alone_KF[idx_test_cluster, idx_time].T[0]
            EKF_recon_Y = mhi_alone_KF[idx_test_cluster, idx_time].T[1]
                                    
            ekf_error = norm(np.array([EKF_recon_X - true_X[idx_test_cluster,idx_time], EKF_recon_Y - true_Y[idx_test_cluster,idx_time]]))
               
        else:
                    
            mhi_alone_KF[idx_test_cluster, idx_time][0] = noise_X[idx_test_cluster,idx_time]
            mhi_alone_KF[idx_test_cluster, idx_time][1] = noise_Y[idx_test_cluster,idx_time]
                       
            Sigma_alone_KF[idx_test_cluster, idx_time] = np.eye(2)
                                
            EKF_recon_X = mhi_alone_KF[idx_test_cluster, idx_time][0]
            EKF_recon_Y = mhi_alone_KF[idx_test_cluster, idx_time][1]
                                    
            ekf_error = norm(np.array([EKF_recon_X - true_X[idx_test_cluster,idx_time], EKF_recon_Y - true_Y[idx_test_cluster,idx_time]]))
    
    return ekf_error, EKF_recon_X, EKF_recon_Y



def EKF_with_Lapl(mhi_KF, mhi_hat_KF, Sigma_KF, Sigma_hat_KF, clusters, idx_time, idx_test_cluster, KM_comp_x, KM_comp_y, L_bar, test_lapl_Points, delta_X, delta_Y, G, Q, R, test_A, anchors):    

    
    state = int(clusters_size[idx_test_cluster, idx_time])
                
    if (idx_time == 0): #if time = 0, initialize Kalman with other estimations 
       
        mhi_KF[idx_test_cluster, idx_time] = np.concatenate([test_lapl_Points[:,0], test_lapl_Points[:,1]])
                     
        Sigma_KF[idx_test_cluster, idx_time] = np.eye(2*state)
            
                        
    else:
                        
        if (clusters[idx_test_cluster, idx_time] != clusters[idx_test_cluster, idx_time-1]): #if the cluster is not the same as was the previous time instant, initialize with other estimations
            
            mhi_KF[idx_test_cluster, idx_time] = np.concatenate([test_lapl_Points[:,0], test_lapl_Points[:,1]])
                        
            Sigma_KF[idx_test_cluster, idx_time] = np.eye(2*state)
                            
                
        else:
                
            
            u = 0
            for i in clusters[idx_test_cluster][idx_time]:
                                
                mhi_hat_KF[idx_test_cluster, idx_time][u] = mhi_KF[idx_test_cluster, idx_time-1].T[u] + KM_comp_x[i, idx_time]
                                
                mhi_hat_KF[idx_test_cluster, idx_time][u+state] = mhi_KF[idx_test_cluster, idx_time-1].T[u+state] + KM_comp_y[i, idx_time]
                                
                u += 1
                            
            Sigma_hat_KF[idx_test_cluster, idx_time] = G[idx_test_cluster, idx_time]@Sigma_KF[idx_test_cluster, idx_time-1]@G[idx_test_cluster, idx_time].T + R[idx_test_cluster, idx_time]
                            
            Kalman_gain = np.zeros((2*state, 4*state))
        
            H = np.zeros((Kalman_gain.shape[1], Kalman_gain.shape[0]))
                            
            H[:2*state,:state] = np.copy(L_bar[idx_test_cluster, idx_time])
                            
            H[2*state:,state:] = np.copy(L_bar[idx_test_cluster, idx_time])
                            
            Kalman_gain = Sigma_hat_KF[idx_test_cluster, idx_time] @ H.T @ (inv(H @ Sigma_hat_KF[idx_test_cluster, idx_time] @ H.T + Q[idx_test_cluster, idx_time]))
                            
            z, h = np.zeros((H.shape[0],1)), np.zeros((H.shape[0],1))
                        
            z = np.concatenate([delta_X, anchors[:,0], delta_Y, anchors[:,1]])
                            
            h_delta_X = np.zeros(state)
                            
            h_delta_Y = np.zeros(state)
                    
            for u in range(state):
                for l in range(state):
                    if (test_A[u,l] == 1):
                        h_delta_X[u] += mhi_hat_KF[idx_test_cluster, idx_time][u] - mhi_hat_KF[idx_test_cluster, idx_time][l]
                        h_delta_Y[u] += mhi_hat_KF[idx_test_cluster, idx_time][u+state] - mhi_hat_KF[idx_test_cluster, idx_time][l+state]
                                        
            h = np.concatenate([h_delta_X, mhi_hat_KF[idx_test_cluster, idx_time][:state], h_delta_Y, mhi_hat_KF[idx_test_cluster, idx_time][state:]])
       
            measurement_mat = np.zeros((H.shape[0],1))
                                        
            measurement_mat = np.reshape(z - h, (H.shape[0],1))
           
            aggreg_vec = Kalman_gain @ measurement_mat
                                        
            aggreg_matrix = Kalman_gain@H
                            
            mhi_KF[idx_test_cluster, idx_time] = mhi_hat_KF[idx_test_cluster, idx_time] + aggreg_vec.T
                                
            Sigma_KF[idx_test_cluster, idx_time] = Sigma_hat_KF[idx_test_cluster, idx_time] - aggreg_matrix@Sigma_hat_KF[idx_test_cluster, idx_time]
                                
    return mhi_KF[idx_test_cluster][idx_time], Sigma_KF[idx_test_cluster][idx_time]


###### Mean and variance of white Gaussian noise ######
mu = 0
sigma_x = 3
sigma_y = 2.5
sigma_gyro = 0.2
sigma_d = 1
sigma_a = 4
###### Mean and variance of white Gaussian noise ######

Dt = 0.3 #time step generated by the simulation data


range_of_tranceivers = 20

number_of_connected_neighbours = 6


time_instances = 500 #duration of simulation horizon

#maximum number of simultation horizon: 573 time instances

temp_locations = pd.read_excel('locations_4.xlsx')

locations = np.array(temp_locations)

locations = np.delete(locations, 0, 1)

true_X = np.zeros((200, time_instances))
true_Y = np.zeros((200, time_instances))
true_Z = np.zeros((200, time_instances))

vel_X = np.zeros((true_X.shape[0], time_instances))
vel_Y = np.zeros((true_X.shape[0], time_instances))
vel_Z = np.zeros((true_X.shape[0], time_instances))

ang_vel_X = np.zeros((true_X.shape[0], time_instances))
ang_vel_Y = np.zeros((true_X.shape[0], time_instances))
ang_vel_Z = np.zeros((true_X.shape[0], time_instances))

heading = np.zeros((true_X.shape[0], time_instances))

heading_deg = np.zeros((true_X.shape[0], time_instances))

gps_error = np.zeros((true_X.shape[0], true_X.shape[1]))
lapl_error = np.zeros((true_X.shape[0], true_X.shape[1]))
cgcl_ekf_error = np.zeros((true_X.shape[0], true_X.shape[1]))
ekf_alone_error = np.zeros((true_X.shape[0], true_X.shape[1]))

mse_gps_error = np.zeros(true_X.shape[1])
mse_lapl_error = np.zeros(true_X.shape[1])
mse_cgcl_ekf_error = np.zeros(true_X.shape[1])

max_gps_error = np.zeros(true_X.shape[1])
max_lapl_error = np.zeros(true_X.shape[1])
max_cgcl_ekf_error = np.zeros(true_X.shape[1])

####### Create the matrices of location, velocity, angular velocity, and heading angle for every vehicle #######
for i in range (true_X.shape[0]):
    
    step = 0
    print ('i: ', i)
    for j in range (true_X.shape[1]):
        
        true_X[i,j] = np.copy(locations[i+step,1])
        true_Y[i,j] = np.copy(locations[i+step,2])
        true_Z[i,j] = np.copy(locations[i+step,3])
        
        vel_X[i,j] = np.copy(locations[i+step,5])
        vel_Y[i,j] = np.copy(locations[i+step,6])
        vel_Z[i,j] = np.copy(locations[i+step,7])
        
        ang_vel_X[i,j] = np.copy(locations[i+step,11]) 
        ang_vel_Y[i,j] = np.copy(locations[i+step,12]) 
        ang_vel_Z[i,j] = np.copy(locations[i+step,13])
        
        heading[i,j] = math.radians(np.copy(locations[i+step,4])) + np.random.normal(0,math.radians(sigma_gyro))
        
        heading_deg[i,j] = (np.copy(locations[i+step,4]))
        
        step += true_X.shape[0]

velocity = np.zeros((true_X.shape[0], time_instances))
angular = np.zeros((true_X.shape[0], time_instances))

for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]): 
        
        n_ang_x = np.random.normal(0, math.radians(sigma_gyro))
        
        n_ang_y = np.random.normal(0, math.radians(sigma_gyro))
        
        angular[i,j] = math.sqrt((ang_vel_X[i,j] + n_ang_x)**2 + (ang_vel_Y[i,j] + n_ang_y)**2)
        

        
for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
       
        n_vel_x = np.random.normal(0, 0.1*abs(vel_X[i,j]))
        
        n_vel_y = np.random.normal(0, 0.1*abs(vel_Y[i,j]))
        
        velocity[i,j] = math.sqrt( (vel_X[i,j] + n_vel_x)**2 + (vel_Y[i,j] + n_vel_y)**2 )
        
####### Create the matrices of location, velocity, angular velocity, and heading angle for every vehicle #######

####### Create the kinematic model for every vehicle according to velocity, angular velocity and heading #######        
temp_traj_x = np.zeros((true_X.shape[0], true_X.shape[1]))  
temp_traj_y = np.zeros((true_X.shape[0], true_X.shape[1]))

KM_comp_x = np.zeros((true_X.shape[0], true_X.shape[1]))  
KM_comp_y = np.zeros((true_X.shape[0], true_X.shape[1]))

for j in range (true_X.shape[0]):
    
    for i in range (true_X.shape[1]):
        
        if (i == 0):
            temp_traj_x[j,i] = true_X[j,i]
            temp_traj_y[j,i] = true_Y[j,i]
          
        else:
            
            if ((angular[j,i]) == 0):
                
                
                KM_comp_x[j,i] = (velocity[j,i])*math.cos(heading[j,i])*Dt 
                KM_comp_y[j,i] = (velocity[j,i])*math.sin(heading[j,i])*Dt 
            
                temp_traj_x[j,i] = temp_traj_x[j,i-1] + KM_comp_x[j,i]
                temp_traj_y[j,i] = temp_traj_y[j,i-1] + KM_comp_y[j,i]
                
            
            else:    
                
                KM_comp_x[j,i] = (-(velocity[j,i])/(angular[j,i]))*np.sin(heading[j,i]) + ((velocity[j,i] )/(angular[j,i]))*np.sin(heading[j,i] + angular[j,i]*Dt) 
                KM_comp_y[j,i] = ((velocity[j,i])/(angular[j,i]))*np.cos(heading[j,i]) + (-(velocity[j,i] )/(angular[j,i]))*np.cos(heading[j,i] + angular[j,i]*Dt) 
                
                temp_traj_x[j,i] = temp_traj_x[j,i-1] + KM_comp_x[j,i]
                temp_traj_y[j,i] = temp_traj_y[j,i-1] + KM_comp_y[j,i]
            
####### Create the kinematic model for every vehicle according to velocity, angular velocity and heading #######

####### Create GPS measurements by adding white Gaussian noise to the true location #######                
noise_X = np.zeros((true_X.shape[0], true_X.shape[1]))
noise_Y = np.zeros((true_X.shape[0], true_X.shape[1]))


for i in range (true_X.shape[1]):
    
    gps_noise_x = np.zeros(true_X.shape[0])
    gps_noise_y = np.zeros(true_X.shape[0])
    
    gps_noise_x = GPS_noise(mu, sigma_x, sigma_y, true_X.shape[0])[0]
    gps_noise_y = GPS_noise(mu, sigma_x, sigma_y, true_X.shape[0])[1]
    
    noise_X[:,i] = true_X[:,i] + gps_noise_x
    noise_Y[:,i] = true_Y[:,i] + gps_noise_y
    
####### Create GPS measurements by adding white Gaussian noise to the true location #######

####### Create distance matrix (D), adjacency matrix (A) and degree matrix (Deg) ####### 
    
list_of_vehicles = np.arange(true_X.shape[0])

D = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

noisy_D = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

A = np.zeros((D.shape[0], D.shape[1], D.shape[2]))

for i in range (true_X.shape[1]):
    l = 0
    for j in (list_of_vehicles):
        u = 0
        
        for k in (list_of_vehicles):
           
            D[i,l,u] = norm(np.array([true_X[j,i]-true_X[k,i], true_Y[j,i]-true_Y[k,i]]),2)
            
            noisy_D[i,l,u] = D[i,l,u] + np.random.normal(mu, sigma_d)
            
            if (noisy_D[i,l,u] <= 0):
                noisy_D[i,l,u] *= -1
                
            if (D[i,l,u] <= range_of_tranceivers and D[i,l,u] > 0):
                
                A[i,l,u] = 1
                
            u += 1
         
        l += 1
                
                

Deg = np.zeros((D.shape[0], D.shape[1], D.shape[2]))


for i in range (true_X.shape[1]):
    for j in range (D.shape[1]):
        count = 0
        for k in range (D.shape[1]):
            if (A[i,j,k] == 1):
                count += 1
                if (count > number_of_connected_neighbours):
                    A[i,j,k] = 0
                    A[i,k,j] = 0
                    D[i,j,k] = 0
                    D[i,k,j] = 0


       
for i in range (true_X.shape[1]):
    
    for j in range (D.shape[1]):
        
        Deg[i,j,j] = np.sum(A[i,j,:])
        
####### Create distance matrix (D), adjacency matrix (A) and degree matrix (Deg) #######

    
    
Azim_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

noisy_Azim_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

Concatenated_CGCL_Points_X = np.zeros((2*true_X.shape[0], true_X.shape[1]))
Concatenated_CGCL_Points_Y = np.zeros((2*true_X.shape[0], true_X.shape[1]))

EKF_CGCL_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_CGCL_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))


EKF_alone_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_alone_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

####### Calculate the azimuth angle between the vehicles #######

for i in range (true_X.shape[1]):
    
    l = 0
    for j in (list_of_vehicles):
        u = 0
        
        for k in (list_of_vehicles):
            
            if (A[i,l,u] == 1):
                
                Azim_Angle[i,l,u] = CalculateAzimuthAngle(true_X[j,i], true_X[k,i], true_Y[j,i], true_Y[k,i])
                
                noisy_Azim_Angle[i,l,u] = Azim_Angle[i,l,u] + np.random.normal(mu, math.radians(sigma_a))
                
                    
            u += 1
         
        l += 1
        
####### Calculate the azimuth angle between the vehicles #######

####### Create the clusters of vehicles for the simulation horizon #######
        
clusters, clusters_size, clusters_index = Create_Clusters(true_X, A, Deg)

####### Create the clusters of vehicles for the simulation horizon #######

size_of_measurement_model = 2

L_bar = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

####### Initialize Kalman Filters #######
mhi_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

mhi_UKF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_hat_UKF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_UKF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_hat_UKF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Q = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

R = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

G = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

mhi_alone_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_alone_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_alone_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_alone_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

flag_for_EKF_alone = np.zeros((true_X.shape[0], true_X.shape[1]))

for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
        
        mhi_KF[i,j] = np.zeros((int(2*clusters_size[i,j])), dtype=np.object) 
        
        mhi_KF[i,j] = mhi_KF[i,j].astype(float)
        
        mhi_hat_KF[i,j] = np.zeros((int(2*clusters_size[i,j])), dtype=np.object)
        
        mhi_hat_KF[i,j] = mhi_hat_KF[i,j].astype(float)
        
        Sigma_KF[i,j] = np.zeros((int(2*clusters_size[i,j]), int(2*clusters_size[i,j])), dtype=np.object) 
        
        Sigma_KF[i,j] = Sigma_KF[i,j].astype(float)
        
        Sigma_hat_KF[i,j] = np.zeros((int(2*clusters_size[i,j]), int(2*clusters_size[i,j])), dtype=np.object)
        
        Sigma_hat_KF[i,j] = Sigma_hat_KF[i,j].astype(float)
        

        Q[i,j] = np.zeros((int(4*clusters_size[i,j]), int(4*clusters_size[i,j])), dtype=np.object)
        
        Q[i,j] = np.eye(int(4*clusters_size[i,j]))
        
        Q[i,j] = Q[i,j].astype(float)
        
        R[i,j] = np.zeros((int(2*clusters_size[i,j]), int(2*clusters_size[i,j])), dtype=np.object)
        
        R[i,j] = np.eye(int(2*clusters_size[i,j]))
        
        R[i,j] = R[i,j].astype(float)
        
        G[i,j] = np.zeros((int(2*clusters_size[i,j]), int(2*clusters_size[i,j])), dtype=np.object)
        
        G[i,j][:int(clusters_size[i,j]),:int(clusters_size[i,j])] = 1
        
        G[i,j] = G[i,j].astype(float)
        
        G[i,j][int(clusters_size[i,j]):,int(clusters_size[i,j]):] = 1
        
        G[i,j] = G[i,j].astype(float)
        
        L_bar[i,j] = np.zeros((int(2*clusters_size[i,j]), int(clusters_size[i,j])), dtype=np.object)
        
        L_bar[i,j] = L_bar[i,j].astype(float)
        
        mhi_alone_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_KF[i,j] = mhi_alone_KF[i,j].astype(float)
        
        mhi_alone_hat_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_hat_KF[i,j] = mhi_alone_hat_KF[i,j].astype(float)
        
        Sigma_alone_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_KF[i,j] = Sigma_alone_KF[i,j].astype(float)
        
        Sigma_alone_hat_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_hat_KF[i,j] = Sigma_alone_hat_KF[i,j].astype(float)
        
####### Initialize Kalman Filters #######
        
####### GPS error #######   
        
for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
        
        gps_error[i,j] = norm(np.array([true_X[i,j] - noise_X[i,j] , true_Y[i,j] - noise_Y[i,j]]), 2)

for i in range (true_X.shape[1]):
    
    mse_gps_error[i] = np.sum(gps_error[:,i]**2)/true_X.shape[0]
    
    max_gps_error[i] = np.max(gps_error[:,i])
    
####### GPS error #######
    
    
for idx_time in range (true_X.shape[1]):
    
    #if (idx_time == 0):
    print ('Time index: ', idx_time)
    
    if (np.array_equal(A[idx_time,:,:],A[idx_time,:,:].T) == 0):
        print ('Problem with A')
        
    for idx_test_cluster in range(true_X.shape[0]):

       
##################!!!!!!!!!!!!!! EKF with GPS measurements only (when a vehicle doesn't belong to any cluster) !!!!!!!!################# 
        
        if (idx_time == 0):
                
            mhi_alone_KF[idx_test_cluster, idx_time][0] = noise_X[idx_test_cluster,idx_time]
            mhi_alone_KF[idx_test_cluster, idx_time][1] = noise_Y[idx_test_cluster,idx_time]
                   
            Sigma_alone_KF[idx_test_cluster, idx_time] = np.eye(2)
                            
            EKF_alone_recon_X[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time][0]
            EKF_alone_recon_Y[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time][1]
                                
            ekf_alone_error[idx_test_cluster,idx_time] = norm(np.array([EKF_alone_recon_X[idx_test_cluster,idx_time] - true_X[idx_test_cluster,idx_time], EKF_alone_recon_Y[idx_test_cluster,idx_time] - true_Y[idx_test_cluster,idx_time]]))
                
        else:
                 
            state = 2
                
            if (Deg[idx_time-1, idx_test_cluster, idx_test_cluster] == 0):
                    
                flag_for_EKF_alone[idx_test_cluster, idx_time] = 1
                    
                G_alone = np.eye(state)
                    
                mhi_alone_hat_KF[idx_test_cluster, idx_time][0] = mhi_alone_KF[idx_test_cluster, idx_time-1].T[0] + KM_comp_x[idx_test_cluster, idx_time]
                                    
                mhi_alone_hat_KF[idx_test_cluster, idx_time][1] = mhi_alone_KF[idx_test_cluster, idx_time-1].T[1] + KM_comp_y[idx_test_cluster, idx_time]
                        
                Sigma_alone_hat_KF[idx_test_cluster, idx_time] = G_alone@Sigma_alone_KF[idx_test_cluster, idx_time-1]@G_alone.T + (1**2)*np.eye(state)
                                
                Kalman_gain_alone = np.zeros((state, size_of_measurement_model))
            
                H_alone = np.eye(state)
                     
                Q_alone = np.eye(state)
                    
                Q_alone[0,0] *= 1**2
                    
                Q_alone[1,1] *= 1**2
                    
                Kalman_gain_alone = Sigma_alone_hat_KF[idx_test_cluster, idx_time] @ H_alone.T @ (inv(H_alone @ Sigma_alone_hat_KF[idx_test_cluster, idx_time] @ H_alone.T + Q_alone))
                                
                z_alone, h_alone = np.zeros((H_alone.shape[0],1)), np.zeros((H_alone.shape[0],1))
                    
                z_alone = np.array([noise_X[idx_test_cluster,idx_time], noise_Y[idx_test_cluster,idx_time]])                        
                    
                h_alone = np.array([mhi_alone_hat_KF[idx_test_cluster, idx_time][0], mhi_alone_hat_KF[idx_test_cluster, idx_time][1]])
           
                measurement_mat_alone = np.zeros((H_alone.shape[0],1))
                                            
                measurement_mat_alone = np.reshape(z_alone - h_alone, (H_alone.shape[0],1))
                                            
                aggreg_vec_alone = Kalman_gain_alone @ measurement_mat_alone
                                            
                aggreg_matrix_alone = Kalman_gain_alone@H_alone
                                
                mhi_alone_KF[idx_test_cluster, idx_time] = mhi_alone_hat_KF[idx_test_cluster, idx_time] + aggreg_vec_alone.T
                                    
                Sigma_alone_KF[idx_test_cluster, idx_time] = Sigma_alone_hat_KF[idx_test_cluster, idx_time] - aggreg_matrix_alone@Sigma_alone_hat_KF[idx_test_cluster, idx_time]
                    
                EKF_alone_recon_X[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time].T[0]
                EKF_alone_recon_Y[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time].T[1]
                                    
                ekf_alone_error[idx_test_cluster,idx_time] = norm(np.array([EKF_alone_recon_X[idx_test_cluster,idx_time] - true_X[idx_test_cluster,idx_time], EKF_alone_recon_Y[idx_test_cluster,idx_time] - true_Y[idx_test_cluster,idx_time]]))
               
            else:
                    
                mhi_alone_KF[idx_test_cluster, idx_time][0] = noise_X[idx_test_cluster,idx_time]
                mhi_alone_KF[idx_test_cluster, idx_time][1] = noise_Y[idx_test_cluster,idx_time]
                       
                Sigma_alone_KF[idx_test_cluster, idx_time] = np.eye(2)
                                
                EKF_alone_recon_X[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time][0]
                EKF_alone_recon_Y[idx_test_cluster,idx_time] = mhi_alone_KF[idx_test_cluster, idx_time][1]
                                    
                ekf_alone_error[idx_test_cluster,idx_time] = norm(np.array([EKF_alone_recon_X[idx_test_cluster,idx_time] - true_X[idx_test_cluster,idx_time], EKF_alone_recon_Y[idx_test_cluster,idx_time] - true_Y[idx_test_cluster,idx_time]]))
        
       
##################!!!!!!!!!!!!!! EKF with GPS measurements only (when a vehicle doesn't belong to any cluster) !!!!!!!!#################
         
##################!!!!!!!!!!!!!! Centralized Laplacian !!!!!!!!#################           
        if (Deg[idx_time, idx_test_cluster, idx_test_cluster] == 0): #if a vehicle doesn't belong to any cluster
            
            lapl_error[idx_test_cluster, idx_time] = gps_error[idx_test_cluster, idx_time]
            
            Concatenated_CGCL_Points_X[idx_test_cluster,idx_time] = noise_X[idx_test_cluster,idx_time]
            Concatenated_CGCL_Points_Y[idx_test_cluster,idx_time] = noise_Y[idx_test_cluster,idx_time]
            
            cgcl_ekf_error[idx_test_cluster, idx_time] = np.copy(ekf_alone_error[idx_test_cluster,idx_time])
            
            EKF_CGCL_recon_X[idx_test_cluster,idx_time] = np.copy(EKF_alone_recon_X[idx_test_cluster,idx_time])
        
            EKF_CGCL_recon_Y[idx_test_cluster,idx_time] = np.copy(EKF_alone_recon_Y[idx_test_cluster,idx_time])
            
            
            
        else:
            
            if (sum(clusters[idx_test_cluster][idx_time]) > 0): #if cluster contains vehicles
                
                test_Points = np.zeros((len(clusters[idx_test_cluster][idx_time]) , 2))
                noisy_test_Points = np.zeros((len(clusters[idx_test_cluster][idx_time]) , 2))
                
                u = 0
                for i in (clusters[idx_test_cluster][idx_time]):
                    
                    test_Points[u,0] = true_X[i,idx_time]
                    test_Points[u,1] = true_Y[i,idx_time]
                    
                    noisy_test_Points[u,0] = noise_X[i,idx_time]
                    noisy_test_Points[u,1] = noise_Y[i,idx_time]
                    
                    u += 1
                
                test_gps_error = np.zeros(len(clusters[idx_test_cluster][idx_time]))
                
                for i in range (test_Points.shape[0]):
                    test_gps_error[i] = norm(np.array([test_Points[i,0]-noisy_test_Points[i,0], test_Points[i,1]-noisy_test_Points[i,1]]),2)
                    
                delta_X = np.zeros((test_Points.shape[0]))
                delta_Y = np.zeros((test_Points.shape[0]))
                
                u = 0
                
                for i in clusters[idx_test_cluster][idx_time]:
                    for j in clusters[idx_test_cluster][idx_time]:
                        
                        if (A[idx_time,i,j] == 1):
                                
                                delta_X[u] += -abs(noisy_D[idx_time,i,j])*math.sin(abs(noisy_Azim_Angle[idx_time,i,j]))
                                delta_Y[u] += -abs(noisy_D[idx_time,i,j])*math.cos(abs(noisy_Azim_Angle[idx_time,i,j]))
                                
                                
                    u += 1
                                
                
                test_A = np.zeros( (test_Points.shape[0],test_Points.shape[0]) )
                test_Deg = np.zeros( (test_Points.shape[0],test_Points.shape[0]) )
                
                for i in range (test_A.shape[0]):
                    for j in range (test_A.shape[0]):
                        if (D[idx_time, clusters[idx_test_cluster][idx_time][i], clusters[idx_test_cluster][idx_time][j]] <= range_of_tranceivers and D[idx_time, clusters[idx_test_cluster][idx_time][i], clusters[idx_test_cluster][idx_time][j]] > 0):
                            test_A[i,j] = 1
                            
                for i in range (test_A.shape[0]):
                    test_Deg[i,i] = np.sum(test_A[i,:])
                
                test_L = test_Deg - test_A            
                
                anchors_size = int(1.0*clusters_size[idx_test_cluster, idx_time])
                
                anchors_index = np.arange(anchors_size)
                
                anchors = np.copy(noisy_test_Points[:anchors_size,:])
                
                test_lapl_Points = np.zeros((test_Points.shape[0], test_Points.shape[1]))
                    
                test_lapl_error = np.zeros((test_Points.shape[0]))
                    
                test_L_bar = np.zeros((test_Points.shape[0] + anchors_index.size, test_Points.shape[1]))
                    
                test_lapl_Points, test_lapl_error, test_L_bar = Solve_The_System(test_Points, test_L, anchors_index, anchors, delta_X, delta_Y)
                
               
                L_bar[idx_test_cluster,idx_time] = test_L_bar
                

                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                    lapl_error[i,idx_time] = test_lapl_error[u]
                    
                    Concatenated_CGCL_Points_X[i,idx_time] = test_lapl_Points[u,0]
                    Concatenated_CGCL_Points_Y[i,idx_time] = test_lapl_Points[u,1]
                    
                    u += 1
               
##################!!!!!!!!!!!!!! EKF with Laplacian !!!!!!!!#################
                  
                state = int(clusters_size[idx_test_cluster, idx_time])
                
                mhi_KF[idx_test_cluster][idx_time], Sigma_KF[idx_test_cluster][idx_time] = EKF_with_Lapl(mhi_KF, mhi_hat_KF, Sigma_KF, Sigma_hat_KF, clusters, idx_time, idx_test_cluster, KM_comp_x, KM_comp_y, L_bar, test_lapl_Points, delta_X, delta_Y, G, Q, R, test_A, anchors)
                
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                          
                    EKF_CGCL_recon_X[i,idx_time] = mhi_KF[idx_test_cluster, idx_time].T[u]
                    EKF_CGCL_recon_Y[i,idx_time] = mhi_KF[idx_test_cluster, idx_time].T[u+state]
                    
                    cgcl_ekf_error[i,idx_time] = norm(np.array([EKF_CGCL_recon_X[i,idx_time] - true_X[i,idx_time], EKF_CGCL_recon_Y[i,idx_time] - true_Y[i,idx_time]]))
                    
                    u += 1 
                 
               
##################!!!!!!!!!!!!!! EKF with Laplacian !!!!!!!!#################

##################!!!!!!!!!!!!!! Centralized Laplacian !!!!!!!!#################
                   

    mse_lapl_error[idx_time] = (np.sum(lapl_error[:,idx_time]**2)/true_X.shape[0])
    mse_cgcl_ekf_error[idx_time] = (np.sum(cgcl_ekf_error[:,idx_time]**2)/true_X.shape[0])
    
    
    max_lapl_error[idx_time] = np.max(lapl_error[:,idx_time])
    max_cgcl_ekf_error[idx_time] = np.max(cgcl_ekf_error[:,idx_time])


sorted_x_mse_gps_error = np.sort(mse_gps_error)
sorted_x_mse_lapl_error = np.sort(mse_lapl_error)
sorted_x_mse_cgcl_ekf_error = np.sort(mse_cgcl_ekf_error)

sorted_y_mse_gps_error = np.arange(len(np.sort(mse_gps_error)))/float(len(mse_gps_error))
sorted_y_mse_lapl_error = np.arange(len(np.sort(mse_lapl_error)))/float(len(mse_lapl_error))
sorted_y_mse_cgcl_ekf_error = np.arange(len(np.sort(mse_cgcl_ekf_error)))/float(len(mse_cgcl_ekf_error))

color_index = random.uniform(0, 1)
color_index = random.uniform(0, 1)
color_index = random.uniform(0, 1)
    
plt_12.figure(12)
plt_12.plot(sorted_x_mse_gps_error, sorted_y_mse_gps_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_cgcl_ekf_error, sorted_y_mse_cgcl_ekf_error, 'k*-', label="CCEKF", linewidth = 4, markersize = 6)
plt_12.xticks(fontsize=28)
plt_12.yticks(fontsize=28)
plt_12.tick_params(direction='out', length=8)
plt_12.grid(b=True)
plt_12.legend(facecolor='white', fontsize = 27 )
plt_12.xlabel('Localization Mean Square Error [m$^2$]', fontsize = 35)
plt_12.ylabel('CDF', fontsize = 35)
plt_12.show()

sorted_x_max_gps_error = np.sort(max_gps_error)
sorted_x_max_lapl_error = np.sort(max_lapl_error)
sorted_x_max_cgcl_ekf_error = np.sort(max_cgcl_ekf_error)

sorted_y_max_gps_error = np.arange(len(np.sort(max_gps_error)))/float(len(max_gps_error))
sorted_y_max_lapl_error = np.arange(len(np.sort(max_lapl_error)))/float(len(max_lapl_error))
sorted_y_max_cgcl_ekf_error = np.arange(len(np.sort(max_cgcl_ekf_error)))/float(len(max_cgcl_ekf_error))

plt_13.figure(13)
plt_13.plot(sorted_x_max_gps_error, sorted_y_max_gps_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_cgcl_ekf_error, sorted_y_max_cgcl_ekf_error, 'k*-', label="CCEKF", linewidth = 4, markersize = 6)
plt_13.xticks(fontsize=28)
plt_13.yticks(fontsize=28)
plt_13.tick_params(direction='out', length=8)
plt_13.grid(b=True)
plt_13.legend(facecolor='white', fontsize = 27 )
plt_13.xlabel('Localization Maximum Absolute Error [m]', fontsize = 35)
plt_13.ylabel('CDF', fontsize = 35)
plt_13.show()


if (norm(mse_lapl_error) < norm(mse_gps_error)):
    print ('MSE reduction with CGCL: ', norm(mse_lapl_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with CGCL: ', norm(mse_lapl_error- mse_gps_error)/norm(mse_gps_error))


if (norm(mse_cgcl_ekf_error) < norm(mse_gps_error)):
    print ('MSE reduction with CCEKF: ', norm(mse_cgcl_ekf_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with CCEKF: ', norm(mse_cgcl_ekf_error- mse_gps_error)/norm(mse_gps_error))
    

print ('\n')

if (norm(max_lapl_error) < norm(max_gps_error)):
    print ('MAX reduction with CGCL: ', norm(max_lapl_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with CGCL: ', norm(max_lapl_error- max_gps_error)/norm(max_gps_error))
    

if (norm(max_cgcl_ekf_error) < norm(max_gps_error)):
    print ('MAX reduction with CCEKF: ', norm(max_cgcl_ekf_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with CCEKF: ', norm(max_cgcl_ekf_error- max_gps_error)/norm(max_gps_error)) 


vehicle_idx = random.randint(0, true_X.shape[0]-1)

stop = vehicle_idx + 1
time_2 = true_X.shape[1]



if (norm(cgcl_ekf_error[vehicle_idx, :]) < norm(gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with CCEKF:', norm(cgcl_ekf_error[vehicle_idx, :]- gps_error[vehicle_idx, :])/norm(gps_error[vehicle_idx, :]))
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with CCEKF:', norm(cgcl_ekf_error[vehicle_idx, :]- gps_error[vehicle_idx, :])/norm(gps_error[vehicle_idx, :]))
    
if (norm(lapl_error[vehicle_idx, :]) < norm(gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with CGCL alone:', norm(lapl_error[vehicle_idx, :] - gps_error[vehicle_idx, :])/norm(gps_error[vehicle_idx, :]))
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with CGCL alone:', norm(lapl_error[vehicle_idx, :] - gps_error[vehicle_idx, :])/norm(gps_error[vehicle_idx, :]))
    
