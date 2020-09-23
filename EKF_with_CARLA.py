import numpy as np
import pandas as pd
from IPython import get_ipython
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
from numpy.linalg import norm,matrix_rank, svd, inv
from scipy.sparse.linalg import lsmr
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import cvxpy as cvx

def GPS_noise(mu, sigma_x, sigma_y, samples):
    
    if ((samples % 2) != 0):
        samples = samples + 1
        
    white_x = np.random.normal(mu, sigma_x, size=samples) 
    white_y = np.random.normal(mu, sigma_y, size=samples)
    
    '''
    for i in range(white_x.size):
        
        white_x[i] = (white_x[i] - mu)/sigma_x
        white_y[i] = (white_y[i] - mu)/sigma_y
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

    '''
    for i in range(pink_x.size):
        pink_x[i] = (pink_x[i] - np.mean(pink_x))/np.std(pink_x)
        pink_y[i] = (pink_y[i] - np.mean(pink_y))/np.std(pink_y)
    '''   
    
    noise_x = white_x + pink_x
    noise_y = white_y + pink_y
    
    return white_x, white_y, pink_x, pink_y, noise_x, noise_y


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

    Ident_Matrix = np.identity((L_bar[0,:].size))
    
    for i in range (Ident_Matrix[:,0].size):
        for j in range (Ident_Matrix[0,:].size):
            Ident_Matrix[i,j] *= 0.001
    
    
    if (np.linalg.matrix_rank(L_bar.T@L_bar) < (L_bar.T@L_bar)[0,:].size):
        print ('Problem with rank of L_bar')
        
    
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

def Distributed_Lapl (neighbors_mat, clusters, mhi_local_KF, mhi_hat_local_KF, Sigma_local_KF, Sigma_hat_local_KF, idx_time, idx_test_cluster, G_local, Q_local, R_local, g, Deg, A, delta_X, delta_Y, test_noise_X, test_noise_Y, test_true_X, test_true_Y):
#def Distributed_Lapl (Deg, A, delta_X, delta_Y, test_noise_X, test_noise_Y, test_true_X, test_true_Y):
    
    num_of_iter_distr = 1
    
    x_final_local = np.zeros((test_noise_X.size, num_of_iter_distr))
    y_final_local = np.zeros((test_noise_X.size, num_of_iter_distr))
    
    distr_delta_X = np.zeros(test_noise_X.size)
    distr_delta_Y = np.zeros(test_noise_Y.size)
    
    distr_delta_X = np.copy(delta_X)
    distr_delta_Y = np.copy(delta_Y)
    
    x_ekf_vector = np.zeros(test_noise_X.size)
    y_ekf_vector = np.zeros(test_noise_X.size)
    
    for l in range (num_of_iter_distr):
        
        for k in range (test_noise_X.size):  
              
            temp_index = []
            
            for j in range (A[k,:].size):
                if (A[k,j] == 1):
        
                    temp_index.append(j)
                    
            temp_index.append(k)
               
            index = np.zeros(len(temp_index))
        
            index = (np.asarray(temp_index))
            
            L_local = np.zeros((index.size+1, index.size))
            
            for i in range (index.size):
                L_local[i,i] = 1
                if (i == index.size-1):
                    for j in range (index.size-1):
                        L_local[i,j] = -1
                        
                    L_local[i,j+1] = Deg[index[i],index[i]]
            
            L_local[i+1, index.size - 1] = 1
        
            b = np.zeros(index.size  +1)
            q = np.zeros(index.size  +1)
            
            for i in range (index.size-1):
                
                if (l == 0):
                    b[i] = test_noise_X[index[i]]
                    q[i] = test_noise_Y[index[i]]
                else:
                    b[i] = x_final_local[index[i],l-1]
                    q[i] = y_final_local[index[i],l-1]
                
                
            b[i+1] = distr_delta_X[index[i+1]]
            q[i+1] = distr_delta_Y[index[i+1]]  
            
            if (l == 0):
                b[i+2] = test_noise_X[k]
                q[i+2] = test_noise_Y[k]
            else:
                b[i+2] = x_final_local[k,l-1]
                q[i+2] = y_final_local[k,l-1]
            
            temp_x = np.zeros(index.size   )   
            temp_y = np.zeros(index.size   )
           
            temp_x = lsmr(L_local,b)[0]
            temp_y = lsmr(L_local,q)[0]
            
            x_final_local[k,l] = temp_x[index.size-1]
            y_final_local[k,l] = temp_y[index.size-1]
            
            
            state = int(index.size)
                
            if (idx_time == 0):
                                
                mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = np.concatenate([temp_x, temp_y])
                             
                Sigma_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = np.eye(2*state)
                    
                                
            else:
                      
                
                if (np.array_equal(neighbors_mat[clusters[idx_test_cluster, idx_time][k], idx_time], neighbors_mat[clusters[idx_test_cluster, idx_time][k], idx_time-1]) == False):
    
                    mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = np.concatenate([temp_x, temp_y])
                                
                    Sigma_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = np.eye(2*state)
                                    
                        
                else:
                                    
                    u = 0
                    for i in range(index.size):
                                        
                        mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][u] = mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time-1].T[u] + g[clusters[idx_test_cluster, idx_time][k], idx_time][u]
                        
                        
                        mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][u+state] = mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time-1].T[u+state] + g[clusters[idx_test_cluster, idx_time][k], idx_time][u+state]
                                        
                        u += 1
                    
                    
                    Sigma_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = G_local[clusters[idx_test_cluster, idx_time][k], idx_time] @ Sigma_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time-1] @ G_local[clusters[idx_test_cluster, idx_time][k], idx_time].T + R_local[clusters[idx_test_cluster, idx_time][k], idx_time]
                                    
                    Kalman_gain = np.zeros((2*(state+1), 2*(state+1) ))
                
                    H = np.zeros((2*(state+1), 2*state))
                                    
                    H[:state+1,:state] = np.copy(L_local)
                                    
                    H[state+1:,state:] = np.copy(L_local)
                        
                    Kalman_gain = Sigma_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] @ H.T @ (inv(H @ Sigma_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] @ H.T + Q_local[clusters[idx_test_cluster, idx_time][k], idx_time]))
                                    
                    z, h = np.zeros((2*(state+1),1)), np.zeros((2*(state+1),1))
                                
                    z = np.concatenate([np.reshape(b, (state+1,1)), np.reshape(q, (state+1,1))])
                    
                    b_h_X = np.zeros(((state+1),1))
                    
                    b_h_Y = np.zeros(((state+1),1))
                    
                    for i in range (b_h_X.shape[0] - 2):
                        
                        b_h_X[i] = mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][i]
                        
                        b_h_Y[i] = mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][i+state]
                        
                        b_h_X[-2] += mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][index.size-1] - mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][i]
                        
                        b_h_Y[-2] += mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][-1] - mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][i+state]
                    
                    
                    b_h_X[-1] = mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][index.size-1]
                        
                    b_h_Y[-1] = mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time][-1]
                    
                    h = np.concatenate([b_h_X, b_h_Y])
                    
                    measurement_mat = np.zeros((H.shape[0],1))
                    
                    measurement_mat = np.reshape(z - h, (H.shape[0],1))
                                                
                    aggreg_vec = Kalman_gain @ measurement_mat
                                                
                    aggreg_matrix = Kalman_gain@H
                   
                    mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = mhi_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] + aggreg_vec.T
                                        
                    Sigma_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] = Sigma_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time] - aggreg_matrix@Sigma_hat_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time]
            
            
           
            x_ekf_vector[k] = mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time].T[index.size-1] 
            
            y_ekf_vector[k] = mhi_local_KF[clusters[idx_test_cluster, idx_time][k], idx_time].T[-1] 
            
    local_lapl_error = np.zeros(test_true_X.shape[0])
    
    temp_ekf_local_error = np.zeros(test_true_X.shape[0])
    
    for i in range (local_lapl_error.shape[0]):
        
        local_lapl_error[i] = norm(np.array([test_true_X[i] - x_final_local[i,num_of_iter_distr-1] , test_true_Y[i] - y_final_local[i,num_of_iter_distr-1]]), 2)
        
        temp_ekf_local_error[i] = norm(np.array([test_true_X[i] - x_ekf_vector[i] , test_true_Y[i] - y_ekf_vector[i]]), 2)
        
    return x_final_local[:,num_of_iter_distr-1], y_final_local[:,num_of_iter_distr-1], x_ekf_vector, y_ekf_vector, local_lapl_error,temp_ekf_local_error

def Optimization_AoA_GD (A, noisy_D, tan_noisy_AoA, noise_X, noise_Y, true_X, true_Y, deg_noisy_AoA, init_X, init_Y):
    
    delta = 0.001
    
    low_a = 70
    high_a = 110
    
    low_a_2 = 250
    high_a_2 = 290
    
    num_of_iter_GD = 500
    
    x_final = np.zeros((noise_X.size, num_of_iter_GD))
    y_final = np.zeros((noise_X.size, num_of_iter_GD))
   
    
    for l in range (1, num_of_iter_GD):
        
        for i in range (noise_X.size):
            
            sum_x_d = 0
            sum_y_d = 0
            
            sum_x_a = 0
            sum_y_a = 0
            
            for j in range (noise_X.size):
                
                if (A[i,j] == 1):
                    
                    
                    if (l == 1):
                        
                        sum_x_d += 2*((noisy_D[i,j] - norm(np.array([init_X[i] - init_X[j], init_Y[i] - init_Y[j]]),2))*( (init_X[i] - init_X[j]) / (norm(np.array([init_X[i] - init_X[j], init_Y[i] - init_Y[j]]),2))))
                        
                        sum_y_d += 2*((noisy_D[i,j] - norm(np.array([init_X[i] - init_X[j], init_Y[i] - init_Y[j]]),2))*( (init_Y[i] - init_Y[j]) / (norm(np.array([init_X[i] - init_X[j], init_Y[i] - init_Y[j]]),2))))
                       
                        
                        if (deg_noisy_AoA[i,j] > 0):
                            
                            if ((deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] <= low_a_2 or deg_noisy_AoA[i,j] >= high_a_2)):
                                
                                    sum_x_a += 2*( (init_X[j] - init_X[i])*(tan_noisy_AoA[i,j]**2) - (init_Y[j] - init_Y[i])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (init_Y[j] - init_Y[i]) - (init_X[j] - init_X[i])*tan_noisy_AoA[i,j])
                        else:
                            
                            if ((deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] >= -low_a_2 or deg_noisy_AoA[i,j] <= -high_a_2)):
                                
                                    sum_x_a += 2*( (init_X[j] - init_X[i])*(tan_noisy_AoA[i,j]**2) - (init_Y[j] - init_Y[i])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (init_Y[j] - init_Y[i]) - (init_X[j] - init_X[i])*tan_noisy_AoA[i,j])
                      
                        
                    else:
                        
                        sum_x_d += 2*((noisy_D[i,j] - norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))*( (x_final[i,l-1] - x_final[j,l-1]) / (norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))))
                        
                        sum_y_d += 2*((noisy_D[i,j] - norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))*( (y_final[i,l-1] - y_final[j,l-1]) / (norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))))
                        
                        
                        if (deg_noisy_AoA[i,j] > 0):
                            
                            if ((deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] <= low_a_2 or deg_noisy_AoA[i,j] >= high_a_2)):
                                
                                    sum_x_a += 2*( (x_final[j,l-1] - x_final[i,l-1])*(tan_noisy_AoA[i,j]**2) - (y_final[j,l-1] - y_final[i,l-1])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (y_final[j,l-1] - y_final[i,l-1]) - (x_final[j,l-1] - x_final[i,l-1])*tan_noisy_AoA[i,j])
                        
                        else:
                            
                            if ((deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] >= -low_a_2 or deg_noisy_AoA[i,j] <= -high_a_2)):
                                
                                    sum_x_a += 2*( (x_final[j,l-1] - x_final[i,l-1])*(tan_noisy_AoA[i,j]**2) - (y_final[j,l-1] - y_final[i,l-1])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (y_final[j,l-1] - y_final[i,l-1]) - (x_final[j,l-1] - x_final[i,l-1])*tan_noisy_AoA[i,j])
                       
                        
            if (l==1):
                
                x_final[i,l] = init_X[i] + delta*(sum_x_d + sum_x_a)
                
                y_final[i,l] = init_Y[i] + delta*(sum_y_d + sum_y_a)
                
                
            else:
                
                x_final[i,l] =  x_final[i,l-1] + delta*(sum_x_d + sum_x_a + 2*(noise_X[i] - x_final[i,l-1])) 
                
                y_final[i,l] =  y_final[i,l-1] + delta*(sum_y_d + sum_y_a + 2*(noise_Y[i] - y_final[i,l-1])) 
                
    
    
    tcl_error = np.zeros(noise_X.shape[0])

    for i in range (tcl_error.shape[0]):
        
        tcl_error[i] = norm(np.array([true_X[i] - x_final[i,num_of_iter_GD-1] , true_Y[i] - y_final[i,num_of_iter_GD-1]]), 2)
        
    return x_final[:,num_of_iter_GD-1], y_final[:,num_of_iter_GD-1], tcl_error


def Create_Batch(clusters, start_time, end_time, Points_X, Points_Y, flag_rank, num_of_vehicles, flag_vis):
    
    if (flag_rank == 0):
        
        batch_X = np.zeros((len(clusters), end_time - start_time + 1))
        batch_Y = np.zeros((len(clusters), end_time - start_time + 1))
        
    else:
        
        batch_X = np.zeros((2*len(clusters), end_time - start_time + 1))
        batch_Y = np.zeros((2*len(clusters), end_time - start_time + 1))
    
    u = 0
    
    for i in range (start_time, end_time+1):
        
        l = 0
        
        for j in clusters:
            
            batch_X[l,u] = Points_X[j,i]
            batch_Y[l,u] = Points_Y[j,i]
            
            if (flag_rank == 1):
                
                batch_X[l + len(clusters),u] = Points_X[j + num_of_vehicles,i]
                batch_Y[l + len(clusters),u] = Points_Y[j + num_of_vehicles,i]
            
            l += 1
            
        u += 1
    
    if (flag_rank == 0):
          
        if (flag_vis == 1):
            
            plt_2.figure(2)
            
            for i in range(batch_X.shape[0]):
                color_index_1 = random.uniform(0, 1)
                color_index_2 = random.uniform(0, 1)
                color_index_3 = random.uniform(0, 1)
                for j in range (batch_X.shape[1] - 1):
                    plt_2.plot(np.array([batch_X[i,j],batch_X[i,j+1]]),np.array([batch_Y[i,j],batch_Y[i,j+1]]), '-', c = (color_index_1, color_index_2, color_index_3), marker = 'o', linewidth = 3)
                    if (j == 0):
                        plt_2.annotate(str(clusters[i]), (batch_X[i,j], batch_Y[i,j]), fontsize=12)
                    
            plt_2.xlabel('x-axis', fontsize = 35)
            plt_2.ylabel('y-axis', fontsize = 35)
            #plt_2.title('True trajectories up to time instant t = ' + str(batch_X.shape[1]), fontsize=20)    
            plt_2.xticks(fontsize=28)
            plt_2.yticks(fontsize=28)
            plt_2.tick_params(direction='out', length=8)
            plt_2.grid(b=True)
            plt_2.show()    
    
    return batch_X, batch_Y

def LRMR(clusters, B_x, B_y, u, s, vh, rank):
    
    target_x = np.copy(B_x)
    
    target_y = np.copy(B_y)
   
    W_X = u.T@target_x
        
    D_r_X = np.eye(s.size)
        
    D_r_X = D_r_X*s
        
    u_W_X, s_W_X, vh_W_X = svd(W_X,  full_matrices=False)
        
    s_W_X[rank:] = 0
        
    Theta_matrix_X = vh.T@(inv(D_r_X)@u_W_X@(np.eye(s_W_X.size)*s_W_X)@vh_W_X)
       
    W_Y = u.T@target_y
        
    D_r_Y = np.eye(s.size)
        
    D_r_Y = D_r_Y*s
        
    u_W_Y, s_W_Y, vh_W_Y = svd(W_Y,  full_matrices=False)
       
    s_W_Y[rank:] = 0
        
    Theta_matrix_Y = vh.T@(inv(D_r_Y)@u_W_Y@(np.eye(s_W_Y.size)*s_W_Y)@vh_W_Y)
    
    return Theta_matrix_X, Theta_matrix_Y

def shrink_operator(Matrix, v):
    
    Zeros_Matrix = np.zeros((Matrix.shape[0], Matrix.shape[1]))
    
    return np.sign(Matrix)*np.maximum(abs(Matrix) - v, Zeros_Matrix)

def RPCA(Ex, Ey, num_of_iter):
    
    lamda = 1/(math.sqrt(max(Ex[:,0].size, num_of_iter)))

    #lamda = 0.1
    
    '''
    mhi = 0.0001
    
    RPCA_iter = 300
    '''
    
    mhi = 0.1
    
    RPCA_iter = 600
    
    
    Yx = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yx = Ex / max(norm(Ex), norm((lamda**(-1))*Ex, np.inf))
    
    Nx = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yy = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yy = Ey / max(norm(Ey), norm((lamda**(-1))*Ey, np.inf))
    
    Ny = np.zeros((Ex[:,0].size, num_of_iter))
    
    
    for i in range (RPCA_iter):
        
        Ux, Sigmax, Vhx = svd((Ex - Nx + (mhi**(-1))*Yx), full_matrices=False)
        
        Sx = np.copy(Ux@shrink_operator(np.eye(Sigmax.size)*Sigmax, (mhi**(-1)))@Vhx)
        
        Nx = np.copy(shrink_operator(Ex - Sx + (mhi**(-1))*Yx ,lamda*(mhi**(-1))))
        
        Yx = np.copy(Yx + mhi*(Ex - Sx - Nx))
        
        Uy, Sigmay, Vhy = svd((Ey - Ny + (mhi**(-1))*Yy), full_matrices=False)
        
        Sy = np.copy(Uy@shrink_operator(np.eye(Sigmay.size)*Sigmay, (mhi**(-1)))@Vhy)
        
        Ny = np.copy(shrink_operator(Ey - Sy + (mhi**(-1))*Yy ,lamda*(mhi**(-1))))
        
        Yy = np.copy(Yy + mhi*(Ey - Sy - Ny))
        
        mhi = mhi*2.2
        
   
    return Sx, Sy, Nx, Ny


def Errors(temp_x, temp_y, noise_temp_x, noise_temp_y, cgcl_temp_x, cgcl_temp_y, LR_X, LR_Y):
    
    cgcl_error = np.zeros((temp_x.shape[0], temp_x.shape[1]))
    
    rpca_error = np.zeros((temp_x.shape[0], temp_x.shape[1]))
    
    temp_error = np.zeros((temp_x.shape[0], temp_x.shape[1]))
    
    mse_cgcl_error = np.zeros(temp_x.shape[1])
    
    mse_rpca_error = np.zeros(temp_x.shape[1])
    
    mse_temp_error = np.zeros(temp_x.shape[1])
     
    for i in range (temp_x.shape[0]):
        
        for j in range (temp_x.shape[1]):
            
            cgcl_error[i,j] = norm(np.array([cgcl_temp_x[i,j] - temp_x[i,j], cgcl_temp_y[i,j] - temp_y[i,j]]))
            
            rpca_error[i,j] = norm(np.array([LR_X[i,j] - temp_x[i,j], LR_Y[i,j] - temp_y[i,j]]))
            
            temp_error[i,j] = norm(np.array([noise_temp_x[i,j] - temp_x[i,j], noise_temp_y[i,j] - temp_y[i,j]]))
       
    
    for i in range (temp_x.shape[1]):
        
        mse_cgcl_error[i] = np.mean(cgcl_error[:,i]**2)
        
        mse_rpca_error[i] = np.mean(rpca_error[:,i]**2)
        
        mse_temp_error[i] = np.mean(temp_error[:,i]**2)
    
    return cgcl_error, rpca_error, mse_cgcl_error, mse_rpca_error, mse_temp_error
    

def Visualize (clusters, start, end, true_X, true_Y, noise_X, noise_Y, Concatenated_CGCL_Points_X, Concatenated_CGCL_Points_Y):


    temp_x, temp_y = Create_Batch(clusters, start, end, true_X, true_Y, 0, true_X.shape[0], 1)
    
    noise_temp_x, noise_temp_y = Create_Batch(clusters, start, end, noise_X, noise_Y, 0, true_X.shape[0], 0)
    
    cgcl_temp_x, cgcl_temp_y = Create_Batch(clusters, start, end, Concatenated_CGCL_Points_X, Concatenated_CGCL_Points_Y, 0, true_X.shape[0], 0)
    
    LR_X, LR_Y, S_X, S_Y = RPCA(cgcl_temp_x, cgcl_temp_y, cgcl_temp_x.shape[1])
    
    cgcl_error, rpca_error, mse_cgcl_error, mse_rpca_error, mse_temp_error = Errors(temp_x, temp_y, noise_temp_x, noise_temp_y, cgcl_temp_x, cgcl_temp_y, LR_X, LR_Y)

    plt_13.figure(13)
    
    plt_13.plot(np.sort(mse_cgcl_error), np.arange(len(np.sort(mse_cgcl_error)))/float(len(mse_cgcl_error)), 'r*-',  label="CGCL", linewidth = 4, markersize = 6)
    plt_13.plot(np.sort(mse_rpca_error), np.arange(len(np.sort(mse_rpca_error)))/float(len(mse_rpca_error)), 'b*-',  label="RPCA", linewidth = 4, markersize = 6)    
    
    plt_13.legend(facecolor='white', fontsize = 37 )
    
    
    if (norm(mse_cgcl_error) < norm(mse_temp_error)):
        print ('MSE reduction with CGCL: ', norm(mse_cgcl_error- mse_temp_error)/norm(mse_temp_error))
    
    else:
        print ('MSE increment with CGCL: ', norm(mse_cgcl_error- mse_temp_error)/norm(mse_temp_error))
        
    if (norm(mse_rpca_error) < norm(mse_temp_error)):
        print ('MSE reduction with RPCA: ', norm(mse_rpca_error- mse_temp_error)/norm(mse_temp_error))
    
    else:
        print ('MSE increment with RPCA: ', norm(mse_rpca_error- mse_temp_error)/norm(mse_temp_error))
        
    return temp_x, temp_y, cgcl_temp_x, cgcl_temp_y, LR_X, LR_Y, S_X, S_Y, cgcl_error, rpca_error, mse_cgcl_error, mse_rpca_error

def Animation (true_X, true_Y, noise_X, noise_Y, rgcl_recon_X, rgcl_recon_Y, rtcl_recon_X, rtcl_recon_Y, vehicle_idx):
    
    fig, ax = plt_10.figure(10), plt_10.figure(10)

    plt_10.xticks(fontsize=25)
    plt_10.yticks(fontsize=25)
    plt_10.tick_params(direction='out', length=8)
    plt_10.xlabel('x-axis', fontsize = 32)
    plt_10.ylabel('y-axis', fontsize = 32)
    t = true_X[vehicle_idx,:]
    s = true_Y[vehicle_idx,:]
    l = plt_10.plot(t, s, 'b-', linewidth = 6.0)
    
    min_1_x = np.minimum(true_X[vehicle_idx,:], rgcl_recon_X[vehicle_idx,:])
    min_2_x = np.minimum(rtcl_recon_X[vehicle_idx,:], noise_X[vehicle_idx,:])
    min_3_x = np.minimum(min_1_x, min_2_x)
    
    min_1_y = np.minimum(true_Y[vehicle_idx,:], rgcl_recon_Y[vehicle_idx,:])
    min_2_y = np.minimum(rtcl_recon_Y[vehicle_idx,:], noise_Y[vehicle_idx,:])
    min_3_y = np.minimum(min_1_y, min_2_y)
    
    max_1_x = np.maximum(true_X[vehicle_idx,:], rgcl_recon_X[vehicle_idx,:])
    max_2_x = np.maximum(rtcl_recon_X[vehicle_idx,:], noise_X[vehicle_idx,:])
    max_3_x = np.maximum(max_1_x, max_2_x)
    
    max_1_y = np.maximum(true_Y[vehicle_idx,:], rgcl_recon_Y[vehicle_idx,:])
    max_2_y = np.maximum(rtcl_recon_Y[vehicle_idx,:], noise_Y[vehicle_idx,:])
    max_3_y = np.maximum(max_1_y, max_2_y)
    #ax = plt_10.axis([np.min(np.minimum(true_X[vehicle_idx,:], rgcl_recon_X[vehicle_idx,:], rtcl_recon_X[vehicle_idx,:], noise_X[vehicle_idx,:]))-1, np.max(np.maximum(true_X[vehicle_idx,:], rgcl_recon_X[vehicle_idx,:], rtcl_recon_X[vehicle_idx,:], noise_X[vehicle_idx,:]))+1, np.min(np.minimum(true_Y[vehicle_idx,:], rgcl_recon_Y[vehicle_idx,:], rtcl_recon_Y[vehicle_idx,:], noise_Y[vehicle_idx,:]))-1, np.max(np.maximum(true_Y[vehicle_idx,:], rgcl_recon_Y[vehicle_idx,:], rtcl_recon_Y[vehicle_idx,:], noise_Y[vehicle_idx,:]))+1])
    
    ax = plt_10.axis([np.min(min_3_x)-1, np.max(max_3_x)+1, np.min(min_3_y)-1, np.max(max_3_y)+1])
    #ax = plt_10.axis([np.min(true_X[vehicle_idx,:])-1, np.max(true_X[vehicle_idx,:])+1, np.min(true_Y[vehicle_idx,:])-1, np.max(true_Y[vehicle_idx,:])+1])
    
    redDot, = plt_10.plot([true_X[vehicle_idx,0]], [true_Y[vehicle_idx,0]], 'ro', markersize = 10.0)
    greDot, = plt_10.plot([rgcl_recon_X[vehicle_idx,0]], [rgcl_recon_Y[vehicle_idx,0]], 'go', markersize = 10.0)
    yelDot, = plt_10.plot([rtcl_recon_X[vehicle_idx,0]], [rtcl_recon_Y[vehicle_idx,0]], 'yo', markersize = 10.0)
    magDot, = plt_10.plot([noise_X[vehicle_idx,0]], [noise_Y[vehicle_idx,0]], 'mo', markersize = 10.0)
    
    
    def animate(i):
        x = true_X[vehicle_idx,i]
        y = true_Y[vehicle_idx,i]
        redDot.set_data(x, y)
        
        x_2 = rgcl_recon_X[vehicle_idx,i]
        y_2 = rgcl_recon_Y[vehicle_idx,i]
        greDot.set_data(x_2, y_2)
        
        x_3 = noise_X[vehicle_idx,i]
        y_3 = noise_Y[vehicle_idx,i]
        magDot.set_data(x_3, y_3)
        
        x_4 = rtcl_recon_X[vehicle_idx,i]
        y_4 = rtcl_recon_Y[vehicle_idx,i]
        yelDot.set_data(x_4, y_4)
        
        return redDot, greDot, magDot, yelDot,
        #return redDot,
   
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(time_instances), interval=6500, blit=True, repeat=True)
    
    plt_10.title('Vehicle ' + str(vehicle_idx), fontsize = 27)
    myAnimation.save('demo/True.gif', writer=PillowWriter(fps=5))
    #plt_10.show()
    
    fig, ax = plt_14.figure(14), plt_14.figure(14)
    
    plt_14.xticks(fontsize=25)
    plt_14.yticks(fontsize=25)
    plt_14.tick_params(direction='out', length=8)
    plt_14.xlabel('x-axis', fontsize = 32)
    plt_14.ylabel('y-axis', fontsize = 32)
    t = rgcl_recon_X[vehicle_idx,:]
    s = rgcl_recon_Y[vehicle_idx,:]
    l = plt_14.plot(t, s, 'b-', linewidth = 6.0)
    
    ax = plt_14.axis([np.min(rgcl_recon_X[vehicle_idx,:])-1, np.max(rgcl_recon_X[vehicle_idx,:])+1, np.min(rgcl_recon_Y[vehicle_idx,:])-1, np.max(rgcl_recon_Y[vehicle_idx,:])+1])
    
    greDot_temp, = plt_14.plot([rgcl_recon_X[vehicle_idx,0]], [rgcl_recon_Y[vehicle_idx,0]], 'go', markersize = 10.0)
    
    
    def animate(i):
        x_temp = rgcl_recon_X[vehicle_idx,i]
        y_temp = rgcl_recon_Y[vehicle_idx,i]
        greDot_temp.set_data(x_temp, y_temp)
        return redDot,
   
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(time_instances), interval=6500, blit=True, repeat=True)
    
    plt_14.title('Vehicle ' + str(vehicle_idx), fontsize = 27)
    
    myAnimation.save('demo/RGCL.gif', writer=PillowWriter(fps=5))
    #plt_14.show()
    
    fig, ax = plt_16.figure(16), plt_16.figure(16)
    
    plt_16.xticks(fontsize=25)
    plt_16.yticks(fontsize=25)
    plt_16.tick_params(direction='out', length=8)
    plt_16.xlabel('x-axis', fontsize = 32)
    plt_16.ylabel('y-axis', fontsize = 32)
    t = noise_X[vehicle_idx,:]
    s = noise_Y[vehicle_idx,:]
    l = plt_16.plot(t, s, 'b-', linewidth = 6.0)
    
    ax = plt_16.axis([np.min(noise_X[vehicle_idx,:])-1, np.max(noise_X[vehicle_idx,:])+1, np.min(noise_Y[vehicle_idx,:])-1, np.max(noise_Y[vehicle_idx,:])+1])
    
    magDot_temp, = plt_16.plot([noise_X[vehicle_idx,0]], [noise_Y[vehicle_idx,0]], 'mo', markersize = 10.0)
    
    def animate(i):
        x_temp_2 = noise_X[vehicle_idx,i]
        y_temp_2 = noise_Y[vehicle_idx,i]
        magDot_temp.set_data(x_temp_2, y_temp_2)
        return magDot_temp,
 
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(time_instances), interval=6500, blit=True, repeat=True)
    
    plt_16.title('Vehicle ' + str(vehicle_idx), fontsize = 27)
    myAnimation.save('demo/GPS.gif', writer=PillowWriter(fps=5))
    #plt_16.show()

    fig, ax = plt_16.figure(15), plt_16.figure(15)
    
    plt_15.xticks(fontsize=25)
    plt_15.yticks(fontsize=25)
    plt_15.tick_params(direction='out', length=8)
    plt_15.xlabel('x-axis', fontsize = 32)
    plt_15.ylabel('y-axis', fontsize = 32)
    t = rtcl_recon_X[vehicle_idx,:]
    s = rtcl_recon_Y[vehicle_idx,:]
    l = plt_15.plot(t, s, 'b-', linewidth = 6.0)
    
    ax = plt_15.axis([np.min(rtcl_recon_X[vehicle_idx,:])-1, np.max(rtcl_recon_X[vehicle_idx,:])+1, np.min(rtcl_recon_Y[vehicle_idx,:])-1, np.max(rtcl_recon_Y[vehicle_idx,:])+1])
    
    yelDot_temp, = plt_15.plot([rtcl_recon_X[vehicle_idx,0]], [rtcl_recon_Y[vehicle_idx,0]], 'yo', markersize = 10.0)
    
    def animate(i):
        x_temp_4 = rtcl_recon_X[vehicle_idx,i]
        y_temp_4 = rtcl_recon_Y[vehicle_idx,i]
        yelDot_temp.set_data(x_temp_4, y_temp_4)
        return yelDot_temp,
 
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(time_instances), interval=6500, blit=True, repeat=True)
    
    plt_15.title('Vehicle ' + str(vehicle_idx), fontsize = 27)
    myAnimation.save('demo/RTCL.gif', writer=PillowWriter(fps=5))
    #plt_15.show()
    
    return 0;

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
                        
            Sigma_alone_hat_KF[idx_test_cluster, idx_time] = G_alone@Sigma_alone_KF[idx_test_cluster, idx_time-1]@G_alone.T + (sigma_trans**2)*np.eye(state)
                                
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


def EKF_with_Lapl(mhi_KF, mhi_hat_KF, Sigma_KF, Sigma_hat_KF, clusters, idx_time, idx_test_cluster, L_bar, test_lapl_Points, delta_X, delta_Y, G, Q, R, test_A, anchors):    

    
    state = int(clusters_size[idx_test_cluster, idx_time])
                
    if (idx_time == 0):
       
        mhi_KF[idx_test_cluster, idx_time] = np.concatenate([test_lapl_Points[:,0], test_lapl_Points[:,1]])
                     
        Sigma_KF[idx_test_cluster, idx_time] = np.eye(2*state)
            
                        
    else:
                        
        if (clusters[idx_test_cluster, idx_time] != clusters[idx_test_cluster, idx_time-1]):
            
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
                                
    return mhi_KF

def EKF_with_Lapl_Est(noise_X, noise_Y, Concatenated_CGCL_Points_X, Concatenated_CGCL_Points_Y, mhi_alone_lapl_KF, mhi_alone_hat_lapl_KF, Sigma_alone_lapl_KF, Sigma_alone_hat_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_Y, sigma_x, sigma_y, sigma_trans):    
    
    
    if (idx_time == 0):
                
        mhi_alone_lapl_KF[idx_test_cluster, idx_time][0] = Concatenated_CGCL_Points_X[idx_test_cluster,idx_time]
        mhi_alone_lapl_KF[idx_test_cluster, idx_time][1] = Concatenated_CGCL_Points_Y[idx_test_cluster,idx_time]
                   
        Sigma_alone_lapl_KF[idx_test_cluster, idx_time] = np.eye(2)
        
    else:
        
        if (Deg[idx_time-1, idx_test_cluster, idx_test_cluster] != -3):
            
            state = 2
                    
            size_of_measurement_model = 2
            
            G_alone_lapl = np.eye(state)
                                    
            mhi_alone_hat_lapl_KF[idx_test_cluster, idx_time][0] = mhi_alone_lapl_KF[idx_test_cluster, idx_time-1].T[0] + KM_comp_x[idx_test_cluster, idx_time]
                                                    
            mhi_alone_hat_lapl_KF[idx_test_cluster, idx_time][1] = mhi_alone_lapl_KF[idx_test_cluster, idx_time-1].T[1] + KM_comp_y[idx_test_cluster, idx_time]
                                        
            Sigma_alone_hat_lapl_KF[idx_test_cluster, idx_time] = G_alone_lapl@Sigma_alone_lapl_KF[idx_test_cluster, idx_time-1]@G_alone_lapl.T + (sigma_trans**2)*np.eye(state)
                                                
            Kalman_gain_alone_lapl = np.zeros((state, size_of_measurement_model))
                            
            H_alone_lapl = np.eye(state)
                                     
            Q_alone_lapl = np.eye(state)
                                    
            Q_alone_lapl[0,0] *= sigma_x**2
                                    
            Q_alone_lapl[1,1] *= sigma_y**2
                                    
            Kalman_gain_alone_lapl = Sigma_alone_hat_lapl_KF[idx_test_cluster, idx_time] @ H_alone_lapl.T @ (inv(H_alone_lapl @ Sigma_alone_hat_lapl_KF[idx_test_cluster, idx_time] @ H_alone_lapl.T + Q_alone_lapl))
                                                
            z_alone_lapl, h_alone_lapl = np.zeros((H_alone_lapl.shape[0],1)), np.zeros((H_alone_lapl.shape[0],1))
                             
            z_alone_lapl = np.array([Concatenated_CGCL_Points_X[idx_test_cluster, idx_time], Concatenated_CGCL_Points_Y[idx_test_cluster, idx_time]])
                                
            h_alone_lapl = np.array([mhi_alone_hat_lapl_KF[idx_test_cluster, idx_time][0], mhi_alone_hat_lapl_KF[idx_test_cluster, idx_time][1]])
                           
            measurement_mat_alone_lapl = np.zeros((H_alone_lapl.shape[0],1))
                                                            
            measurement_mat_alone_lapl = np.reshape(z_alone_lapl - h_alone_lapl, (H_alone_lapl.shape[0],1))
                                                            
            aggreg_vec_alone_lapl = Kalman_gain_alone_lapl @ measurement_mat_alone_lapl
                                                            
            aggreg_matrix_alone_lapl = Kalman_gain_alone_lapl@H_alone_lapl
                                                
            mhi_alone_lapl_KF[idx_test_cluster, idx_time] = mhi_alone_hat_lapl_KF[idx_test_cluster, idx_time] + aggreg_vec_alone_lapl.T
                                                    
            Sigma_alone_lapl_KF[idx_test_cluster, idx_time] = Sigma_alone_hat_lapl_KF[idx_test_cluster, idx_time] - aggreg_matrix_alone_lapl@Sigma_alone_hat_lapl_KF[idx_test_cluster, idx_time]
        
        
    return mhi_alone_lapl_KF[idx_test_cluster, idx_time]

def EKF_with_Local_Lapl_Est(noise_X, noise_Y, Concatenated_LGCL_Points_X, Concatenated_LGCL_Points_Y, mhi_alone_local_lapl_KF, mhi_alone_hat_local_lapl_KF, Sigma_alone_local_lapl_KF, Sigma_alone_hat_local_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_Y, sigma_x, sigma_y, sigma_trans):    

    if (idx_time == 0):
                
        mhi_alone_local_lapl_KF[idx_test_cluster, idx_time][0] = Concatenated_LGCL_Points_X[idx_test_cluster,idx_time]
        mhi_alone_local_lapl_KF[idx_test_cluster, idx_time][1] = Concatenated_LGCL_Points_Y[idx_test_cluster,idx_time]
                   
        Sigma_alone_local_lapl_KF[idx_test_cluster, idx_time] = np.eye(2)
        
    else:
        
        if (Deg[idx_time-1, idx_test_cluster, idx_test_cluster] != -3):
            
            state = 2
                    
            size_of_measurement_model = 2
            
            G_alone_local_lapl = np.eye(state)
                                    
            mhi_alone_hat_local_lapl_KF[idx_test_cluster, idx_time][0] = mhi_alone_local_lapl_KF[idx_test_cluster, idx_time-1].T[0] + KM_comp_x[idx_test_cluster, idx_time]
                                                    
            mhi_alone_hat_local_lapl_KF[idx_test_cluster, idx_time][1] = mhi_alone_local_lapl_KF[idx_test_cluster, idx_time-1].T[1] + KM_comp_y[idx_test_cluster, idx_time]
                                        
            Sigma_alone_hat_local_lapl_KF[idx_test_cluster, idx_time] = G_alone_local_lapl@Sigma_alone_local_lapl_KF[idx_test_cluster, idx_time-1]@G_alone_local_lapl.T + (sigma_trans**2)*np.eye(state)
                                                
            Kalman_gain_alone_local_lapl = np.zeros((state, size_of_measurement_model))
                            
            H_alone_local_lapl = np.eye(state)
                                     
            Q_alone_local_lapl = np.eye(state)
                                    
            Q_alone_local_lapl[0,0] *= sigma_x**2
                                    
            Q_alone_local_lapl[1,1] *= sigma_y**2
                                    
            Kalman_gain_alone_local_lapl = Sigma_alone_hat_local_lapl_KF[idx_test_cluster, idx_time] @ H_alone_local_lapl.T @ (inv(H_alone_local_lapl @ Sigma_alone_hat_local_lapl_KF[idx_test_cluster, idx_time] @ H_alone_local_lapl.T + Q_alone_local_lapl))
                                                
            z_alone_local_lapl, h_alone_local_lapl = np.zeros((H_alone_local_lapl.shape[0],1)), np.zeros((H_alone_local_lapl.shape[0],1))
                             
            z_alone_local_lapl = np.array([Concatenated_LGCL_Points_X[idx_test_cluster, idx_time], Concatenated_LGCL_Points_Y[idx_test_cluster, idx_time]])
                                
            h_alone_local_lapl = np.array([mhi_alone_hat_local_lapl_KF[idx_test_cluster, idx_time][0], mhi_alone_hat_local_lapl_KF[idx_test_cluster, idx_time][1]])
                           
            measurement_mat_alone_local_lapl = np.zeros((H_alone_local_lapl.shape[0],1))
                                                            
            measurement_mat_alone_local_lapl = np.reshape(z_alone_local_lapl - h_alone_local_lapl, (H_alone_local_lapl.shape[0],1))
                                                            
            aggreg_vec_alone_local_lapl = Kalman_gain_alone_local_lapl @ measurement_mat_alone_local_lapl
                                                            
            aggreg_matrix_alone_local_lapl = Kalman_gain_alone_local_lapl@H_alone_local_lapl
                                                
            mhi_alone_local_lapl_KF[idx_test_cluster, idx_time] = mhi_alone_hat_local_lapl_KF[idx_test_cluster, idx_time] + aggreg_vec_alone_local_lapl.T
                                                    
            Sigma_alone_local_lapl_KF[idx_test_cluster, idx_time] = Sigma_alone_hat_local_lapl_KF[idx_test_cluster, idx_time] - aggreg_matrix_alone_local_lapl@Sigma_alone_hat_local_lapl_KF[idx_test_cluster, idx_time]
        
        
    return mhi_alone_local_lapl_KF[idx_test_cluster, idx_time]

def Robust_Laplacian(true_Points, L, anchors, anchors_index, delta_X, delta_Y):

    
    L_bar = np.zeros((L.shape[0], anchors.shape[0]))
    
    
    anchors_matrix = np.zeros((anchors.shape[0], delta_X.size))

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
        
    
    
    x = cvx.Variable(shape = L.shape[0])
    y = cvx.Variable(shape = L.shape[0])
    out_x = cvx.Variable(shape = b.size)    
    out_y = cvx.Variable(shape = b.size)
    
    
    opt_prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_bar@x - (b - out_x)) + cvx.sum_squares(L_bar@y - (q - out_y)) + cvx.norm(out_x[L.shape[0]:],1) + cvx.norm(out_y[L.shape[0]:],1) )
    
                           ,[out_x[:L.shape[0]] == np.zeros(L.shape[0]),
                            out_y[:L.shape[0]] == np.zeros(L.shape[0]),
                                   ]
                                   )
    
    
    #opt_prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_bar@x - (b )) + cvx.sum_squares(L_bar@y - (q ))))        
    opt_prob.solve()
    
    reconstr_error = np.zeros((x.size))

    for i in range(reconstr_error.size):
        
        reconstr_error[i] = norm(np.array([true_Points[i,0] - x.value[i], true_Points[i,1] - y.value[i]]),2)
        
    return reconstr_error, L_bar, x.value, y.value, out_x.value, out_y.value   

def Robust_TCLMLE(A,z_d,z_c_x,z_c_y,z_a, deg_noisy_AoA, true_Points):
    
    x = cvx.Variable(shape = A[:,0].size)
    y = cvx.Variable(shape = A[:,0].size)
    out_x = cvx.Variable(shape = 2*A[:,0].size)    
    out_y = cvx.Variable(shape = 2*A[:,0].size)
    
    f_1 = 0
    f_2 = 0
    f_3 = 0
   
    low_a = 70
    high_a = 110
    
    low_a_2 = 250
    high_a_2 = 290
    
    for i in range(z_c_x.size):
        f_1 += (((z_c_x[i] - out_x[A[:,0].size + i]) - x[i])**2) + (((z_c_y[i] - out_y[A[:,0].size + i]) - y[i])**2)
   
             
    for i in range(z_d[:,0].size):
        for j in range(z_d[0,:].size):
            if (A[i,j] == 1):
                            
                f_2 += (cvx.power(cvx.pos(-z_d[i,j] + (cvx.norm(cvx.vstack( [x[i] - x[j], y[i] - y[j]] ),2))),2))
                
                if (deg_noisy_AoA[i,j] > 0):
                            
                    if ((deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a)): 
                                
                        if ((deg_noisy_AoA[i,j] <= low_a_2 or deg_noisy_AoA[i,j] >= high_a_2)):
                     
                            f_3 += ((z_a[i,j]*(x[j] - x[i]) - (y[j] - y[i]))**2)
                            
                else:
                            
                    if ((deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a)): 
                                
                        if ((deg_noisy_AoA[i,j] >= -low_a_2 or deg_noisy_AoA[i,j] <= -high_a_2)):
                            
                            f_3 += ((z_a[i,j]*(x[j] - x[i]) - (y[j] - y[i]))**2)
        
                
    opt_prob = cvx.Problem(cvx.Minimize(f_1 + f_2 + f_3 + 1.1*cvx.norm(out_x[A[:,0].size:],1) + 1.1*cvx.norm(out_y[A[:,0].size:],1)),
                           
                           [out_x[:A[:,0].size] == np.zeros(A[:,0].size),
                            out_y[:A[:,0].size] == np.zeros(A[:,0].size)]
                           )
    
    opt_prob.solve()
    
    reconstr_error = np.zeros((x.size))

    for i in range(reconstr_error.size):
        
        reconstr_error[i] = norm(np.array([true_Points[i,0] - x.value[i], true_Points[i,1] - y.value[i]]),2)
        
    return x.value, y.value, reconstr_error
 
mu = 0

sigma_x = 3
sigma_y = 2.5
sigma_trans = 0.8
sigma_d = 1
sigma_a = 4
Dt = 0.3


range_of_tranceivers = 20

number_of_connected_neighbours = 6

#573
time_instances = 100

#temp_locations = pd.read_excel('locations.xls')

#temp_locations = pd.read_excel('locations_2.xlsx')

#temp_locations = pd.read_excel('locations_3.xlsx')

temp_locations = pd.read_excel('locations_4.xlsx')

locations = np.array(temp_locations)

locations = np.delete(locations, 0, 1)

#locations = np.delete(locations, np.arange(84), 0)

#num_of_vehicles = 30

true_X = np.zeros((200, time_instances))
true_Y = np.zeros((200, time_instances))
true_Z = np.zeros((200, time_instances))

vel_X = np.zeros((true_X.shape[0], time_instances))
vel_Y = np.zeros((true_X.shape[0], time_instances))
vel_Z = np.zeros((true_X.shape[0], time_instances))

ang_vel_X = np.zeros((true_X.shape[0], time_instances))
ang_vel_Y = np.zeros((true_X.shape[0], time_instances))
ang_vel_Z = np.zeros((true_X.shape[0], time_instances))

acc_X = np.zeros((true_X.shape[0], time_instances))
acc_Y = np.zeros((true_X.shape[0], time_instances))
acc_Z = np.zeros((true_X.shape[0], time_instances))

heading = np.zeros((true_X.shape[0], time_instances))

heading_deg = np.zeros((true_X.shape[0], time_instances))

gps_error = np.zeros((true_X.shape[0], true_X.shape[1]))
intact_gps_error = np.zeros((true_X.shape[0], true_X.shape[1]))
lapl_error = np.zeros((true_X.shape[0], true_X.shape[1]))
local_lapl_error = np.zeros((true_X.shape[0], true_X.shape[1]))
tcl_error = np.zeros((true_X.shape[0], true_X.shape[1]))
ekf_error = np.zeros((true_X.shape[0], true_X.shape[1]))
cgcl_ekf_error = np.zeros((true_X.shape[0], true_X.shape[1]))
ekf_alone_error = np.zeros((true_X.shape[0], true_X.shape[1]))
lgcl_ekf_local_error = np.zeros((true_X.shape[0], true_X.shape[1]))
ekf_local_error = np.zeros((true_X.shape[0], true_X.shape[1]))

mse_gps_error = np.zeros(true_X.shape[1])
mse_intact_gps_error = np.zeros(true_X.shape[1])
mse_lapl_error = np.zeros(true_X.shape[1])
mse_local_lapl_error = np.zeros(true_X.shape[1])
mse_tcl_error = np.zeros(true_X.shape[1])
mse_ekf_error = np.zeros(true_X.shape[1])
mse_cgcl_ekf_error = np.zeros(true_X.shape[1])
mse_ekf_alone_error = np.zeros(true_X.shape[1])
mse_lgcl_ekf_local_error = np.zeros(true_X.shape[1])
mse_ekf_local_error = np.zeros(true_X.shape[1])

max_gps_error = np.zeros(true_X.shape[1])
max_intact_gps_error = np.zeros(true_X.shape[1])
max_lapl_error = np.zeros(true_X.shape[1])
max_local_lapl_error = np.zeros(true_X.shape[1])
max_tcl_error = np.zeros(true_X.shape[1])
max_ekf_error = np.zeros(true_X.shape[1])
max_cgcl_ekf_error = np.zeros(true_X.shape[1])
max_lgcl_ekf_local_error = np.zeros(true_X.shape[1])
max_ekf_local_error = np.zeros(true_X.shape[1])

CGCL_distance_of_est_from_gps = np.zeros((true_X.shape[0], true_X.shape[1]))
LGCL_distance_of_est_from_gps = np.zeros((true_X.shape[0], true_X.shape[1]))

EKF_CGCL_distance_of_est_from_gps = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_LGCL_distance_of_est_from_gps = np.zeros((true_X.shape[0], true_X.shape[1]))

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
        
        acc_X[i,j] = np.copy(locations[i+step,8])
        acc_Y[i,j] = np.copy(locations[i+step,9])
        acc_Z[i,j] = np.copy(locations[i+step,10])
        
        ang_vel_X[i,j] = np.copy(locations[i+step,11]) 
        ang_vel_Y[i,j] = np.copy(locations[i+step,12]) 
        ang_vel_Z[i,j] = np.copy(locations[i+step,13])
        
        heading[i,j] = math.radians(np.copy(locations[i+step,4]))
        
        heading_deg[i,j] = (np.copy(locations[i+step,4]))
        
        step += true_X.shape[0]
        #step += 200

velocity = np.zeros((true_X.shape[0], time_instances))
angular = np.zeros((true_X.shape[0], time_instances))
accelaration = np.zeros((true_X.shape[0], time_instances))

for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]-1):
        
        #angular[i,j+1] = (heading[i,j+1] - heading[i,j]) / Dt
        
        angular[i,j] = (heading[i,j+1] - heading[i,j]) / Dt
        
        if (j == true_X.shape[1]-1):
            angular[i,j] = math.sqrt(ang_vel_X[i,j]**2 + ang_vel_Y[i,j]**2)
        
   
#angular[:,-1] = np.mean(np.array([angular[:,-5], angular[:,-4], angular[:,-3], angular[:,-2]]).T, 1)

sin_heading = np.zeros((true_X.shape[0], true_X.shape[1]))
cos_heading = np.zeros((true_X.shape[0], true_X.shape[1]))

for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
        sin_heading[i,j] = math.sin(heading[i,j])
        cos_heading[i,j] = math.cos(heading[i,j])
        
for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
       
           
        velocity[i,j] = math.sqrt(vel_X[i,j]**2 + vel_Y[i,j]**2 )
        accelaration[i,j] = math.sqrt(acc_X[i,j]**2 + acc_Y[i,j]**2 )
            
        
temp_traj_x = np.zeros((true_X.shape[0], true_X.shape[1]))  
temp_traj_y = np.zeros((true_X.shape[0], true_X.shape[1]))

KM_comp_x = np.zeros((true_X.shape[0], true_X.shape[1]))  
KM_comp_y = np.zeros((true_X.shape[0], true_X.shape[1]))

#vehicle_idx = random.randint(0, true_X.shape[0]-11)
#vehicle_idx = 0
#stop = true_X.shape[0]

for j in range (true_X.shape[0]):
    
    for i in range (true_X.shape[1]):
        
        if (i == 0):
            temp_traj_x[j,i] = true_X[j,i]
            temp_traj_y[j,i] = true_Y[j,i]
          
        else:
            
            #KM_comp_x[j,i] = vel_X[j,i]*Dt + 0.5*acc_X[j,i]*(Dt**2) + np.random.normal(0, sigma_trans)
            #KM_comp_y[j,i] = vel_Y[j,i]*Dt + 0.5*acc_Y[j,i]*(Dt**2) + np.random.normal(0, sigma_trans)
            '''
            KM_comp_x[j,i] = velocity[j,i]*math.cos(heading[j,i])*Dt + np.random.normal(0, sigma_trans)
            KM_comp_y[j,i] = velocity[j,i]*math.sin(heading[j,i])*Dt + np.random.normal(0, sigma_trans)
            
            temp_x[j,i] = temp_x[j,i-1] + KM_comp_x[j,i]
            temp_y[j,i] = temp_y[j,i-1] + KM_comp_y[j,i]
            '''
            
            
            if ((angular[j,i]) == 0):
                
                #KM_comp_x[j,i] = vel_X[j,i]*Dt + 0.5*acc_X[j,i]*(Dt**2) + np.random.normal(mu, sigma_trans)
                #KM_comp_y[j,i] = vel_Y[j,i]*Dt + 0.5*acc_Y[j,i]*(Dt**2) + np.random.normal(mu, sigma_trans)
                
                KM_comp_x[j,i] = velocity[j,i]*math.cos(heading[j,i])*Dt + np.random.normal(0, sigma_trans)
                KM_comp_y[j,i] = velocity[j,i]*math.sin(heading[j,i])*Dt + np.random.normal(0, sigma_trans)
            
                temp_traj_x[j,i] = temp_traj_x[j,i-1] + KM_comp_x[j,i]
                temp_traj_y[j,i] = temp_traj_y[j,i-1] + KM_comp_y[j,i]
                
            
            else:    
                
                KM_comp_x[j,i] = (-velocity[j,i]/angular[j,i])*np.sin(heading[j,i]) + (velocity[j,i]/angular[j,i])*np.sin(heading[j,i] + angular[j,i]*Dt) + np.random.normal(0, sigma_trans)
                KM_comp_y[j,i] = (velocity[j,i]/angular[j,i])*np.cos(heading[j,i]) + (-velocity[j,i]/angular[j,i])*np.cos(heading[j,i] + angular[j,i]*Dt) + np.random.normal(0, sigma_trans)
                
                temp_traj_x[j,i] = temp_traj_x[j,i-1] + KM_comp_x[j,i]
                temp_traj_y[j,i] = temp_traj_y[j,i-1] + KM_comp_y[j,i]
            
                
noise_X = np.zeros((true_X.shape[0], true_X.shape[1]))
noise_Y = np.zeros((true_X.shape[0], true_X.shape[1]))


for i in range (true_X.shape[1]):
    
    gps_noise_x = np.zeros(true_X.shape[0])
    gps_noise_y = np.zeros(true_X.shape[0])
    
    gps_noise_x = GPS_noise(mu, sigma_x, sigma_y, true_X.shape[0])[0][:true_X.shape[0]]
    gps_noise_y = GPS_noise(mu, sigma_x, sigma_y, true_X.shape[0])[1][:true_X.shape[0]]
    
    noise_X[:,i] = true_X[:,i] + gps_noise_x
    noise_Y[:,i] = true_Y[:,i] + gps_noise_y



    
    
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
        

        
L = np.zeros((D.shape[0], D.shape[1], D.shape[2]))  

      
for i in range (L.shape[0]):

    L[i,:,:] = Deg[i,:,:] - A[i,:,:]     


    
    
Azim_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))
AoA_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

noisy_Azim_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))
noisy_AoA_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))

tan_noisy_AoA_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))
deg_noisy_AoA_Angle = np.zeros((true_X.shape[1], len(list_of_vehicles), len(list_of_vehicles)))


Concatenated_TCL_Points_X = np.zeros((true_X.shape[0], true_X.shape[1]))
Concatenated_TCL_Points_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

B_X = np.zeros((2*true_X.shape[0], true_X.shape[1]))
B_Y = np.zeros((2*true_X.shape[0], true_X.shape[1]))

Concatenated_CGCL_Points_X = np.zeros((2*true_X.shape[0], true_X.shape[1]))
Concatenated_CGCL_Points_Y = np.zeros((2*true_X.shape[0], true_X.shape[1]))

Concatenated_LGCL_Points_X = np.zeros((2*true_X.shape[0], true_X.shape[1]))
Concatenated_LGCL_Points_Y = np.zeros((2*true_X.shape[0], true_X.shape[1]))

max_degree = np.zeros((true_X.shape[0], true_X.shape[1]))

EKF_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

EKF_CGCL_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_CGCL_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))


EKF_alone_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_alone_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

EKF_local_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_local_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

EKF_LGCL_recon_X = np.zeros((true_X.shape[0], true_X.shape[1]))
EKF_LGCL_recon_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

true_delta_X = np.zeros((true_X.shape[0], true_X.shape[1]))
true_delta_Y = np.zeros((true_X.shape[0], true_X.shape[1]))

est_delta_X = np.zeros((true_X.shape[0], true_X.shape[1]))
est_delta_Y = np.zeros((true_X.shape[0], true_X.shape[1]))
for i in range (true_X.shape[1]):
    
    l = 0
    for j in (list_of_vehicles):
        u = 0
        
        for k in (list_of_vehicles):
            
            if (A[i,l,u] == 1):
                
                Azim_Angle[i,l,u] = CalculateAzimuthAngle(true_X[j,i], true_X[k,i], true_Y[j,i], true_Y[k,i])
                
                noisy_Azim_Angle[i,l,u] = Azim_Angle[i,l,u] + np.random.normal(mu, math.radians(sigma_a))
                
                AoA_Angle[i,l,u] = CalculateAoA(true_X[j,i], true_X[k,i], true_Y[j,i], true_Y[k,i])
                
                noisy_AoA_Angle[i,l,u] = AoA_Angle[i,l,u] + np.random.normal(mu, math.radians(sigma_a))
                
                if (noisy_AoA_Angle[i,l,u] > math.radians(360)):
                    noisy_AoA_Angle[i,l,u] = math.radians(360)
                if (noisy_AoA_Angle[i,l,u] < math.radians(-360)):
                    noisy_AoA_Angle[i,l,u] = math.radians(-360)
                    
                deg_noisy_AoA_Angle[i,l,u] = math.degrees(noisy_AoA_Angle[i,l,u])
                
                tan_noisy_AoA_Angle[i,l,u] = (math.tan(noisy_AoA_Angle[i,l,u]))
                    
            u += 1
         
        l += 1
        
rank_of_L = np.zeros(true_X.shape[1])

for i in range (rank_of_L.shape[0]):
    
    rank_of_L[i] = matrix_rank(L[i,:,:])


clusters, clusters_size, clusters_index = Create_Clusters(true_X, A, Deg)

for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
        
        intact_gps_error[i,j] = norm(np.array([true_X[i,j] - noise_X[i,j] , true_Y[i,j] - noise_Y[i,j]]), 2)       

    
#state = true_X.shape[0]

size_of_measurement_model = 2

mhi_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Q = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

R = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

G = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

L_bar = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

mhi_alone_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_alone_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_alone_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_alone_hat_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

mhi_alone_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_alone_hat_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_alone_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_alone_hat_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

mhi_alone_local_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_alone_hat_local_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_alone_local_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_alone_hat_local_lapl_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

flag_for_EKF_alone = np.zeros((true_X.shape[0], true_X.shape[1]))

count_for_degrees_in_Laplacian = 0
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
        '''
        for u in range (int(clusters_size[i,j])):
    
            Q[i,j][u,u] *= 1
    
        for u in range (int(2*clusters_size[i,j]), int(3*clusters_size[i,j])):
    
            Q[i,j][u,u] *= 1
        
        for u in range (int(clusters_size[i,j]), int(2*clusters_size[i,j])):
    
            Q[i,j][u,u] *= sigma_x**2
    
        for u in range (int(3*clusters_size[i,j]), int(4*clusters_size[i,j])):
    
            Q[i,j][u,u] *= sigma_y**2 
        '''
        R[i,j] = np.zeros((int(2*clusters_size[i,j]), int(2*clusters_size[i,j])), dtype=np.object)
        
        R[i,j] = np.eye(int(2*clusters_size[i,j]))
        
        R[i,j] = R[i,j].astype(float)
        '''
        for u in range (R[i,j].shape[0]):
    
            R[i,j][u,u] *= sigma_trans**2
        '''
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
        
        mhi_alone_lapl_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_lapl_KF[i,j] = mhi_alone_lapl_KF[i,j].astype(float)
        
        mhi_alone_hat_lapl_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_hat_lapl_KF[i,j] = mhi_alone_hat_lapl_KF[i,j].astype(float)
        
        Sigma_alone_lapl_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_lapl_KF[i,j] = Sigma_alone_lapl_KF[i,j].astype(float)
        
        Sigma_alone_hat_lapl_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_hat_lapl_KF[i,j] = Sigma_alone_hat_lapl_KF[i,j].astype(float)
        
        mhi_alone_local_lapl_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_local_lapl_KF[i,j] = mhi_alone_local_lapl_KF[i,j].astype(float)
        
        mhi_alone_hat_local_lapl_KF[i,j] = np.zeros(2, dtype=np.object) 
        mhi_alone_hat_local_lapl_KF[i,j] = mhi_alone_hat_local_lapl_KF[i,j].astype(float)
        
        Sigma_alone_local_lapl_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_local_lapl_KF[i,j] = Sigma_alone_local_lapl_KF[i,j].astype(float)
        
        Sigma_alone_hat_local_lapl_KF[i,j] = np.zeros((2,2), dtype=np.object) 
        Sigma_alone_hat_local_lapl_KF[i,j] = Sigma_alone_hat_local_lapl_KF[i,j].astype(float)


mhi_local_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
mhi_hat_local_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Sigma_local_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)
Sigma_hat_local_KF = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

Q_local = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

R_local = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

G_local = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

g = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

neighbors_mat = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

node_and_its_cluster_size = np.zeros((true_X.shape[0], true_X.shape[1]))

for idx_time in range (true_X.shape[1]):

    for i in range (true_X.shape[0]):
        
        neighbors = np.zeros(int(Deg[idx_time,i,i] + 1 ))
        
        neighbors[-1] = i
        
        neighbors[:-1] = np.argwhere( (1 == A[idx_time,i,:]) ).T
       
        neighbors_mat[i,idx_time] = np.zeros((int(Deg[idx_time, i,i])), dtype=np.object)
        
        neighbors_mat[i,idx_time] = neighbors_mat[i,idx_time].astype(float)
        
        neighbors_mat[i,idx_time] = np.argwhere( (1 == A[idx_time,i,:]) )
        
        if (int(clusters_index[i,idx_time]) != 5000):
            
            node_and_its_cluster_size[i,idx_time] = clusters_size[int(clusters_index[i,idx_time]),idx_time]
        
        mhi_local_KF[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
        
        mhi_local_KF[i,idx_time] = mhi_local_KF[i,idx_time].astype(float)
        
        mhi_hat_local_KF[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
        
        mhi_hat_local_KF[i,idx_time] = mhi_hat_local_KF[i,idx_time].astype(float)
        
        Sigma_local_KF[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1), 2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
       
        Sigma_local_KF[i,idx_time] = Sigma_local_KF[i,idx_time].astype(float)
        
        Sigma_hat_local_KF[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1), 2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
       
        Sigma_hat_local_KF[i,idx_time] = Sigma_local_KF[i,idx_time].astype(float)
        
        Q_local[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 2), 2*int(Deg[idx_time, i,i] + 2)), dtype=np.object)
        
        Q_local[i,idx_time] = np.eye(2*int(Deg[idx_time, i,i] + 2))
        
        Q_local[i,idx_time] = Q_local[i,idx_time].astype(float)
        
        '''
        for k in range (int(Deg[idx_time, i,i])):
            
            Q_local[i,idx_time][k,k] = sigma_x**2
       
        for k in range (int(Deg[idx_time, i,i]), 2*int(Deg[idx_time, i,i])):
            
            Q_local[i,idx_time][k,k] = sigma_y**2
        
        Q_local[i,idx_time][-4,-4] = sigma_x**2
        
        Q_local[i,idx_time][-2,-2] = sigma_y**2
        '''
        
        R_local[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1), 2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
        
        R_local[i,idx_time] = np.eye(2*int(Deg[idx_time, i,i] + 1))
        
        R_local[i,idx_time] = (1**2)*R_local[i,idx_time].astype(float)
        
        G_local[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1), 2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
        
        G_local[i,idx_time][:2*int(Deg[idx_time, i,i] + 1),:2*int(Deg[idx_time, i,i] + 1)] = 1
        
        G_local[i,idx_time] = G_local[i,idx_time].astype(float)
        
        G_local[i,idx_time][2*int(Deg[idx_time, i,i] + 1):,2*int(Deg[idx_time, i,i] + 1):] = 1
        
        G_local[i,idx_time] = G_local[i,idx_time].astype(float)
        
        g[i,idx_time] = np.zeros((2*int(Deg[idx_time, i,i] + 1)), dtype=np.object)
        
        g[i,idx_time] = g[i,idx_time].astype(float)
        
        u = 0
        
        for j in neighbors:
            
            g[i,idx_time][u] = KM_comp_x[int(j), idx_time]
            
            g[i,idx_time][u + int(Deg[idx_time, i,i] + 1)] = KM_comp_y[int(j), idx_time]
            
            u += 1

list_of_attacked_vehicles = np.zeros((true_X.shape[0], true_X.shape[1]), dtype=np.object)

for i in range (true_X.shape[0]):
    
    for j in range (true_X.shape[1]):

        list_of_attacked_vehicles[i,j] = []
        
        if (clusters_size[i,j] != 0):

          size = int(round(0.2*clusters_size[i,j]))
          
          
          if (size != 0):
              
              
              for k in range (size):
                  
                  vehicle_idx = random.choice(clusters[i,j])
                  
                  while ((vehicle_idx in list_of_attacked_vehicles[i,j])):
                      
                      vehicle_idx = random.choice(clusters[i,j])
                      
                  list_of_attacked_vehicles[i,j].append(vehicle_idx) 
                  
          #else:
                  #list_of_attacked_vehicles[i,j] = []
                  
        #else:
            
            #list_of_attacked_vehicles[i,j] = []
                
              

##################!!!!!!!!!!!!!! Attack initialization !!!!!!!!#################


    
for i in range(true_X.shape[0]):
    
    for j in range (true_X.shape[1]):
   
        if (len(list_of_attacked_vehicles[i,j]) != 0):
            
            for k in list_of_attacked_vehicles[i,j]:
                
                noise_X[k,j] += np.random.uniform(5, 40)
                            
                noise_Y[k,j] += np.random.uniform(5, 40)



##################!!!!!!!!!!!!!! Attack initialization !!!!!!!!#################
      
    
for i in range (true_X.shape[0]):
    for j in range (true_X.shape[1]):
        
        gps_error[i,j] = norm(np.array([true_X[i,j] - noise_X[i,j] , true_Y[i,j] - noise_Y[i,j]]), 2)

for i in range (true_X.shape[1]):
    
    mse_gps_error[i] = np.sum(gps_error[:,i]**2)/true_X.shape[0]
    
    max_gps_error[i] = np.max(gps_error[:,i])
    
    mse_intact_gps_error[i] = np.sum(intact_gps_error[:,i]**2)/true_X.shape[0]
    
    max_intact_gps_error[i] = np.max(intact_gps_error[:,i])
        
for idx_time in range (true_X.shape[1]):
    
    #if (idx_time == 0):
    print ('Time index: ', idx_time)
    
    if (np.array_equal(A[idx_time,:,:],A[idx_time,:,:].T) == 0):
        print ('Problem with A')
        
    for idx_test_cluster in range(true_X.shape[0]):

       
##################!!!!!!!!!!!!!! EKF with GPS only !!!!!!!!################# 
        '''
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
                        
                Sigma_alone_hat_KF[idx_test_cluster, idx_time] = G_alone@Sigma_alone_KF[idx_test_cluster, idx_time-1]@G_alone.T + (sigma_trans**2)*np.eye(state)
                                
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
        '''
##################!!!!!!!!!!!!!! EKF with GPS only !!!!!!!!#################
         
##################!!!!!!!!!!!!!! Centralized Laplacian !!!!!!!!#################           
        if (Deg[idx_time, idx_test_cluster, idx_test_cluster] == 0):
            
            lapl_error[idx_test_cluster, idx_time] = gps_error[idx_test_cluster, idx_time]
            
            Concatenated_CGCL_Points_X[idx_test_cluster,idx_time] = noise_X[idx_test_cluster,idx_time]
            Concatenated_CGCL_Points_Y[idx_test_cluster,idx_time] = noise_Y[idx_test_cluster,idx_time]
            
            Concatenated_LGCL_Points_X[idx_test_cluster,idx_time] = noise_X[idx_test_cluster,idx_time]
            Concatenated_LGCL_Points_Y[idx_test_cluster,idx_time] = noise_Y[idx_test_cluster,idx_time]
            #
            
            ekf_error[idx_test_cluster,idx_time], EKF_recon_X[idx_test_cluster,idx_time], EKF_recon_Y[idx_test_cluster,idx_time] = EKF_with_GPS(flag_for_EKF_alone, mhi_alone_KF, mhi_alone_hat_KF, Sigma_alone_KF, Sigma_alone_hat_KF, Deg, true_X, true_Y, idx_time, idx_test_cluster, KM_comp_x, KM_comp_y, sigma_x, sigma_y)
            '''
            ekf_error[idx_test_cluster,idx_time] = ekf_alone_error[idx_test_cluster,idx_time]
            
            EKF_recon_X[idx_test_cluster,idx_time] = EKF_alone_recon_X[idx_test_cluster,idx_time]
            
            EKF_recon_Y[idx_test_cluster,idx_time] = EKF_alone_recon_Y[idx_test_cluster,idx_time]
            '''
            lgcl_ekf_local_error[idx_test_cluster, idx_time] = np.copy(ekf_error[idx_test_cluster, idx_time])
            
            EKF_local_recon_X[idx_test_cluster,idx_time] = np.copy(EKF_recon_X[idx_test_cluster,idx_time])
        
            EKF_local_recon_Y[idx_test_cluster,idx_time] = np.copy(EKF_recon_Y[idx_test_cluster,idx_time])
            
            cgcl_ekf_error[idx_test_cluster, idx_time] = np.copy(ekf_error[idx_test_cluster, idx_time])
            
            EKF_CGCL_recon_X[idx_test_cluster,idx_time] = np.copy(EKF_recon_X[idx_test_cluster,idx_time])
        
            EKF_CGCL_recon_Y[idx_test_cluster,idx_time] = np.copy(EKF_recon_Y[idx_test_cluster,idx_time])
            
        else:
            
            if (sum(clusters[idx_test_cluster][idx_time]) > 0):
                
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
                                
                                '''  
                                if (math.degrees(noisy_Azim_Angle[idx_time,i,j]) > 10):    
                                    
                                    delta_X[u] += -abs(noisy_D[idx_time,i,j])*math.sin(abs(noisy_Azim_Angle[idx_time,i,j]))
                                    delta_Y[u] += -abs(noisy_D[idx_time,i,j])*math.cos(abs(noisy_Azim_Angle[idx_time,i,j]))
                                else:
                                    count_for_degrees_in_Laplacian += 1
                                    delta_X[u] += noise_X[i,idx_time] - noise_X[j,idx_time]
                                    delta_Y[u] += noise_Y[i,idx_time] - noise_Y[j,idx_time]
                                '''
                               
                                true_delta_X[i,idx_time] += true_X[i,idx_time] - true_X[j,idx_time]
                                true_delta_Y[i,idx_time] += true_Y[i,idx_time] - true_Y[j,idx_time]
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
                
                max_degree[idx_test_cluster, idx_time] = np.max(test_L)
                
                anchors_size = int(1.0*clusters_size[idx_test_cluster, idx_time])
                
                anchors_index = np.arange(anchors_size)
                
                anchors = np.copy(noisy_test_Points[:anchors_size,:])
                
                test_lapl_Points = np.zeros((test_Points.shape[0], test_Points.shape[1]))
                    
                test_lapl_error = np.zeros((test_Points.shape[0]))
                    
                test_L_bar = np.zeros((test_Points.shape[0] + anchors_index.size, test_Points.shape[1]))
                    
                #test_lapl_Points, test_lapl_error, test_L_bar = Solve_The_System(test_Points, test_L, anchors_index, anchors, delta_X, delta_Y)
                
                test_lapl_error, test_L_bar, test_lapl_Points[:,0], test_lapl_Points[:,1] = Robust_Laplacian(test_Points, test_L, anchors, anchors_index, delta_X, delta_Y)[:4]
                
                
                if (anchors_size < int(clusters_size[idx_test_cluster, idx_time])):
                    
                    additional_anchors = np.zeros((int(clusters_size[idx_test_cluster, idx_time])-anchors_size, int(clusters_size[idx_test_cluster, idx_time])))
                    
                    u = 0
                    for i in range (additional_anchors.shape[0]):
                        additional_anchors[i, additional_anchors.shape[1]-1-u] = 1
                        u += 1
                    
                    
                    L_bar[idx_test_cluster,idx_time] = np.concatenate((test_L_bar, additional_anchors))
                    
                else:
                    
                    L_bar[idx_test_cluster,idx_time] = test_L_bar
                
                
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                    lapl_error[i,idx_time] = test_lapl_error[u]
                    
                    Concatenated_CGCL_Points_X[i,idx_time] = test_lapl_Points[u,0]
                    Concatenated_CGCL_Points_Y[i,idx_time] = test_lapl_Points[u,1]
                    
                    est_delta_X[i,idx_time] = delta_X[u]
                    est_delta_Y[i,idx_time] = delta_Y[u]
                    
                    CGCL_distance_of_est_from_gps[i,idx_time] = norm(np.array([test_lapl_Points[u,0] - noise_X[i,idx_time], test_lapl_Points[u,1] - noise_Y[i,idx_time]]))
                    u += 1
                
        
##################!!!!!!!!!!!!!! EKF + Laplacian !!!!!!!!#################
                  
                state = int(clusters_size[idx_test_cluster, idx_time])
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                          
                    EKF_CGCL_recon_X[i,idx_time] = EKF_with_Lapl(mhi_KF, mhi_hat_KF, Sigma_KF, Sigma_hat_KF, clusters, idx_time, idx_test_cluster, L_bar, test_lapl_Points, delta_X, delta_Y, G, Q, R, test_A, anchors)[idx_test_cluster, idx_time].T[u]
                    EKF_CGCL_recon_Y[i,idx_time] = EKF_with_Lapl(mhi_KF, mhi_hat_KF, Sigma_KF, Sigma_hat_KF, clusters, idx_time, idx_test_cluster, L_bar, test_lapl_Points, delta_X, delta_Y, G, Q, R, test_A, anchors)[idx_test_cluster, idx_time].T[u+state]
                            
                    cgcl_ekf_error[i,idx_time] = norm(np.array([EKF_CGCL_recon_X[i,idx_time] - true_X[i,idx_time], EKF_CGCL_recon_Y[i,idx_time] - true_Y[i,idx_time]]))
                    
                    #EKF_CGCL_distance_of_est_from_gps[i,idx_time] = norm(np.array([EKF_recon_X[i,idx_time] - noise_X[i,idx_time], EKF_recon_Y[i,idx_time] - noise_Y[i,idx_time]]))
                    
                    u += 1  
                  
##################!!!!!!!!!!!!!! EKF + Laplacian !!!!!!!!#################
   
##################!!!!!!!!!!!!!! CGCL -> EKF !!!!!!!!#################                 
                
        #sigma_x, sigma_y, sigma_trans
        #1,1,1
        EKF_recon_X[idx_test_cluster,idx_time] = EKF_with_Lapl_Est(noise_X, noise_Y, Concatenated_CGCL_Points_X, Concatenated_CGCL_Points_Y, mhi_alone_lapl_KF, mhi_alone_hat_lapl_KF, Sigma_alone_lapl_KF, Sigma_alone_hat_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_y, 1,1,1).T[0]
        EKF_recon_Y[idx_test_cluster,idx_time] = EKF_with_Lapl_Est(noise_X, noise_Y, Concatenated_CGCL_Points_X, Concatenated_CGCL_Points_Y, mhi_alone_lapl_KF, mhi_alone_hat_lapl_KF, Sigma_alone_lapl_KF, Sigma_alone_hat_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_y, 1,1,1).T[1]
            
        ekf_error[idx_test_cluster,idx_time] = norm(np.array([EKF_recon_X[idx_test_cluster,idx_time] - true_X[idx_test_cluster,idx_time], EKF_recon_Y[idx_test_cluster,idx_time] - true_Y[idx_test_cluster,idx_time]]))
        
##################!!!!!!!!!!!!!! CGCL -> EKF !!!!!!!!#################
           
##################!!!!!!!!!!!!!! Centralized Laplacian !!!!!!!!#################
                   
##################!!!!!!!!!!!!!! Local Laplacian !!!!!!!!#################
        '''
        if (Deg[idx_time, idx_test_cluster, idx_test_cluster] == 0):
            
            local_lapl_error[idx_test_cluster, idx_time] = gps_error[idx_test_cluster, idx_time]
            
        else:
            
            if (sum(clusters[idx_test_cluster][idx_time]) > 0):
                
                test_local_lapl_Points = np.zeros((test_Points.shape[0], test_Points.shape[1]))
                    
                test_local_EKF_Points = np.zeros((test_Points.shape[0], test_Points.shape[1]))
                
                test_local_lapl_error = np.zeros((test_Points.shape[0]))
                
                test_local_ekf_error = np.zeros((test_Points.shape[0]))
                
                #test_local_lapl_Points[:,0], test_local_lapl_Points[:,1], test_local_lapl_error = Distributed_Lapl(test_Deg, test_A, delta_X, delta_Y, noisy_test_Points[:,0], noisy_test_Points[:,1], test_Points[:,0], test_Points[:,1])
                
                test_local_lapl_Points[:,0], test_local_lapl_Points[:,1], test_local_EKF_Points[:,0], test_local_EKF_Points[:,1], test_local_lapl_error, test_local_ekf_error = Distributed_Lapl(neighbors_mat, clusters, mhi_local_KF, mhi_hat_local_KF, Sigma_local_KF, Sigma_hat_local_KF, idx_time, idx_test_cluster, G_local, Q_local, R_local, g, test_Deg, test_A, delta_X, delta_Y, noisy_test_Points[:,0], noisy_test_Points[:,1], test_Points[:,0], test_Points[:,1])
                
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                    
                    local_lapl_error[i,idx_time] = test_local_lapl_error[u]
                    lgcl_ekf_local_error[i,idx_time] = test_local_ekf_error[u]
                    
                    EKF_LGCL_recon_X[i,idx_time] = test_local_EKF_Points[u,0]
                    EKF_LGCL_recon_Y[i,idx_time] = test_local_EKF_Points[u,1]
                    
                    Concatenated_LGCL_Points_X[i,idx_time] = test_local_lapl_Points[u,0]
                    Concatenated_LGCL_Points_Y[i,idx_time] = test_local_lapl_Points[u,1]
                    
                    u += 1
                    
        '''
##################!!!!!!!!!!!!!! Local Laplacian !!!!!!!!#################  

##################!!!!!!!!!!!!!! LGCL -> EKF !!!!!!!!#################                 
        '''     
        #sigma_x, sigma_y, sigma_trans
        #1,1,1
        EKF_local_recon_X[idx_test_cluster,idx_time] = EKF_with_Local_Lapl_Est(noise_X, noise_Y, Concatenated_LGCL_Points_X, Concatenated_LGCL_Points_Y, mhi_alone_local_lapl_KF, mhi_alone_hat_local_lapl_KF, Sigma_alone_local_lapl_KF, Sigma_alone_hat_local_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_y, 1,1,1).T[0]
        EKF_local_recon_Y[idx_test_cluster,idx_time] = EKF_with_Local_Lapl_Est(noise_X, noise_Y, Concatenated_LGCL_Points_X, Concatenated_LGCL_Points_Y, mhi_alone_local_lapl_KF, mhi_alone_hat_local_lapl_KF, Sigma_alone_local_lapl_KF, Sigma_alone_hat_local_lapl_KF, idx_time, idx_test_cluster, Deg, KM_comp_x, KM_comp_y, 1,1,1).T[1]
            
        ekf_local_error[idx_test_cluster,idx_time] = norm(np.array([EKF_local_recon_X[idx_test_cluster,idx_time] - true_X[idx_test_cluster,idx_time], EKF_local_recon_Y[idx_test_cluster,idx_time] - true_Y[idx_test_cluster,idx_time]]))
        '''
##################!!!!!!!!!!!!!! LGCL -> EKF !!!!!!!!#################
                  
##################!!!!!!!!!!!!!! TCL-MLE !!!!!!!!################# 
        #
        if (Deg[idx_time, idx_test_cluster, idx_test_cluster] == 0):
            
            tcl_error[idx_test_cluster, idx_time] = gps_error[idx_test_cluster, idx_time]
            
            Concatenated_TCL_Points_X[idx_test_cluster, idx_time] = noise_X[idx_test_cluster, idx_time]
            Concatenated_TCL_Points_Y[idx_test_cluster, idx_time] = noise_Y[idx_test_cluster, idx_time]
            
        else:
            if (sum(clusters[idx_test_cluster][idx_time]) > 0):
                
                test_tcl_Points = np.zeros((test_Points.shape[0], test_Points.shape[1]))
                    
                test_tcl_error = np.zeros((test_Points.shape[0]))
                
                test_tan_noisy_AoA_Angle = np.zeros((test_Points.shape[0],test_Points.shape[0]))
                test_deg_noisy_AoA_Angle = np.zeros((test_Points.shape[0],test_Points.shape[0]))
                test_noisy_D = np.zeros((test_Points.shape[0],test_Points.shape[0]))
                
                u = 0
                
                for i in clusters[idx_test_cluster][idx_time]:
                    
                    l = 0
                    for j in clusters[idx_test_cluster][idx_time]:
                        
                        if (A[idx_time,i,j] == 1):
                                
                               test_tan_noisy_AoA_Angle[u,l] = tan_noisy_AoA_Angle[idx_time,i,j]
                               test_deg_noisy_AoA_Angle[u,l] = deg_noisy_AoA_Angle[idx_time,i,j]
                               test_noisy_D[u,l] = noisy_D[idx_time,i,j]
                        l += 1
                                
                    u += 1
                
               
                    
                #test_tcl_Points[:,0], test_tcl_Points[:,1], test_tcl_error = Optimization_AoA_GD(test_A, test_noisy_D, test_tan_noisy_AoA_Angle, noisy_test_Points[:,0], noisy_test_Points[:,1], test_Points[:,0], test_Points[:,1], test_deg_noisy_AoA_Angle, noisy_test_Points[:,0], noisy_test_Points[:,1])
               
                test_tcl_Points[:,0], test_tcl_Points[:,1], test_tcl_error = Robust_TCLMLE(test_A, test_noisy_D, noisy_test_Points[:,0], noisy_test_Points[:,1], test_tan_noisy_AoA_Angle, test_deg_noisy_AoA_Angle, test_Points)
                
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                    tcl_error[i,idx_time] = test_tcl_error[u]
                    
                    Concatenated_TCL_Points_X[i,idx_time] = test_tcl_Points[u,0]
                    Concatenated_TCL_Points_Y[i,idx_time] = test_tcl_Points[u,1]
                    
                    u += 1
        #
##################!!!!!!!!!!!!!! TCL-MLE !!!!!!!!################# 
                    
    mse_lapl_error[idx_time] = (np.sum(lapl_error[:,idx_time]**2)/true_X.shape[0])
    mse_local_lapl_error[idx_time] = np.sum(local_lapl_error[:,idx_time]**2)/true_X.shape[0]
    mse_tcl_error[idx_time] = np.sum(tcl_error[:,idx_time]**2)/true_X.shape[0]
    mse_ekf_error[idx_time] = (np.sum(ekf_error[:,idx_time]**2)/true_X.shape[0])
    mse_cgcl_ekf_error[idx_time] = (np.sum(cgcl_ekf_error[:,idx_time]**2)/true_X.shape[0])
    #mse_ekf_alone_error[idx_time] = (np.sum(ekf_alone_error[:,idx_time]**2)/true_X.shape[0])
    mse_lgcl_ekf_local_error[idx_time] = (np.sum(lgcl_ekf_local_error[:,idx_time]**2)/true_X.shape[0])
    mse_ekf_local_error[idx_time] = (np.sum(ekf_local_error[:,idx_time]**2)/true_X.shape[0])
    
    max_lapl_error[idx_time] = np.max(lapl_error[:,idx_time])
    max_local_lapl_error[idx_time] = np.max(local_lapl_error[:,idx_time])
    max_tcl_error[idx_time] = np.max(tcl_error[:,idx_time])
    max_ekf_error[idx_time] = np.max(ekf_error[:,idx_time])
    max_cgcl_ekf_error[idx_time] = np.max(cgcl_ekf_error[:,idx_time])
    max_lgcl_ekf_local_error[idx_time] = np.max(lgcl_ekf_local_error[:,idx_time])
    max_ekf_local_error[idx_time] = np.max(ekf_local_error[:,idx_time])


sorted_x_mse_gps_error = np.sort(mse_gps_error)
sorted_x_mse_intact_gps_error = np.sort(mse_intact_gps_error)
sorted_x_mse_lapl_error = np.sort(mse_lapl_error)
sorted_x_mse_local_lapl_error = np.sort(mse_local_lapl_error)
sorted_x_mse_tcl_error = np.sort(mse_tcl_error)
sorted_x_mse_ekf_error = np.sort(mse_ekf_error)
sorted_x_mse_cgcl_ekf_error = np.sort(mse_cgcl_ekf_error)
sorted_x_mse_ekf_alone_error = np.sort(mse_ekf_alone_error)
sorted_x_mse_lgcl_ekf_local_error = np.sort(mse_lgcl_ekf_local_error)
sorted_x_mse_ekf_local_error = np.sort(mse_ekf_local_error)

sorted_y_mse_gps_error = np.arange(len(np.sort(mse_gps_error)))/float(len(mse_gps_error))
sorted_y_mse_intact_gps_error = np.arange(len(np.sort(mse_intact_gps_error)))/float(len(mse_intact_gps_error))
sorted_y_mse_lapl_error = np.arange(len(np.sort(mse_lapl_error)))/float(len(mse_lapl_error))
sorted_y_mse_local_lapl_error = np.arange(len(np.sort(mse_local_lapl_error)))/float(len(mse_local_lapl_error))
sorted_y_mse_tcl_error = np.arange(len(np.sort(mse_tcl_error)))/float(len(mse_tcl_error))
sorted_y_mse_ekf_error = np.arange(len(np.sort(mse_ekf_error)))/float(len(mse_ekf_error))
sorted_y_mse_cgcl_ekf_error = np.arange(len(np.sort(mse_cgcl_ekf_error)))/float(len(mse_cgcl_ekf_error))
sorted_y_mse_ekf_alone_error = np.arange(len(np.sort(mse_ekf_alone_error)))/float(len(mse_ekf_alone_error))
sorted_y_mse_lgcl_ekf_local_error = np.arange(len(np.sort(mse_lgcl_ekf_local_error)))/float(len(mse_lgcl_ekf_local_error))
sorted_y_mse_ekf_local_error = np.arange(len(np.sort(mse_ekf_local_error)))/float(len(mse_ekf_local_error))

color_index = random.uniform(0, 1)
color_index = random.uniform(0, 1)
color_index = random.uniform(0, 1)
    
plt_12.figure(12)
plt_12.plot(sorted_x_mse_gps_error, sorted_y_mse_gps_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_intact_gps_error, sorted_y_mse_intact_gps_error, '-', c = 'tab:orange', marker = '*',  label="Intact GPS", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_local_lapl_error, sorted_y_mse_local_lapl_error, 'm*-', label="LGCL", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_tcl_error, sorted_y_mse_tcl_error, '-', c = 'lime', marker = '*', label="TCL-MLE", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_ekf_error, sorted_y_mse_ekf_error, 'g*-', label="CGCL->EKF", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_cgcl_ekf_error, sorted_y_mse_cgcl_ekf_error, 'k*-', label="CGCL+EKF", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_ekf_alone_error, sorted_y_mse_ekf_alone_error, 'y*-', label="EKF with GPS only", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_lgcl_ekf_local_error, sorted_y_mse_lgcl_ekf_local_error, 'y*-', label="LGCL+EKF", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_ekf_local_error, sorted_y_mse_ekf_local_error, 'c*-', label="LGCL->EKF", linewidth = 4, markersize = 6)
plt_12.xticks(fontsize=28)
plt_12.yticks(fontsize=28)
plt_12.tick_params(direction='out', length=8)
plt_12.grid(b=True)
plt_12.legend(facecolor='white', fontsize = 27 )
plt_12.xlabel('Localization Mean Square Error [m$^2$]', fontsize = 35)
plt_12.ylabel('CDF', fontsize = 35)
plt_12.show()

sorted_x_max_gps_error = np.sort(max_gps_error)
sorted_x_max_intact_gps_error = np.sort(max_intact_gps_error)
sorted_x_max_lapl_error = np.sort(max_lapl_error)
sorted_x_max_local_lapl_error = np.sort(max_local_lapl_error)
sorted_x_max_tcl_error = np.sort(max_tcl_error)
sorted_x_max_ekf_error = np.sort(max_ekf_error)
sorted_x_max_cgcl_ekf_error = np.sort(max_cgcl_ekf_error)
sorted_x_max_lgcl_ekf_local_error = np.sort(max_lgcl_ekf_local_error)
sorted_x_max_ekf_local_error = np.sort(max_ekf_local_error)

sorted_y_max_gps_error = np.arange(len(np.sort(max_gps_error)))/float(len(max_gps_error))
sorted_y_max_intact_gps_error = np.arange(len(np.sort(max_intact_gps_error)))/float(len(max_intact_gps_error))
sorted_y_max_lapl_error = np.arange(len(np.sort(max_lapl_error)))/float(len(max_lapl_error))
sorted_y_max_local_lapl_error = np.arange(len(np.sort(max_local_lapl_error)))/float(len(max_local_lapl_error))
sorted_y_max_tcl_error = np.arange(len(np.sort(max_tcl_error)))/float(len(max_tcl_error))
sorted_y_max_ekf_error = np.arange(len(np.sort(max_ekf_error)))/float(len(max_ekf_error))
sorted_y_max_cgcl_ekf_error = np.arange(len(np.sort(max_cgcl_ekf_error)))/float(len(max_cgcl_ekf_error))
sorted_y_max_lgcl_ekf_local_error = np.arange(len(np.sort(max_lgcl_ekf_local_error)))/float(len(max_lgcl_ekf_local_error))
sorted_y_max_ekf_local_error = np.arange(len(np.sort(max_ekf_local_error)))/float(len(max_ekf_local_error))

plt_13.figure(13)
plt_13.plot(sorted_x_max_gps_error, sorted_y_max_gps_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_intact_gps_error, sorted_y_max_intact_gps_error, '-', c = 'tab:orange', marker = '*',  label="Intact GPS", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_ekf_error, sorted_y_max_ekf_error, 'g*-', label="CGCL->EKF", linewidth = 4, markersize = 6)
#plt_13.plot(sorted_x_max_cgcl_ekf_error, sorted_y_max_cgcl_ekf_error, 'k*-', label="CGCL+EKF", linewidth = 4, markersize = 6)
#plt_13.plot(sorted_x_max_lgcl_ekf_local_error, sorted_y_max_lgcl_ekf_local_error, 'y*-', label="LGCL+EKF", linewidth = 4, markersize = 6)
#plt_13.plot(sorted_x_max_ekf_local_error, sorted_y_max_ekf_local_error, 'c*-', label="LGCL->EKF", linewidth = 4, markersize = 6)
#plt_13.plot(sorted_x_max_local_lapl_error, sorted_y_max_local_lapl_error, 'm*-', label="LGCL", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_tcl_error, sorted_y_max_tcl_error, '-', c = 'lime', marker = '*', label="TCL-MLE", linewidth = 4, markersize = 6)
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

if (norm(mse_local_lapl_error) < norm(mse_gps_error)):
    print ('MSE reduction with LGCL: ', norm(mse_local_lapl_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with LGCL: ', norm(mse_local_lapl_error- mse_gps_error)/norm(mse_gps_error))
    
if (norm(mse_ekf_error) < norm(mse_gps_error)):
    print ('MSE reduction with CGCL->EKF: ', norm(mse_ekf_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with CGCL->EKF: ', norm(mse_ekf_error- mse_gps_error)/norm(mse_gps_error))

if (norm(mse_ekf_local_error) < norm(mse_gps_error)):
    print ('MSE reduction with LGCL->EKF: ', norm(mse_ekf_local_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with LGCL->EKF: ', norm(mse_ekf_local_error- mse_gps_error)/norm(mse_gps_error))
    
if (norm(mse_cgcl_ekf_error) < norm(mse_gps_error)):
    print ('MSE reduction with CGCL+EKF: ', norm(mse_cgcl_ekf_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with CGCL+EKF: ', norm(mse_cgcl_ekf_error- mse_gps_error)/norm(mse_gps_error))
    
if (norm(mse_lgcl_ekf_local_error) < norm(mse_gps_error)):
    print ('MSE reduction with LGCL+EKF: ', norm(mse_lgcl_ekf_local_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with LGCL+EKF: ', norm(mse_lgcl_ekf_local_error- mse_gps_error)/norm(mse_gps_error))
    
print ('\n')

if (norm(max_lapl_error) < norm(max_gps_error)):
    print ('MAX reduction with CGCL: ', norm(max_lapl_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with CGCL: ', norm(max_lapl_error- max_gps_error)/norm(max_gps_error))
    
if (norm(max_ekf_error) < norm(max_gps_error)):
    print ('MAX reduction with CGCL->EKF: ', norm(max_ekf_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with CGCL->EKF: ', norm(max_ekf_error- max_gps_error)/norm(max_gps_error))   

if (norm(max_ekf_local_error) < norm(max_gps_error)):
    print ('MAX reduction with LGCL->EKF: ', norm(max_ekf_local_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with LGCL->EKF: ', norm(max_ekf_local_error- max_gps_error)/norm(max_gps_error))
    
if (norm(max_cgcl_ekf_error) < norm(max_gps_error)):
    print ('MAX reduction with CGCL+EKF: ', norm(max_cgcl_ekf_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with CGCL+EKF: ', norm(max_cgcl_ekf_error- max_gps_error)/norm(max_gps_error)) 
    
if (norm(max_lgcl_ekf_local_error) < norm(max_gps_error)):
    print ('MAX reduction with LGCL+EKF: ', norm(max_lgcl_ekf_local_error- max_gps_error)/norm(max_gps_error))
    
else:
    print ('MAX increment with LGCL+EKF: ', norm(max_lgcl_ekf_local_error- max_gps_error)/norm(max_gps_error))

    
'''
if (norm(mse_tcl_error) < norm(mse_gps_error)):
    print ('MSE reduction with TCL-MLE: ', norm(mse_tcl_error- mse_gps_error)/norm(mse_gps_error))
    
else:
    print ('MSE increment with TCL-MLE: ', norm(mse_tcl_error- mse_gps_error)/norm(mse_gps_error))

'''
time = random.randint(0, true_X.shape[1]-1)

#time = 65

plt_11.figure(11)
for i in range(A.shape[1]):
    if (Deg[time,i,i] == 0):
        plt_11.plot(true_X[i,time], true_Y[i,time], 'ko', markersize = 10)
        plt_11.annotate(str(i), (true_X[i,time], true_Y[i,time]), fontsize=15)
    else:
        connected_neighbours = []
        connected_neighbours = np.argwhere( (1 == A[time,i,:]) )
        
        for j in connected_neighbours:
            plt_11.plot(true_X[i,time], true_Y[i,time], 'ko', markersize = 10)
            plt_11.plot(np.array([true_X[i,time], true_X[j,time]]), np.array([true_Y[i,time], true_Y[j,time]]), 'r-', linewidth = 3)
            plt_11.annotate(str(i), (true_X[i,time], true_Y[i,time]), fontsize=15)
       
plt_11.xlabel('x-axis', fontsize = 35) 
plt_11.ylabel('y-axis', fontsize = 35)              
plt_11.title('Clusters at time instant t = ' + str(time), fontsize=28) 
plt_11.xticks(fontsize=28)
plt_11.yticks(fontsize=28)
plt_11.show()


vehicle_idx = random.randint(0, true_X.shape[0])
#vehicle_idx = 128

stop = vehicle_idx + 1
time_2 = true_X.shape[1]


plt_1.figure(1)

for i in range(vehicle_idx, stop):
    
    color_index_1 = random.uniform(0, 1)
    color_index_2 = random.uniform(0, 1)
    color_index_3 = random.uniform(0, 1)
        
    color_index_4 = random.uniform(0, 1)
    color_index_5 = random.uniform(0, 1)
    color_index_6 = random.uniform(0, 1)
        
    color_index_7 = random.uniform(0, 1)
    color_index_8 = random.uniform(0, 1)
    color_index_9 = random.uniform(0, 1)
        
    color_index_10 = random.uniform(0, 1)
    color_index_11 = random.uniform(0, 1)
    color_index_12 = random.uniform(0, 1)
    
    color_index_13 = random.uniform(0, 1)
    color_index_14 = random.uniform(0, 1)
    color_index_15 = random.uniform(0, 1)
    for j in range (time_2 - 1):
        plt_1.plot(np.array([true_X[i,j],true_X[i,j+1]]),np.array([true_Y[i,j],true_Y[i,j+1]]), '-', c = (color_index_1, color_index_2, color_index_3), marker = 'o', linewidth = 4, markersize = 6)
        #plt_1.plot(np.array([temp_traj_x[i,j], temp_traj_x[i,j+1]]),np.array([temp_traj_y[i,j], temp_traj_y[i,j+1]]), '-', c = (color_index_4, color_index_5, color_index_6), marker = 'o', linewidth = 4, markersize = 6)
        plt_1.plot(np.array([noise_X[i,j], noise_X[i,j+1]]),np.array([noise_Y[i,j], noise_Y[i,j+1]]), '-', c = (color_index_7, color_index_8, color_index_9), marker = 'o', linewidth = 4, markersize = 6)
        #plt_1.plot(np.array([EKF_recon_X[i,j], EKF_recon_X[i,j+1]]),np.array([EKF_recon_Y[i,j], EKF_recon_Y[i,j+1]]), '-', c = 'tab:orange', marker = 'o', linewidth = 4, markersize = 6)
        plt_1.plot(np.array([Concatenated_CGCL_Points_X[i,j], Concatenated_CGCL_Points_X[i,j+1]]),np.array([Concatenated_CGCL_Points_Y[i,j], Concatenated_CGCL_Points_Y[i,j+1]]), '-', c = (color_index_13, color_index_14, color_index_15), marker = 'o', linewidth = 4, markersize = 6)
        #plt_1.plot(np.array([true_X[i,j],true_X[i,j+1]]),np.array([true_Y[i,j],true_Y[i,j+1]]), 'ro-', linewidth = 4, markersize = 6)
        #plt_1.plot(np.array([temp_x[i,j], temp_x[i,j+1]]),np.array([temp_y[i,j], temp_y[i,j+1]]), 'bo-', linewidth = 4, markersize = 6)
                  
plt_1.xlabel('x-axis', fontsize = 35)
plt_1.ylabel('y-axis', fontsize = 35)
#plt_1.title('True trajectories up to time instant t = ' + str(time_2), fontsize=20)    
plt_1.xticks(fontsize=28)
plt_1.yticks(fontsize=28)
#plt_1.legend(('CARLA', 'KM', 'GPS', 'EKF'))
#plt_1.legend(('CARLA', 'KM', 'EKF'))
#plt_1.legend(('CARLA', 'GPS', 'EKF'))
#plt_1.legend(('CARLA', 'CGCL->EKF', 'CGCL+EKF'))
#plt_1.legend(('CARLA'))
plt_1.tick_params(direction='out', length=8)
plt_1.grid(b=True)
plt_1.show()

'''
plt_2.figure(2)
for i in range(vehicle_idx, vehicle_idx+1):
    if (i != 510000):
        print (i)
    
        color_index_1 = random.uniform(0, 1)
        color_index_2 = random.uniform(0, 1)
        color_index_3 = random.uniform(0, 1)
        
        color_index_4 = random.uniform(0, 1)
        color_index_5 = random.uniform(0, 1)
        color_index_6 = random.uniform(0, 1)
       
        for j in range (time_2 - 1):
            plt_2.plot(np.array([noise_X[i,j], noise_X[i,j+1]]),np.array([noise_Y[i,j], noise_Y[i,j+1]]), '-', c = (color_index_4, color_index_5, color_index_6), marker = 'o', linewidth = 4, markersize = 6)
            #plt_2.plot(np.array([true_X[i,j],true_X[i,j+1]]),np.array([true_Y[i,j],true_Y[i,j+1]]), '-', c = (color_index_1, color_index_2, color_index_3), marker = 'o', linewidth = 4, markersize = 6)
            #plt_2.plot(np.array([temp_x[i,j], temp_x[i,j+1]]),np.array([temp_y[i,j], temp_y[i,j+1]]), '-', c = (color_index_4, color_index_5, color_index_6), marker = 'o', linewidth = 4, markersize = 6)
            
           
        
plt_2.xlabel('x-axis', fontsize = 35)
plt_2.ylabel('y-axis', fontsize = 35)
plt_2.xticks(fontsize=28)
plt_2.yticks(fontsize=28)
#plt_2.legend(('CARLA', 'KM'))
plt_2.tick_params(direction='out', length=8)
plt_2.grid(b=True)
plt_2.show()
'''

plt_6.figure(6)
plt_6.plot(np.sort(gps_error[vehicle_idx, :]), np.arange(len(np.sort(gps_error[vehicle_idx, :])))/float(len(gps_error[vehicle_idx, :])), 'r*-', label="GPS", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(ekf_error[vehicle_idx, :]), np.arange(len(np.sort(ekf_error[vehicle_idx, :])))/float(len(ekf_error[vehicle_idx, :])), 'g*-', label="CGCL->EKF", linewidth = 4, markersize = 6)
#plt_6.plot(np.sort(cgcl_ekf_error[vehicle_idx, :]), np.arange(len(np.sort(cgcl_ekf_error[vehicle_idx, :])))/float(len(cgcl_ekf_error[vehicle_idx, :])), 'k*-', label="CGCL+EKF", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(lapl_error[vehicle_idx, :]), np.arange(len(np.sort(lapl_error[vehicle_idx, :])))/float(len(lapl_error[vehicle_idx, :])), 'b*-', label="CGCL", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(tcl_error[vehicle_idx, :]), np.arange(len(np.sort(tcl_error[vehicle_idx, :])))/float(len(tcl_error[vehicle_idx, :])), '-', c = 'lime', marker = '*', label="TCL-MLE", linewidth = 4, markersize = 6)
#plt_6.plot(np.sort(lgcl_ekf_local_error[vehicle_idx, :]), np.arange(len(np.sort(lgcl_ekf_local_error[vehicle_idx, :])))/float(len(lgcl_ekf_local_error[vehicle_idx, :])), 'y*-', label="LGCL+EKF", linewidth = 4, markersize = 6)
#plt_6.plot(np.sort(ekf_local_error[vehicle_idx, :]), np.arange(len(np.sort(ekf_local_error[vehicle_idx, :])))/float(len(ekf_local_error[vehicle_idx, :])), 'c*-', label="LGCL->EKF", linewidth = 4, markersize = 6)
#plt_6.plot(np.sort(local_lapl_error[vehicle_idx, :]), np.arange(len(np.sort(local_lapl_error[vehicle_idx, :])))/float(len(local_lapl_error[vehicle_idx, :])), 'm*-', label="LGCL", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(intact_gps_error[vehicle_idx, :]), np.arange(len(np.sort(intact_gps_error[vehicle_idx, :])))/float(len(intact_gps_error[vehicle_idx, :])), '-', c = 'tab:orange', marker = '*', label="Intact GPS", linewidth = 4, markersize = 6)
#plt_6.plot(np.sort(ekf_alone_error[vehicle_idx, :]), np.arange(len(np.sort(ekf_alone_error[vehicle_idx, :])))/float(len(ekf_alone_error[vehicle_idx, :])), 'g*-', label="EKF with GPS only", linewidth = 4, markersize = 6)
plt_6.xticks(fontsize=28)
plt_6.yticks(fontsize=28)
plt_6.tick_params(direction='out', length=8)
plt_6.grid(b=True)
plt_6.legend(facecolor='white', fontsize = 27 )
plt_6.xlabel('Localization Error [m]', fontsize = 35)
plt_6.ylabel('CDF', fontsize = 35)
plt_6.title('Vehicle ' + str(vehicle_idx), fontsize = 35)
plt_6.show()

plt_5.figure(5)
plt_5.plot(np.arange(1,time_instances + 1), mse_ekf_error, 'g*-', label="CGCL->EKF", linewidth = 4, markersize = 6)
plt_5.plot(np.arange(1,time_instances + 1), mse_cgcl_ekf_error, 'm*-', label="CGCL+EKF", linewidth = 4, markersize = 6)
plt_5.plot(np.arange(1,time_instances + 1), mse_lapl_error, 'b*-', label="CGCL alone", linewidth = 4, markersize = 6)
plt_5.xticks(fontsize=28)
plt_5.yticks(fontsize=28)
plt_5.tick_params(direction='out', length=8)
plt_5.grid(b=True)
plt_5.legend(facecolor='white', fontsize = 37 )
plt_5.xlabel('Time instances', fontsize = 35)
plt_5.ylabel('LMSE [m$^2$]', fontsize = 35)
plt_5.show()


ekf_error_reduction = np.zeros(true_X.shape[0])

lgcl_ekf_local_error_reduction = np.zeros(true_X.shape[0])

lapl_error_reduction = np.zeros(true_X.shape[0])

lapl_local_error_reduction = np.zeros(true_X.shape[0])

tcl_error_reduction = np.zeros(true_X.shape[0])

for i in range (true_X.shape[0]):
    
    if (norm(ekf_error[i, :]) < norm(intact_gps_error[i, :])):
        ekf_error_reduction[i] = norm(ekf_error[i, :]- intact_gps_error[i, :])/norm(intact_gps_error[i, :])
        
    if (norm(lgcl_ekf_local_error[i, :]) < norm(intact_gps_error[i, :])):
        lgcl_ekf_local_error_reduction[i] = norm(lgcl_ekf_local_error[i, :]- intact_gps_error[i, :])/norm(intact_gps_error[i, :])
        
    if (norm(lapl_error[i, :]) < norm(intact_gps_error[i, :])):
        lapl_error_reduction[i] = norm(lapl_error[i, :]- intact_gps_error[i, :])/norm(intact_gps_error[i, :])
        
    if (norm(local_lapl_error[i, :]) < norm(intact_gps_error[i, :])):
        lapl_local_error_reduction[i] = norm(local_lapl_error[i, :]- intact_gps_error[i, :])/norm(intact_gps_error[i, :])
        
    if (norm(local_lapl_error) < norm(gps_error)):
        tcl_error_reduction[i] = norm(tcl_error[i, :] - intact_gps_error[i, :])/norm(intact_gps_error[i, :])
        
    
print ('\n')
if (norm(ekf_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with CGCL->EKF:', ekf_error_reduction[vehicle_idx])
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with CGCL->EKF', norm(ekf_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))

if (norm(ekf_local_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with LGCL->EKF:', norm(ekf_local_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with LGCL->EKF', norm(ekf_local_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
if (norm(cgcl_ekf_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with CGCL+EKF:', norm(cgcl_ekf_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with CGCL+EKF', norm(cgcl_ekf_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
if (norm(lgcl_ekf_local_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with LGCL+EKF:', lgcl_ekf_local_error_reduction[vehicle_idx])
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with LGCL+EKF', norm(lgcl_ekf_local_error[vehicle_idx, :]- intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))

if (norm(lapl_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with CGCL alone:', lapl_error_reduction[vehicle_idx])
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with CGCL alone', norm(lapl_error[vehicle_idx, :] - intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
if (norm(local_lapl_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with LGCL alone:', lapl_local_error_reduction[vehicle_idx])
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with LGCL alone', norm(local_lapl_error[vehicle_idx, :] - intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
if (norm(tcl_error[vehicle_idx, :]) < norm(intact_gps_error[vehicle_idx, :])):
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error reduction with TCL-MLE alone:', norm(tcl_error[vehicle_idx, :] - intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))
    
else:
    print ('Vehicle ' + str(vehicle_idx), '- Localization Error increment with TCL-MLE alone', norm(tcl_error[vehicle_idx, :] - intact_gps_error[vehicle_idx, :])/norm(intact_gps_error[vehicle_idx, :]))

print ('\n')
   
if (norm(ekf_error) < norm(gps_error)):
    print ('Localization Error reduction with CGCL->EKF:', norm(ekf_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with CGCL->EKF', norm(ekf_error - gps_error)/norm(gps_error))

if (norm(ekf_local_error) < norm(gps_error)):
    print ('Localization Error reduction with LGCL->EKF:', norm(ekf_local_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with LGCL->EKF', norm(ekf_local_error - gps_error)/norm(gps_error))
    
if (norm(cgcl_ekf_error) < norm(gps_error)):
    print ('Localization Error reduction with CGCL+EKF:', norm(cgcl_ekf_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with CGCL+EKF', norm(cgcl_ekf_error - gps_error)/norm(gps_error))
    
if (norm(lgcl_ekf_local_error) < norm(gps_error)):
    print ('Localization Error reduction with LGCL+EKF:', norm(lgcl_ekf_local_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with LGCL+EKF', norm(lgcl_ekf_local_error - gps_error)/norm(gps_error))
    
if (norm(lapl_error) < norm(gps_error)):
    print ('Localization Error reduction with CGCL alone:', norm(lapl_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with CGCL alone', norm(lapl_error - gps_error)/norm(gps_error))
    
if (norm(local_lapl_error) < norm(gps_error)):
    print ('Localization Error reduction with LGCL alone:', norm(local_lapl_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with LGCL alone', norm(local_lapl_error - gps_error)/norm(gps_error))
    
if (norm(tcl_error) < norm(gps_error)):
    print ('Localization Error reduction with TCL-MLE alone:', norm(tcl_error - gps_error)/norm(gps_error))
    
else:
    print ('Localization Error increment with TCL-MLE alone', norm(tcl_error - gps_error)/norm(gps_error))

'''
Low rank modelling -->> Goes to CGCL
#######################################################################################

                if (idx_test_cluster == 2 and idx_time == 257):
                    
                    U, S, Vh = svd(test_L_bar, full_matrices=False)
                   
                    
                u = 0
                for i in clusters[idx_test_cluster][idx_time]:
                    lapl_error[i,idx_time] = test_lapl_error[u]
                    
                    B_X[i,idx_time] = delta_X[u]
                    B_Y[i,idx_time] = delta_Y[u]
                    
                    B_X[i + true_X.shape[0], idx_time] = test_lapl_Points[u,0]
                    B_Y[i + true_X.shape[0], idx_time] = test_lapl_Points[u,1]
                    
                    Concatenated_CGCL_Points_X[i,idx_time] = test_lapl_Points[u,0]
                    Concatenated_CGCL_Points_Y[i,idx_time] = test_lapl_Points[u,1]
                    
                    u += 1

#######################################################################################
'''

