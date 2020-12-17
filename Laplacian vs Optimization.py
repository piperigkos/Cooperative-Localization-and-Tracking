import numpy as np
import pandas as pd
from timeit import default_timer as timer
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

import math
import random
from numpy.linalg import inv,cholesky,norm,matrix_rank,eigvals,svd,pinv
import cvxpy as cvx
from scipy.sparse.linalg import lsmr,svds
from scipy.linalg import orthogonal_procrustes,toeplitz
from scipy.spatial import procrustes

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
        
        print ('into')
        if (y_observer < y_target):
            
            return math.radians(0)
        
        else:
            
            return math.radians(180)
        
    if (y_observer == y_target):
        
        print ('into')
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
    
    flag_1 = 0
    
    '''
    if (x_observer == x_target or y_observer == y_target):
        return 0
    
    if (x_target >= x_observer and y_target >= y_observer):
        
        a = (math.atan((y_target-y_observer)/(x_target-x_observer)))
        flag_1 = 0
        
    elif (x_target >= x_observer and y_target <= y_observer):
        
        a = (math.atan((y_observer-y_target)/(x_target-x_observer)))
        flag_1 = 1
        
    elif (x_target <= x_observer and y_target <= y_observer):
        
        a = (math.atan((y_observer-y_target)/(x_observer-x_target)))
        flag_1 = 2
        
    elif (x_target <= x_observer and y_target >= y_observer):
        
        a = (math.atan((y_target-y_observer)/(x_observer-x_target)))
        flag_1 = 3
    '''
    
    a = (math.atan((y_target-y_observer)/(x_target-x_observer)))
    
    return a, flag_1

def Solve_The_System (Cartesian_Points, L, anchors_index, anchors, delta_X, delta_Y):
    
    
    L_bar = np.zeros((L[:,0].size, anchors[:,0].size))

    L_chol = np.zeros((L_bar[:,0].size, L_bar[0,:].size))
    
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
   
    X = np.zeros((Cartesian_Points[:,0].size))
    Y = np.zeros((Cartesian_Points[:,0].size))

   
    X = lsmr(L_bar,b)[0]
    Y = lsmr(L_bar,q)[0]
    
    Points = np.zeros((X.size, 3))

    Points[:,0] = X
    Points[:,1] = Y

    reconstr_error = np.zeros((X.size))

    for i in range(reconstr_error.size):
        reconstr_error[i] = norm(np.array([Cartesian_Points[i,0] - X[i], Cartesian_Points[i,1] - Y[i]]),2)
        
    return Points, reconstr_error, L_bar    

def Optimization(A,z_d,z_c_x,z_c_y,z_a,a_flag):
   
    y = cvx.Variable(shape = A[:,0].size)
    x = cvx.Variable(shape = A[:,0].size)
                 
    f_1 = 0
    f_2 = 0
    f_3 = 0
            
    for i in range(z_c_x.size):
        f_1 += (z_c_x[i] - x[i])**2 + (z_c_y[i] - y[i])**2
                    
    for i in range(z_d[:,0].size):
        for j in range(z_d[0,:].size):
            if (A[i,j] == 1):
                            
                f_2 += cvx.power(cvx.pos(-z_d[i,j] + (cvx.norm(cvx.vstack( [x[i] - x[j], y[i] - y[j] ] ),2))),2)
                    
                if (a_flag[i,j] == 0):
                     
                    f_3 += (z_a[i,j]*(y[j] - y[i]) - (x[j] - x[i]))**2
                                
                elif (a_flag[i,j] == 1):
                            
                    f_3 += (z_a[i,j]*(x[j] - x[i]) - (y[i] - y[j]))**2
                    
                elif (a_flag[i,j] == 2):
                            
                    f_3 += (z_a[i,j]*(y[i] - y[j]) - (x[i] - x[j]))**2
                    
                else:
                    
                    f_3 += (z_a[i,j]*(x[i] - x[j]) - (y[j] - y[i]))**2
                
                
                    
    opt_prob = cvx.Problem(cvx.Minimize(f_1 + f_2 +f_3))
    opt_prob.solve()
        
    return x.value, y.value


def Optimization_AoA(A,z_d,z_c_x,z_c_y,z_a,deg_noisy_AoA):
   
    y = cvx.Variable(shape = A[:,0].size)
    x = cvx.Variable(shape = A[:,0].size)
          
    low_a = 70
    high_a = 110
    
    low_a_2 = 250
    high_a_2 = 290
    
    f_1 = 0
    f_2 = 0
    f_3 = 0
    

    for i in range(z_c_x.size):
        f_1 += (((z_c_x[i]) - x[i])**2) + (((z_c_y[i]) - y[i])**2)
                    
    for i in range(z_d[:,0].size):
        for j in range(z_d[0,:].size):
            if (A[i,j] == 1):
                            
                f_2 += (cvx.power(cvx.pos(-z_d[i,j] + (cvx.norm(cvx.vstack( [x[i] - x[j], y[i] - y[j] ] ),2))),2))
                    
    
                if (deg_noisy_AoA[i,j] > 0):
                            
                    if ((deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a)): 
                                
                        if ((deg_noisy_AoA[i,j] <= low_a_2 or deg_noisy_AoA[i,j] >= high_a_2)):
                     
                            f_3 += ((z_a[i,j]*(x[j] - x[i]) - (y[j] - y[i]))**2)
                                
                else:
                            
                    if ((deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a)): 
                                
                        if ((deg_noisy_AoA[i,j] >= -low_a_2 or deg_noisy_AoA[i,j] <= -high_a_2)):
                            
                            f_3 += ((z_a[i,j]*(x[j] - x[i]) - (y[j] - y[i]))**2)
                
                    
    opt_prob = cvx.Problem(cvx.Minimize(f_1 + f_2 + f_3 ))
    opt_prob.solve()
        
    return x.value, y.value




def Optimization_AoA_GD (A, noisy_D, tan_noisy_AoA, noise_X, noise_Y, deg_noisy_AoA):
    '''
    delta = 0.001
    num_of_iter_GD = 300
    '''
    
    delta = 0.001
    num_of_iter_GD = 300
    
    low_a = 70
    high_a = 110
    
    low_a_2 = 250
    high_a_2 = 290
    
    
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
                        
                        sum_x_d += 2*((noisy_D[i,j] - norm(np.array([noise_X[i] - noise_X[j], noise_Y[i] - noise_Y[j]]),2))*( (noise_X[i] - noise_X[j]) / (norm(np.array([noise_X[i] - noise_X[j], noise_Y[i] - noise_Y[j]]),2))))
                        
                        sum_y_d += 2*((noisy_D[i,j] - norm(np.array([noise_X[i] - noise_X[j], noise_Y[i] - noise_Y[j]]),2))*( (noise_Y[i] - noise_Y[j]) / (norm(np.array([noise_X[i] - noise_X[j], noise_Y[i] - noise_Y[j]]),2))))
                       
                        
                        if (deg_noisy_AoA[i,j] > 0):
                            
                            if ((deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] <= low_a_2 or deg_noisy_AoA[i,j] >= high_a_2)):
                                
                                    sum_x_a += 2*( (noise_X[j] - noise_X[i])*(tan_noisy_AoA[i,j]**2) - (noise_Y[j] - noise_Y[i])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (noise_Y[j] - noise_Y[i]) - (noise_X[j] - noise_X[i])*tan_noisy_AoA[i,j])
                        else:
                            
                            if ((deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a)): 
                                
                                if ((deg_noisy_AoA[i,j] >= -low_a_2 or deg_noisy_AoA[i,j] <= -high_a_2)):
                                
                                    sum_x_a += 2*( (noise_X[j] - noise_X[i])*(tan_noisy_AoA[i,j]**2) - (noise_Y[j] - noise_Y[i])*tan_noisy_AoA[i,j])
                                    
                                    sum_y_a += 2*( (noise_Y[j] - noise_Y[i]) - (noise_X[j] - noise_X[i])*tan_noisy_AoA[i,j])
                      
                        
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
                
                x_final[i,l] = noise_X[i] + delta*(sum_x_d + sum_x_a)
                
                y_final[i,l] = noise_Y[i] + delta*(sum_y_d + sum_y_a)
                
                
            else:
                
                x_final[i,l] =  x_final[i,l-1] + delta*(sum_x_d + sum_x_a + 2*(noise_X[i] - x_final[i,l-1])) 
                
                y_final[i,l] =  y_final[i,l-1] + delta*(sum_y_d + sum_y_a + 2*(noise_Y[i] - y_final[i,l-1])) 
                
   
    return x_final[:,num_of_iter_GD-1], y_final[:,num_of_iter_GD-1]


def Distributed_Lapl (Deg, A, delta_X, delta_Y, noise_X, noise_Y, noisy_D, noisy_a):

    num_of_iter_distr = 1
    
    x_final = np.zeros((noise_X.size, num_of_iter_distr))
    y_final = np.zeros((noise_X.size, num_of_iter_distr))
    
    distr_delta_X = np.zeros(noise_X.size)
    distr_delta_Y = np.zeros(noise_Y.size)
    
    distr_delta_X = np.copy(delta_X)
    distr_delta_Y = np.copy(delta_Y)
    
    for l in range (num_of_iter_distr):
        
        for k in range (noise_X.size):  
              
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
                    b[i] = noise_X[index[i]]
                    q[i] = noise_Y[index[i]]
                else:
                    b[i] = x_final[index[i],l-1]
                    q[i] = y_final[index[i],l-1]
                
                
            b[i+1] = distr_delta_X[index[i+1]]
            q[i+1] = distr_delta_Y[index[i+1]]  
            
            if (l == 0):
                b[i+2] = noise_X[k]
                q[i+2] = noise_Y[k]
            else:
                b[i+2] = x_final[k,l-1]
                q[i+2] = y_final[k,l-1]
            
            temp_x = np.zeros(index.size   +1)   
            temp_y = np.zeros(index.size   +1)
           
            temp_x = lsmr(L_local,b)[0]
            temp_y = lsmr(L_local,q)[0]
            
            x_final[k,l] = temp_x[index.size-1]
            y_final[k,l] = temp_y[index.size-1]
            
        
    return x_final[:,num_of_iter_distr-1], y_final[:,num_of_iter_distr-1]
            
def ATC(Kmax, L, b, q, A, init_Points, Cartesian_Points, noisy_Cartesian, lapl_Points, Deg, v):
    
    MSD = np.zeros(Kmax)
    
    res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    dif_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    C = np.copy(A)
    
    C_2 = np.copy(A)
    
    #L += (10**(-5))*np.eye(Cartesian_Points.shape[0])
    
    for i in range (C.shape[0]):
        C[i,i] = 1/(Deg[i,i] + 1)
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C[i,j] /= (Deg[i,i] + 1)
                #C[i,j] /= np.max(Deg + 1)
            #if (L[i,j] == 0):
                #L[i,j] = 10**(-9)
        #C[i,i] = 1 - np.sum(C[i,:])
           
    
    
    for i in range (C.shape[0]):
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] /= max(Deg[i,i]+1, Deg[j,j]+1)
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    
    w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))

    w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
          
    if (v == 0):
        
       for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
    else:
        
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
                    
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    w_x[i,j,0] = init_Points[i,j,0]
                    w_y[i,j,0] = init_Points[i,j,1]
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
                    
    psi_x = np.copy(w_x)
    
    psi_y = np.copy(w_y)
    
   
    
    P = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    
    for i in range (Cartesian_Points.shape[0]):
        
        P[i,:,:,0] = 0.15*np.eye((Cartesian_Points.shape[0]))
        
    #lam = 0.8
    
    lam = 0.2
    
    flag_for_psd = np.zeros((Cartesian_Points.shape[0],Kmax))
                            
    
    g_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    g_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    p_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    p_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    cor_L = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    conc_L = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    conc_b = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    conc_q = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    step_a_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    step_a_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    flag_for_converg = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    for i in range (Cartesian_Points.shape[0]):
        
        #u = 2
       
        #u = 1
        
        u = 0
        
        conc_L[i] = np.zeros((1*(int(Deg[i,i])+1), Cartesian_Points.shape[0]))
        
        conc_b[i] = np.zeros(1*(int(Deg[i,i])+1))
        conc_q[i] = np.zeros(1*(int(Deg[i,i])+1))
        
        list_of_indices = []
        
        list_of_indices = list(np.argwhere(1 == A[i,:]))
        
        list_of_indices.append(i)
        
        '''
        conc_L[i][0,:] = L[i,:]
        
        conc_b[i][0] = b[i]
        conc_q[i][0] = q[i]
        
        conc_L[i][int(Deg[i,i])+1,:] = L[i+Cartesian_Points.shape[0],:]
        conc_b[i][int(Deg[i,i])+1] = b[i+Cartesian_Points.shape[0]]
        conc_q[i][int(Deg[i,i])+1] = q[i+Cartesian_Points.shape[0]]
        '''
        #for j in (np.argwhere(1 == A[i,:])):
            
        for j in (np.sort(list_of_indices)):
            
            conc_L[i][u,:] = L[j,:]
            conc_b[i][u] = b[j]
            conc_q[i][u] = q[j]
            '''
            conc_L[i][u+int(Deg[i,i])+1,:] = L[j+Cartesian_Points.shape[0],:]
            conc_b[i][u+int(Deg[i,i])+1] = b[j+Cartesian_Points.shape[0]]
            conc_q[i][u+int(Deg[i,i])+1] = q[j+Cartesian_Points.shape[0]]
            '''
            u += 1
        
        '''
        conc_L[i][(int(Deg[i,i])+1):,:] = np.eye(Cartesian_Points.shape[0])  
        conc_b[i][(int(Deg[i,i])+1):] = b[int(Cartesian_Points.shape[0]):]
        conc_q[i][(int(Deg[i,i])+1):] = q[int(Cartesian_Points.shape[0]):]
        
        if (i == 0):
            print (conc_L[i])
            print ('\n')
            print (conc_b[i])
        '''
   
    
    a = 10**(-7)
    #a = 0
    for i in range (Cartesian_Points.shape[0]):
       
        cor_L[i,:,:,0] = (conc_L[i].T@conc_L[i]) + a*np.eye(Cartesian_Points.shape[0])
        
    for i in range (g_x.shape[0]):
        
        #g_x[i,:,0] = L[i,:]*b[i] - cor_L[i,:,:,0]@w_x[i,:,0]
        #g_y[i,:,0] = L[i,:]*q[i] - cor_L[i,:,:,0]@w_y[i,:,0]
        
        #g_x[i,:,0] = (conc_L[i].T@conc_b[i]) - cor_L[i,:,:,0]@w_x[i,:,0]
        #g_y[i,:,0] = (conc_L[i].T@conc_q[i]) - cor_L[i,:,:,0]@w_y[i,:,0]
        
        g_x[i,:,0] = (conc_L[i].T@conc_b[i]) - cor_L[i,:,:,0]@w_x[i,:,0]
        g_y[i,:,0] = (conc_L[i].T@conc_q[i]) - cor_L[i,:,:,0]@w_y[i,:,0]
        
        p_x[i,:,1] = g_x[i,:,0] 
        p_y[i,:,1] = g_y[i,:,0]     
        
        res[i,0] = norm(np.array([conc_b[i], conc_q[i]]).T - conc_L[i]@np.array([w_x[i,:,0], w_y[i,:,0]]).T)**2
    
    for k in range (Kmax-1):
        
                       
        
        for i in range (Cartesian_Points.shape[0]):
            '''
            P[i,:,:,k] = (lam**(-1))*(P[i,:,:,k])
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_P = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C_2[j,i] != 0):
                    
                    #temp_sum_x += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(b[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_x[i,:,k]))
                    
                    #temp_sum_y += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(q[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_y[i,:,k]))
                    
                    #print ((C_2[j,i]*(P[i,:,:,k]@(L.T[j,:]*(b[j] - L[j,:]@w_x[i,:,k]))))/(1 + L.T[j,:]@P[i,:,:,k]@L[j,:]))
                    
                    temp_sum_x += (C_2[j,i]*(P[i,:,:,k]@(L.T[j,:]*(b[j] - L[j,:]@w_x[i,:,k]))))/(1 + C_2[j,i]*L.T[j,:]@P[i,:,:,k]@L[j,:])
                    
                    temp_sum_y += (C_2[j,i]*(P[i,:,:,k]@(L.T[j,:]*(q[j] - L[j,:]@w_y[i,:,k]))))/(1 + C_2[j,i]*L.T[j,:]@P[i,:,:,k]@L[j,:])
                    
                    temp_sum_P += (((C_2[j,i]*P[i,:,:,k]@np.reshape(L[j,:], [L.shape[0],1])@np.reshape(L[j,:], [1,L.shape[0]])@P[i,:,:,k])/(1 + C_2[j,i]*np.reshape(L[j,:], [1,L.shape[0]])@P[i,:,:,k]@np.reshape(L[j,:], [L.shape[0],1]))))
                    
            psi_x[i,:,k+1] = w_x[i,:,k] + temp_sum_x
            
            psi_y[i,:,k+1] = w_y[i,:,k] + temp_sum_y
            
            P[i,:,:,k+1] = P[i,:,:,k] - temp_sum_P
            '''
            #psi_x[i,:,k+1] = w_x[i,:,k] + 0.2*temp_sum_x
            
            #psi_y[i,:,k+1] = w_y[i,:,k] + 0.2*temp_sum_y
            
            #psi_x[i,:,k+1] = w_x[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(b[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_x[i,:,k]))
            
            #psi_y[i,:,k+1] = w_y[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(q[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_y[i,:,k]))
            
            a_x = (p_x[i,:,k+1].T@g_x[i,:,k]) / (p_x[i,:,k+1].T@cor_L[i,:,:,k]@p_x[i,:,k+1])
            
            psi_x[i,:,k+1] = w_x[i,:,k] + a_x*p_x[i,:,k+1] 
              
            g_x_dot = (lam*g_x[i,:,k] + conc_L[i].T@(conc_b[i] - conc_L[i]@w_x[i,:,k]))
            
            g_x[i,:,k+1] = (g_x_dot - a_x*(cor_L[i,:,:,k]@p_x[i,:,k+1]))
           
            temp_g_x = g_x[i,:,k+1] - g_x[i,:,k]
            
            #b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
            
            if (k % 2 == 0):
                b_x = 10**(-4)
                b_x = 0
            else:
                b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
                
                #b_x = ((g_x[i,:,k+1].T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
                #b_x = ((g_x[i,:,k+1].T@g_x[i,:,k+1]) / (-p_x[i,:,k+1]@temp_g_x))
                #b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (-p_x[i,:,k+1]@temp_g_x))
                #b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
                
            #b_x = 0
            
            if (k+2 <= Kmax-1):
                p_x[i,:,k+2] = (g_x[i,:,k+1] + b_x*p_x[i,:,k+1])
            
            a_y = (p_y[i,:,k+1].T@g_y[i,:,k]) / (p_y[i,:,k+1].T@cor_L[i,:,:,k]@p_y[i,:,k+1])
           
            psi_y[i,:,k+1] = w_y[i,:,k] + a_y*p_y[i,:,k+1]
               
            g_y_dot = (lam*g_y[i,:,k] + conc_L[i].T@(conc_q[i] - conc_L[i]@w_y[i,:,k]))
            
            g_y[i,:,k+1] = (g_y_dot - a_y*(cor_L[i,:,:,k]@p_y[i,:,k+1]))
            
            temp_g_y = g_y[i,:,k+1] - g_y[i,:,k]
            
            #b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
            
            if (k % 2 == 0):
                b_y = 10**(-4)
                b_y = 0
            else:
                b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
                
                #b_y = ((g_y[i,:,k+1].T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
                #b_y = ((g_y[i,:,k+1].T@g_y[i,:,k+1]) / (-p_y[i,:,k+1]@temp_g_y))
                #b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (-p_y[i,:,k+1]@temp_g_y))
                #b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
                
            #b_y = 0
            
            if (k+2 <= Kmax-1):
                p_y[i,:,k+2] = (g_y[i,:,k+1] + b_y*p_y[i,:,k+1])
           
            
            cor_L[i,:,:,k+1] = cor_L[i,:,:,k] 
            
            step_a_x[i,k+1] = b_x
            step_a_y[i,k+1] = b_y
            
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C_2.T[j,i] != 0):
                    
                    temp_sum_x += psi_x[j,:,k+1]*C_2.T[j,i]
                    
                    temp_sum_y += psi_y[j,:,k+1]*C_2.T[j,i]
                    
                
            w_x[i,:,k+1] = np.copy(temp_sum_x) 
            
            w_y[i,:,k+1] = np.copy(temp_sum_y) 
         
    
    '''
    for k in range (Kmax-1):
        
            
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C.T[j,i] != 0):
                    
                    temp_sum_x += w_x[j,:,k]*C.T[j,i]
                    
                    temp_sum_y += w_y[j,:,k]*C.T[j,i]
                
            psi_x[i,:,k] = np.copy(temp_sum_x)
            
            psi_y[i,:,k] = np.copy(temp_sum_y)
        
        for i in range (Cartesian_Points.shape[0]):
            
                
            w_x[i,:,k+1] = psi_x[i,:,k] + 0.1*(L[i,:].T*(b[i] - L[i,:]@psi_x[i,:,k]))
            
            w_y[i,:,k+1] = psi_y[i,:,k] + 0.1*(L[i,:].T*(q[i] - L[i,:]@psi_y[i,:,k]))       
            
    
    
    for k in range (Kmax-1):
        
            
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C.T[j,i] != 0):
                    
                    temp_sum_x += w_x[j,:,k]*C.T[j,i]
                    
                    temp_sum_y += w_y[j,:,k]*C.T[j,i]
                
            psi_x[i,:,k] = np.copy(temp_sum_x)
            
            psi_y[i,:,k] = np.copy(temp_sum_y)
        
           
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
            
                if (C[j,i] != 0):
                    
                    temp_sum_x += C[j,i]*conc_L[j].T@(conc_b[j] - conc_L[j]@psi_x[i,:,k])
                    
                    temp_sum_y += C[j,i]*conc_L[j].T@(conc_q[j] - conc_L[j]@psi_y[i,:,k])
            
            
            w_x[i,:,k+1] = psi_x[i,:,k] + 0.03*conc_L[i].T@(conc_b[i] - conc_L[i]@psi_x[i,:,k])
            
            w_y[i,:,k+1] = psi_y[i,:,k] + 0.03*conc_L[i].T@(conc_q[i] - conc_L[i]@psi_y[i,:,k])
            
            #w_x[i,:,k+1] = psi_x[i,:,k] + 0.03*temp_sum_x
            
            #w_y[i,:,k+1] = psi_y[i,:,k] + 0.03*temp_sum_y
            
    '''
    temp_final_w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    temp_final_w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    flag = 0
    
    new_Kmax = Kmax
    for k in range (Kmax):
        
        temp_sum = 0
        
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum += norm(np.array([Cartesian_Points[:,0] - w_x[i,:,k], Cartesian_Points[:,1] - w_y[i,:,k]]))**2
            
            #res[i,k] = norm(np.array([w_x[i,:,0] - w_x[i,:,k], w_y[i,:,0] - w_y[i,:,k]]))/norm(np.array([w_x[i,:,0], w_y[i,:,0]]))
            
            #res[i,k] = norm(np.array([conc_b[i], conc_q[i]]).T - conc_L[i]@np.array([w_x[i,:,k], w_y[i,:,k]]).T)
            #if (k == 0):
                #dif_res[i,k] = 0
                
            #else:
                #dif_res[i,k] = abs(res[i,k] - res[i,1])
                #dif_res[i,k] = (res[i,k]/res[i,k-1])
                
        MSD[k] = 10*np.log10((((1/Cartesian_Points.shape[0])*temp_sum)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2)))
        
        if (flag == 0):
            
            if (abs(MSD[k-1] - MSD[k]) <= 0.01):
                    
                new_Kmax = k
                
                flag = 1
    
    for i in range (Cartesian_Points.shape[0]):
        temp_final_w_x[:,i] = w_x[i,:,-1]
        temp_final_w_y[:,i] = w_y[i,:,-1]
    
    final_w_x = np.zeros(Cartesian_Points.shape[0])   
    
    final_w_y = np.zeros(Cartesian_Points.shape[0])
      
    final_w_x = np.mean(temp_final_w_x, axis = 1)
    
    final_w_y = np.mean(temp_final_w_y, axis = 1)
    
    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - final_w_x[i], Cartesian_Points[i,1] - final_w_y[i]]))
        
    #return final_w_x, final_w_y, w_x, psi_x, w_y, psi_y, diff_error 
  
    return w_x, w_x[:,:,-1], w_y, w_y[:,:,-1], diff_error, C_2, MSD, cor_L[:,:,:,0], step_a_x, step_a_y, new_Kmax, flag_for_psd, res, dif_res, flag_for_converg
       
def CTA(Kmax, L, b, q, A, init_Points, Cartesian_Points, noisy_Cartesian, lapl_Points, Deg, v):
    
    MSD = np.zeros(Kmax)
    
    C = np.copy(A)
    
    C_2 = np.copy(A)
    
    res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    dif_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    for i in range (C.shape[0]):
        C[i,i] = 1/(Deg[i,i]+1)
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C[i,j] = 1/(Deg[i,i]+1)
                #C[i,j] = 1/(np.max(Deg)+1)
            #if (L[i,j] == 0):
                #L[i,j] = 10**(-6)
        #C[i,i] = 1 - np.sum(C[i,:])
                
    for i in range (C_2.shape[0]):
        for j in range (C_2.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] /= max(Deg[i,i]+1, Deg[j,j]+1)
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    '''
    for i in range (C_2.shape[0]):
        temp_sum = 0
        for j in range (C_2.shape[1]):
            if (A[i,j] == 1):
                temp_sum += (norm(L[j,:])**2)*(Deg[j,j]+1)
        for j in range (C_2.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] = (norm(L[j,:])**2)*(Deg[j,j]+1)/temp_sum
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    '''
    w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))

    w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    if (v == 0):
        
       for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
    else:
        
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
                    
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    w_x[i,j,0] = init_Points[i,j,0]
                    w_y[i,j,0] = init_Points[i,j,1]
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
                    
                    
    psi_x = np.copy(w_x)
    
    psi_y = np.copy(w_y)
    
    temp_w_x = np.copy(w_x)
    
    temp_w_y = np.copy(w_y)
    

    #e_x = b - init_Points[:,0]@L
    #e_y = q - init_Points[:,1]@L
    
    #e_2_x = 0.2*e_x
    #e_2_y = 0.2*e_y
    
    cor_L = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    flag_for_psd = np.zeros((Cartesian_Points.shape[0]))
                            
    
       
    g_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    g_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    p_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    p_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    conc_L = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    conc_b = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    conc_q = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    step_a_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    step_a_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    for i in range (Cartesian_Points.shape[0]):
        
        #u = 2
       
        u = 1
        
        conc_L[i] = np.zeros(((int(Deg[i,i])+1), Cartesian_Points.shape[0]))
        
        conc_b[i] = np.zeros((int(Deg[i,i])+1))
        conc_q[i] = np.zeros((int(Deg[i,i])+1))
        
        
        conc_L[i][0,:] = L[i,:] 
        conc_b[i][0] = b[i]
        conc_q[i][0] = q[i]
        for j in (np.argwhere(1 == A[i,:])):
            
            conc_L[i][u,:] = L[j,:]
            conc_b[i][u] = b[j]
            conc_q[i][u] = q[j]
            
            u += 1

    flag = 0
    
    lam = 0.1
    
    a = 10**(-7)
    for i in range (Cartesian_Points.shape[0]):
       
        cor_L[i,:,:,0] = conc_L[i].T@conc_L[i] + a*np.eye(Cartesian_Points.shape[0])
        
        #print (np.real(2/(np.max(np.sort(np.linalg.eigvals(cor_L[i,:,:,0]))) + np.min(np.sort(np.linalg.eigvals(cor_L[i,:,:,0]))))))
        
        
    for i in range (g_x.shape[0]):
       
        g_x[i,:,0] = conc_L[i].T@conc_b[i] - cor_L[i,:,:,0]@w_x[i,:,0]
        g_y[i,:,0] = conc_L[i].T@conc_q[i] - cor_L[i,:,:,0]@w_y[i,:,0]
        
        p_x[i,:,1] = g_x[i,:,0] 
        p_y[i,:,1] = g_y[i,:,0]
        
    flag_for_converg = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    '''
    for k in range (Kmax-1):
        
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C.T[j,i] != 0 ):
                        
                    temp_sum_x += w_x[j,:,k]*C.T[j,i]
                        
                    temp_sum_y += w_y[j,:,k]*C.T[j,i]
                    
            psi_x[i,:,k] = np.copy(temp_sum_x) 
                
            psi_y[i,:,k] = np.copy(temp_sum_y) 
            
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C_2.T[j,i] != 0):
                   
                    #temp_sum_x += C_2[j,i]*(L[j,:]*(b[j] - L[j,:]@psi_x[i,:,k]))
                        
                    #temp_sum_y += C_2[j,i]*(L[j,:]*(q[j] - L[j,:]@psi_y[i,:,k]))
                    
                    temp_sum_x += C_2.T[j,i]*(conc_L[j].T@(conc_b[j] - conc_L[j]@w_x[i,:,k]))
                        
                    temp_sum_y += C_2.T[j,i]*(conc_L[j].T@(conc_q[j] - conc_L[j]@w_y[i,:,k]))
            
            #w_x[i,:,k+1] = psi_x[i,:,k] + 0.2*temp_sum_x
            
            #w_y[i,:,k+1] = psi_y[i,:,k] + 0.2*temp_sum_y
                
            #w_x[i,:,k+1] = psi_x[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(b[i] - np.reshape(L[i,:], [1,L.shape[0]])@psi_x[i,:,k]))
            
            #w_y[i,:,k+1] = psi_y[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(q[i] - np.reshape(L[i,:], [1,L.shape[0]])@psi_y[i,:,k]))
           
            #(np.real(2/(np.max(np.sort(np.linalg.eigvals(cor_L[i,:,:,0]))) + np.min(np.sort(np.linalg.eigvals(cor_L[i,:,:,0]))))))*
            
            w_x[i,:,k+1] = psi_x[i,:,k] + 0.03*conc_L[i].T@(conc_b[i] - conc_L[i]@psi_x[i,:,k])
            
            w_y[i,:,k+1] = psi_y[i,:,k] + 0.03*conc_L[i].T@(conc_q[i] - conc_L[i]@psi_y[i,:,k])
            
            #w_x[i,:,k+1] = psi_x[i,:,k] + 0.01*temp_sum_x
            
            #w_y[i,:,k+1] = psi_y[i,:,k] + 0.01*temp_sum_y
            
            if ((abs(b[i] - psi_x[i,:,k]@L[i,:])) <= e_x[i]):
                
                z_x = psi_x[i,:,k]
                
            elif ((b[i] - psi_x[i,:,k]@L[i,:]) > e_x[i]):
                    
                z_x = psi_x[i,:,k] + ((b[i] - psi_x[i,:,k]@L[i,:] - e_x[i])/(norm(L[i,:])**2))*L[i,:]
                
            elif ((b[i] - psi_x[i,:,k]@L[i,:]) < -e_x[i]):
                
                z_x = psi_x[i,:,k] + ((b[i] - psi_x[i,:,k]@L[i,:] + e_x[i])/(norm(L[i,:])**2))*L[i,:]
                
            if ((abs(q[i] - psi_y[i,:,k]@L[i,:])) <= e_y[i]):
                
                z_y = psi_y[i,:,k]
                
            elif ((q[i] - psi_y[i,:,k]@L[i,:]) > e_y[i]):
                    
                z_y = psi_y[i,:,k] + ((q[i] - psi_y[i,:,k]@L[i,:] - e_y[i])/(norm(L[i,:])**2))*L[i,:]
                
            elif ((q[i] - psi_y[i,:,k]@L[i,:]) < -e_y[i]):
                
                z_y = psi_y[i,:,k] + ((q[i] - psi_y[i,:,k]@L[i,:] + e_y[i])/(norm(L[i,:])**2))*L[i,:]
            
           
            if ((abs(b[i] - psi_x[i,:,k]@L[i,:])) <= e_2_x[i]):
                
                temp_x = z_x
                
            elif ((b[i] - z_x@L[i,:]) > e_2_x[i]):
                    
                temp_x = z_x + ((b[i] - z_x@L[i,:] - e_2_x[i])/(norm(L[i,:])**2))*L[i,:]
                
            elif ((b[i] - z_x@L[i,:]) < -e_2_x[i]):
                
                temp_x = z_x + ((b[i] - z_x@L[i,:] + e_2_x[i])/(norm(L[i,:])**2))*L[i,:]
                
            if ((abs(q[i] - z_y@L[i,:])) <= e_2_x[i]):
                
                temp_y = z_y
                
            elif ((q[i] - z_y@L[i,:]) > e_2_y[i]):
                    
                temp_y = z_y + ((q[i] - z_y@L[i,:] - e_2_y[i])/(norm(L[i,:])**2))*L[i,:]
                
            elif ((q[i] - z_y@L[i,:]) < -e_2_y[i]):
                
                temp_y = z_y + ((q[i] - z_y@L[i,:] + e_2_y[i])/(norm(L[i,:])**2))*L[i,:]
                
            w_x[i,:,k+1] = z_x + a*(temp_x - z_x)
            
            w_y[i,:,k+1] = z_y + a*(temp_y - z_y)
            
            a_x = (p_x[i,:,k+1].T@g_x[i,:,k]) / (p_x[i,:,k+1].T@cor_L[i,:,:,k]@p_x[i,:,k+1])
           
            w_x[i,:,k+1] = psi_x[i,:,k] + a_x*p_x[i,:,k+1]
            
            g_x_dot = (lam*g_x[i,:,k] + conc_L[i].T@(conc_b[i] - conc_L[i]@psi_x[i,:,k]))
        
            g_x[i,:,k+1] = (g_x_dot - a_x*(cor_L[i,:,:,k]@p_x[i,:,k+1]))
            
            temp_g_x = g_x[i,:,k+1] - g_x[i,:,k]
            
            b_x = (temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k])
            
            if (k % 2 == 0):
                b_x = 0.0
            else:
                b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
                
            #b_x = 0
            
            if (k+2 <= Kmax-1):
                p_x[i,:,k+2] = (g_x[i,:,k+1] + b_x*p_x[i,:,k+1])
            
            a_y = (p_y[i,:,k+1].T@g_y[i,:,k]) / (p_y[i,:,k+1].T@cor_L[i,:,:,k]@p_y[i,:,k+1])
            
            w_y[i,:,k+1] = psi_y[i,:,k] + a_y*p_y[i,:,k+1]
            
            g_y_dot = (lam*g_y[i,:,k] + conc_L[i].T@(conc_q[i] - conc_L[i]@psi_y[i,:,k]))
            
            g_y[i,:,k+1] = (g_y_dot - a_y*(cor_L[i,:,:,k]@p_y[i,:,k+1]))
            
            temp_g_y = g_y[i,:,k+1] - g_y[i,:,k]
            
            b_y = (temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k])
            
            if (k % 2 == 0):
                b_y = 0.0
            else:
                b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
            
            #b_y = 0
            
            if (k+2 <= Kmax-1):
                p_y[i,:,k+2] = (g_y[i,:,k+1] + b_y*p_y[i,:,k+1])
            
            step_a_x[i,k+1] = b_x
            step_a_y[i,k+1] = b_y
            
            cor_L[i,:,:,k+1] = cor_L[i,:,:,k]
    '''
      
      
    for k in range (Kmax-1):
        
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            
            for j in range (Cartesian_Points.shape[0]):
                    
                if (C_2[j,i] != 0):
                        
                    temp_sum_x += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(b[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_x[i,:,k]))
                        
                    temp_sum_y += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(q[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_y[i,:,k]))
                    
            #psi_x[i,:,k+1] = w_x[i,:,k] + 0.2*temp_sum_x
                
            #psi_y[i,:,k+1] = w_y[i,:,k] + 0.2*temp_sum_y
        
            psi_x[i,:,k+1] = w_x[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(b[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_x[i,:,k]))
                
            psi_y[i,:,k+1] = w_y[i,:,k] + 0.1*(np.reshape(L[i,:], [L.shape[0],1])@(q[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_y[i,:,k]))
          
        for i in range (Cartesian_Points.shape[0]):
           
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
                
            for j in range (Cartesian_Points.shape[0]):
                    
                if (C_2.T[j,i] != 0):
                        
                    temp_sum_x += psi_x[j,:,k+1]*C_2.T[j,i]
                        
                    temp_sum_y += psi_y[j,:,k+1]*C_2.T[j,i]
                        
            w_x[i,:,k+1] = np.copy(temp_sum_x) 
                
            w_y[i,:,k+1] = np.copy(temp_sum_y)
                 
    
    temp_final_w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    temp_final_w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    new_Kmax = Kmax
     
    for k in range (Kmax):
        
        temp_sum = 0
        
        for i in range (Cartesian_Points.shape[0]):
            
            #res[i,k] = norm(np.array([w_x[i,:,0] - w_x[i,:,k], w_y[i,:,0] - w_y[i,:,k]]))/norm(np.array([w_x[i,:,0], w_y[i,:,0]]))
             
            if (k == 0):
                dif_res[i,k] = 0
                
            else:
                dif_res[i,k] = res[i,k] - res[i,1]
                
            temp_sum += norm(np.array([Cartesian_Points[:,0] - w_x[i,:,k], Cartesian_Points[:,1] - w_y[i,:,k]]))**2
            
        MSD[k] = 10*np.log10((((1/Cartesian_Points.shape[0])*temp_sum)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2)))
        
        if (flag == 0):
            
            if (abs(MSD[k-1] - MSD[k]) <= 0.01):
            #if (abs(res[k-1] - res[k]) <= 0.01):      
                new_Kmax = k
                
                flag = 1
            
    for i in range (Cartesian_Points.shape[0]):
        temp_final_w_x[:,i] = w_x[i,:,-1]
        temp_final_w_y[:,i] = w_y[i,:,-1]
       
    final_w_x = np.zeros(Cartesian_Points.shape[0])   
    
    final_w_y = np.zeros(Cartesian_Points.shape[0])
      
    final_w_x = np.mean(temp_final_w_x, axis = 1)
    
    final_w_y = np.mean(temp_final_w_y, axis = 1)
    
    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - final_w_x[i], Cartesian_Points[i,1] - final_w_y[i]]))
        
        #diff_error[i] = norm(np.array([Cartesian_Points[i,0] - temp_final_w_x[i,i], Cartesian_Points[i,1] - temp_final_w_y[i,i]]))
                
    #return final_w_x, final_w_y, w_x, psi_x, w_y, psi_y, diff_error 

    return cor_L[:,:,:,0], w_x[:,:,-1], w_y, w_y[:,:,-1], diff_error, C_2, MSD, new_Kmax, step_a_x, step_a_y, res, dif_res

def ATCCG(Kmax, L, b, q, A, init_Points, Cartesian_Points, noisy_Cartesian, lapl_Points, Deg, v):
    
    MSD = np.zeros(Kmax)
    
    res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    dif_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    C = np.copy(A)
    
    C_2 = np.copy(A)
    
    for i in range (C.shape[0]):
        C[i,i] = 1/(Deg[i,i] + 1)
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C[i,j] /= (Deg[i,i] + 1)
                
    
    for i in range (C.shape[0]):
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] /= max(Deg[i,i]+1, Deg[j,j]+1)
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    
    w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))

    w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    if (v == 0):
        
       for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
    else:
                    
        
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
            
            count = 0
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                   
                    if ((1 - ( (Cartesian_Points.shape[0]-1-Deg[i,i])/(Cartesian_Points.shape[0]-1) )) < 0.8):  
                        
                        w_x[i,j,0] = init_Points[i,j,0]
                        w_y[i,j,0] = init_Points[i,j,1]
                        
                    else:
                        
                        if (round(0.3*Deg[i,i]) == 0):
                            
                            w_x[i,j,0] = init_Points[i,j,0]
                            w_y[i,j,0] = init_Points[i,j,1]
                        
                        else:
                            if (count <= round(0.3*Deg[i,i])):
                                
                                w_x[i,j,0] = init_Points[i,j,0]
                                w_y[i,j,0] = init_Points[i,j,1]
                                
                            else:
                                
                                w_x[i,j,0] = noisy_Cartesian[j,0]
                                w_y[i,j,0] = noisy_Cartesian[j,1]
                        
                            count += 1
                        
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
                   
          
         
    psi_x = np.copy(w_x)
    
    psi_y = np.copy(w_y)
    
    lam = 0.2
   
    g_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    g_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    p_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    p_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    cor_L = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
    
    conc_L = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    conc_b = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    conc_q = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    for i in range (Cartesian_Points.shape[0]):
        
        u = 0
        
        conc_L[i] = np.zeros((1*(int(Deg[i,i])+1), Cartesian_Points.shape[0]))
        
        conc_b[i] = np.zeros(1*(int(Deg[i,i])+1))
        conc_q[i] = np.zeros(1*(int(Deg[i,i])+1))
        
        list_of_indices = []
        
        list_of_indices = list(np.argwhere(1 == A[i,:]))
        
        list_of_indices.append(i)
       
        for j in (np.sort(list_of_indices)):
            
            conc_L[i][u,:] = L[j,:] 
            conc_b[i][u] = b[j]
            conc_q[i][u] = q[j]
            
            #conc_L[i][u+int(Deg[i,i])+1,:] = L_bar[j+Cartesian_Points.shape[0],:] 
            #conc_b[i][u+int(Deg[i,i])+1] = b[j+Cartesian_Points.shape[0]]
            #conc_q[i][u+int(Deg[i,i])+1] = q[j+Cartesian_Points.shape[0]]
           
            u += 1
            
      
    a = 10**(-7)
   
    for i in range (Cartesian_Points.shape[0]):
       
        cor_L[i,:,:,0] = (conc_L[i].T@conc_L[i]) + a*np.eye(Cartesian_Points.shape[0])
        
    for i in range (g_x.shape[0]):
        
        g_x[i,:,0] = (conc_L[i].T@conc_b[i]) - cor_L[i,:,:,0]@w_x[i,:,0]
        g_y[i,:,0] = (conc_L[i].T@conc_q[i]) - cor_L[i,:,:,0]@w_y[i,:,0]
        
        p_x[i,:,1] = g_x[i,:,0] 
        p_y[i,:,1] = g_y[i,:,0]     
        
        res[i,0] = norm(np.array([conc_b[i], conc_q[i]]).T - conc_L[i]@np.array([w_x[i,:,0], w_y[i,:,0]]).T)**2
    
    for k in range (Kmax-1):
        
                       
        
        for i in range (Cartesian_Points.shape[0]):
            
            a_x = (p_x[i,:,k+1].T@g_x[i,:,k]) / (p_x[i,:,k+1].T@cor_L[i,:,:,k]@p_x[i,:,k+1])
            
            psi_x[i,:,k+1] = w_x[i,:,k] + a_x*p_x[i,:,k+1] 
              
            g_x_dot = (lam*g_x[i,:,k] + conc_L[i].T@(conc_b[i] - conc_L[i]@w_x[i,:,k]))
            
            g_x[i,:,k+1] = (g_x_dot - a_x*(cor_L[i,:,:,k]@p_x[i,:,k+1]))
           
            temp_g_x = g_x[i,:,k+1] - g_x[i,:,k]
            
            #b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
            
            if (k % 2 == 0):
                b_x = 10**(-4)
                b_x = 0
            else:
                b_x = ((temp_g_x.T@g_x[i,:,k+1]) / (g_x[i,:,k].T@g_x[i,:,k]))
               
            if (k+2 <= Kmax-1):
                p_x[i,:,k+2] = (g_x[i,:,k+1] + b_x*p_x[i,:,k+1])
            
            a_y = (p_y[i,:,k+1].T@g_y[i,:,k]) / (p_y[i,:,k+1].T@cor_L[i,:,:,k]@p_y[i,:,k+1])
           
            psi_y[i,:,k+1] = w_y[i,:,k] + a_y*p_y[i,:,k+1]
               
            g_y_dot = (lam*g_y[i,:,k] + conc_L[i].T@(conc_q[i] - conc_L[i]@w_y[i,:,k]))
            
            g_y[i,:,k+1] = (g_y_dot - a_y*(cor_L[i,:,:,k]@p_y[i,:,k+1]))
            
            temp_g_y = g_y[i,:,k+1] - g_y[i,:,k]
            
            #b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
            
            if (k % 2 == 0):
                b_y = 10**(-4)
                b_y = 0
            else:
                b_y = ((temp_g_y.T@g_y[i,:,k+1]) / (g_y[i,:,k].T@g_y[i,:,k]))
                
               
            if (k+2 <= Kmax-1):
                p_y[i,:,k+2] = (g_y[i,:,k+1] + b_y*p_y[i,:,k+1])
           
            
            cor_L[i,:,:,k+1] = cor_L[i,:,:,k] 
            
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (C_2.T[j,i] != 0):
                    
                    temp_sum_x += psi_x[j,:,k+1]*C_2.T[j,i]
                    
                    temp_sum_y += psi_y[j,:,k+1]*C_2.T[j,i]
                    
                
            w_x[i,:,k+1] = np.copy(temp_sum_x) 
            
            w_y[i,:,k+1] = np.copy(temp_sum_y) 
         
    
    temp_final_w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    temp_final_w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    flag = 0
    
    new_Kmax = Kmax
    for k in range (Kmax):
        
        temp_sum = 0
        
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum += norm(np.array([Cartesian_Points[:,0] - w_x[i,:,k], Cartesian_Points[:,1] - w_y[i,:,k]]))**2
            
            #res[i,k] = norm(np.array([w_x[i,:,0] - w_x[i,:,k], w_y[i,:,0] - w_y[i,:,k]]))/norm(np.array([w_x[i,:,0], w_y[i,:,0]]))
            
            #res[i,k] = norm(np.array([conc_b[i], conc_q[i]]).T - conc_L[i]@np.array([w_x[i,:,k], w_y[i,:,k]]).T)
            #if (k == 0):
                #dif_res[i,k] = 0
                
            #else:
                #dif_res[i,k] = abs(res[i,k] - res[i,1])
                #dif_res[i,k] = (res[i,k]/res[i,k-1])
                
        MSD[k] = 10*np.log10((((1/Cartesian_Points.shape[0])*temp_sum)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2)))
        
        if (flag == 0):
            
            if (abs(MSD[k-1] - MSD[k]) <= 0.01):
                    
                new_Kmax = k
                
                flag = 1
    
    for i in range (Cartesian_Points.shape[0]):
        temp_final_w_x[:,i] = w_x[i,:,-1]
        temp_final_w_y[:,i] = w_y[i,:,-1]
    
    final_w_x = np.zeros(Cartesian_Points.shape[0])   
    
    final_w_y = np.zeros(Cartesian_Points.shape[0])
      
    final_w_x = np.mean(temp_final_w_x, axis = 1)
    
    final_w_y = np.mean(temp_final_w_y, axis = 1)
    
    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - final_w_x[i], Cartesian_Points[i,1] - final_w_y[i]]))
        
    #return final_w_x, final_w_y, w_x, psi_x, w_y, psi_y, diff_error 
  
    return w_x, w_x[:,:,-1], w_y, w_y[:,:,-1], diff_error, C_2, MSD, new_Kmax, res, dif_res

def LMSATC(Kmax, L, b, q, A, init_Points, Cartesian_Points, noisy_Cartesian, lapl_Points, Deg, v):
    
    MSD = np.zeros(Kmax)
    
    C = np.copy(A)
    
    C_2 = np.copy(A)
    
    res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    dif_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    for i in range (C.shape[0]):
        C[i,i] = 1/(Deg[i,i]+1)
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C[i,j] = 1/(Deg[i,i]+1)
                
                
    for i in range (C_2.shape[0]):
        for j in range (C_2.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] /= max(Deg[i,i]+1, Deg[j,j]+1)
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    
    w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))

    w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    if (v == 0):
        
       for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
    else:
        '''
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
                    
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    w_x[i,j,0] = init_Points[i,j,0]
                    w_y[i,j,0] = init_Points[i,j,1]
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
        '''
        '''           
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
        '''    
        
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
            
            count = 0
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    if ((1 - ( (Cartesian_Points.shape[0]-1-Deg[i,i])/(Cartesian_Points.shape[0]-1) )) < 0.8):  
                        
                        w_x[i,j,0] = init_Points[i,j,0]
                        w_y[i,j,0] = init_Points[i,j,1]
                        
                    else:
                       
                        if (round(0.3*Deg[i,i]) == 0):
                            
                            w_x[i,j,0] = init_Points[i,j,0]
                            w_y[i,j,0] = init_Points[i,j,1]
                        
                        else:
                            if (count <= round(0.3*Deg[i,i])):
                                
                                w_x[i,j,0] = init_Points[i,j,0]
                                w_y[i,j,0] = init_Points[i,j,1]
                                
                            else:
                                
                                w_x[i,j,0] = noisy_Cartesian[j,0]
                                w_y[i,j,0] = noisy_Cartesian[j,1]
                        
                            count += 1
                        
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
                    
        
    conc_L = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    conc_b = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    conc_q = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    '''
    for i in range (Cartesian_Points.shape[0]):
        
        conc_L[i] = np.zeros((2, Cartesian_Points.shape[0]))
        
        conc_b[i] = np.zeros(2)
        conc_q[i] = np.zeros(2)
        
        
        conc_L[i][0,:] = np.copy(L_bar[i,:])
        
        conc_b[i][0] = b[i]
        conc_q[i][0] = q[i]
        
        conc_L[i][1,:] = np.copy(L_bar[i+Cartesian_Points.shape[0],:])
        conc_b[i][1] = b[i+Cartesian_Points.shape[0]]
        conc_q[i][1] = q[i+Cartesian_Points.shape[0]]
    '''    
        
    for i in range (Cartesian_Points.shape[0]):
        
        temp_L = np.reshape(L[i,:], (Cartesian_Points.shape[0],1))@np.reshape(L[i,:], (1,Cartesian_Points.shape[0]))
        
        #print (np.real(2/((np.max(np.linalg.eigvals(temp_L))))))
        
    psi_x = np.copy(w_x)
    
    psi_y = np.copy(w_y)
    
    Kmax_2 = int(0.75*Kmax)
  
    for k in range (Kmax-1):
            
        for i in range (Cartesian_Points.shape[0]):
                
            temp_L = np.reshape(L[i,:], (Cartesian_Points.shape[0],1))@np.reshape(L[i,:], (1,Cartesian_Points.shape[0]))
            
            fac = np.real(2/((np.max(np.linalg.eigvals(temp_L)))))
            
            psi_x[i,:,k+1] = w_x[i,:,k] + min(fac,0.1)*(np.reshape(L[i,:], [L.shape[0],1])@(b[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_x[i,:,k]))
                    
            psi_y[i,:,k+1] = w_y[i,:,k] + min(fac,0.1)*(np.reshape(L[i,:], [L.shape[0],1])@(q[i] - np.reshape(L[i,:], [1,L.shape[0]])@w_y[i,:,k]))
                
            #psi_x[i,:,k+1] = w_x[i,:,k] + 0.1*(conc_L[i].T@(conc_b[i] - conc_L[i]@w_x[i,:,k]))
                    
            #psi_y[i,:,k+1] = w_y[i,:,k] + 0.1*(conc_L[i].T@(conc_q[i] - conc_L[i]@w_y[i,:,k]))
            
        for i in range (Cartesian_Points.shape[0]):
               
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                    
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
                    
            for j in range (Cartesian_Points.shape[0]):
                        
                if (C_2.T[j,i] != 0):
                            
                    temp_sum_x += psi_x[j,:,k+1]*C_2.T[j,i]
                            
                    temp_sum_y += psi_y[j,:,k+1]*C_2.T[j,i]
                            
            w_x[i,:,k+1] = np.copy(temp_sum_x) 
                    
            w_y[i,:,k+1] = np.copy(temp_sum_y)
    
    temp_final_w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    temp_final_w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    new_Kmax = Kmax
     
    flag = 0
    for k in range (Kmax):
        
        temp_sum = 0
        
        for i in range (Cartesian_Points.shape[0]):
            
            #res[i,k] = norm(np.array([w_x[i,:,0] - w_x[i,:,k], w_y[i,:,0] - w_y[i,:,k]]))/norm(np.array([w_x[i,:,0], w_y[i,:,0]]))
             
            if (k == 0):
                dif_res[i,k] = 0
                
            else:
                dif_res[i,k] = res[i,k] - res[i,1]
                
            temp_sum += norm(np.array([Cartesian_Points[:,0] - w_x[i,:,k], Cartesian_Points[:,1] - w_y[i,:,k]]))**2
            
        MSD[k] = 10*np.log10((((1/Cartesian_Points.shape[0])*temp_sum)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2)))
        
        if (flag == 0):
            
            if (abs(MSD[k-1] - MSD[k]) <= 0.01):
            #if (abs(res[k-1] - res[k]) <= 0.01):      
                new_Kmax = k
                
                flag = 1
            
    for i in range (Cartesian_Points.shape[0]):
        temp_final_w_x[:,i] = w_x[i,:,-1]
        temp_final_w_y[:,i] = w_y[i,:,-1]
       
    final_w_x = np.zeros(Cartesian_Points.shape[0])   
    
    final_w_y = np.zeros(Cartesian_Points.shape[0])
      
    final_w_x = np.mean(temp_final_w_x, axis = 1)
    
    final_w_y = np.mean(temp_final_w_y, axis = 1)
    
    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - final_w_x[i], Cartesian_Points[i,1] - final_w_y[i]]))
        
    return w_x, w_x[:,:,-1], w_y, w_y[:,:,-1], diff_error, C_2, MSD, new_Kmax, res, dif_res

def ATCME(Kmax, L, b, q, A, init_Points, Cartesian_Points, noisy_Cartesian, lapl_Points, Deg, v):
    
    MSD = np.zeros(Kmax)
    
    C = np.copy(A)
    
    C_2 = np.copy(A)
    
    res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    dif_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    flag_for_res = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    for i in range (C.shape[0]):
        C[i,i] = 1/(Deg[i,i]+1)
        for j in range (C.shape[1]):
            if (A[i,j] == 1):
                C[i,j] = 1/(Deg[i,i]+1)
                
                
    for i in range (C_2.shape[0]):
        for j in range (C_2.shape[1]):
            if (A[i,j] == 1):
                C_2[i,j] /= max(Deg[i,i]+1, Deg[j,j]+1)
                
    for i in range (C.shape[0]):
        C_2[i,i] = 1 - np.sum(C_2[i,:])
    
    w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))

    w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    if (v == 0):
        
       for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
        
    else:
        '''
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,:,0] = init_Points[i,:,0]
            w_y[i,:,0] = init_Points[i,:,1]
            
        '''
        '''
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
                    
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    w_x[i,j,0] = init_Points[i,j,0]
                    w_y[i,j,0] = init_Points[i,j,1]
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]
        ''' 
                  
        for i in range (Cartesian_Points.shape[0]):
            
            w_x[i,i,0] = init_Points[i,i,0]
            w_y[i,i,0] = init_Points[i,i,1]
            
            count = 0
            
            for j in range (Cartesian_Points.shape[0]):
                
                if (A[i,j] == 1):
                    
                    if ((1 - ( (Cartesian_Points.shape[0]-1-Deg[i,i])/(Cartesian_Points.shape[0]-1) )) < 0.8):  
                        
                        w_x[i,j,0] = init_Points[i,j,0]
                        w_y[i,j,0] = init_Points[i,j,1]
                        
                    else:
                       
                        if (round(0.3*Deg[i,i]) == 0):
                            
                            w_x[i,j,0] = init_Points[i,j,0]
                            w_y[i,j,0] = init_Points[i,j,1]
                        
                        else:
                            if (count <= round(0.3*Deg[i,i])):
                                
                                w_x[i,j,0] = init_Points[i,j,0]
                                w_y[i,j,0] = init_Points[i,j,1]
                                
                            else:
                                
                                w_x[i,j,0] = noisy_Cartesian[j,0]
                                w_y[i,j,0] = noisy_Cartesian[j,1]
                        
                            count += 1
                        
                    
                else:
                    w_x[i,j,0] = noisy_Cartesian[j,0]
                    w_y[i,j,0] = noisy_Cartesian[j,1]       
           
           
    conc_L = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    
    conc_b = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    conc_q = np.zeros((Cartesian_Points.shape[0]), dtype=np.object)
    '''
    for i in range (Cartesian_Points.shape[0]):
        
        conc_L[i] = np.zeros((2, Cartesian_Points.shape[0]))
        
        conc_b[i] = np.zeros(2)
        conc_q[i] = np.zeros(2)
        
        
        conc_L[i][0,:] = np.copy(L_bar[i,:])
        
        conc_b[i][0] = b[i]
        conc_q[i][0] = q[i]
        
        conc_L[i][1,:] = np.copy(L_bar[i+Cartesian_Points.shape[0],:])
        conc_b[i][1] = b[i+Cartesian_Points.shape[0]]
        conc_q[i][1] = q[i+Cartesian_Points.shape[0]]
    '''   
    psi_x = np.copy(w_x)
    
    psi_y = np.copy(w_y)
    
    Kmax_2 = int(0.75*Kmax)
    
    for k in range (Kmax-1):
           
        for i in range (Cartesian_Points.shape[0]):
                        
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
                       
            temp_L = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
            
            for j in range (Cartesian_Points.shape[0]):
               
                if (C_2[j,i] != 0):
                    
                    #temp_L += L[j,:]
                    
                    temp_L += C_2[j,i]*np.reshape(L[j,:], (L.shape[0],1))@np.reshape(L[j,:], (1,L.shape[0]))
                    
                    temp_sum_x += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(b[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_x[i,:,k]))
                                    
                    temp_sum_y += C_2[j,i]*(np.reshape(L[j,:], [L.shape[0],1])@(q[j] - np.reshape(L[j,:], [1,L.shape[0]])@w_y[i,:,k]))
                                
                    #temp_sum_x += C_2[j,i]*(conc_L[j].T@(conc_b[j] - conc_L[j]@w_x[i,:,k]))
                                    
                    #temp_sum_y += C_2[j,i]*(conc_L[j].T@(conc_q[j] - conc_L[j]@w_y[i,:,k]))
             
            #cor_temp_L = np.reshape(temp_L, (L.shape[0],1))@np.reshape(temp_L, (1,L.shape[0]))
                
            fac = np.real(2/np.max(np.linalg.eigvals(temp_L)))
            
            psi_x[i,:,k+1] = w_x[i,:,k] + min(fac,0.2)*temp_sum_x
                            
            psi_y[i,:,k+1] = w_y[i,:,k] + min(fac,0.2)*temp_sum_y

            
            #print (np.real(2/(np.max(np.linalg.eigvals(temp_L)) + np.min(np.linalg.eigvals(temp_L)))))
                      
        for i in range (Cartesian_Points.shape[0]):
                       
            temp_sum_x = np.zeros(Cartesian_Points.shape[0])
                            
            temp_sum_y = np.zeros(Cartesian_Points.shape[0])
                            
            for j in range (Cartesian_Points.shape[0]):
                                
                if (C_2.T[j,i] != 0):
                                    
                    temp_sum_x += psi_x[j,:,k+1]*C_2.T[j,i]
                                    
                    temp_sum_y += psi_y[j,:,k+1]*C_2.T[j,i]
                                    
            w_x[i,:,k+1] = np.copy(temp_sum_x) 
                            
            w_y[i,:,k+1] = np.copy(temp_sum_y)
    
    temp_final_w_x = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    temp_final_w_y = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0]))
    
    new_Kmax = Kmax
     
    flag = 0
    for k in range (Kmax):
        
        temp_sum = 0
        
        for i in range (Cartesian_Points.shape[0]):
            
            temp_sum += norm(np.array([Cartesian_Points[:,0] - w_x[i,:,k], Cartesian_Points[:,1] - w_y[i,:,k]]))**2
            
        MSD[k] = 10*np.log10((((1/Cartesian_Points.shape[0])*temp_sum)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2)))
        
        if (flag == 0):
            
            if (abs(MSD[k-1] - MSD[k]) <= 0.01):
            #if (abs(res[k-1] - res[k]) <= 0.01):      
                new_Kmax = k
                
                flag = 1
            
    for i in range (Cartesian_Points.shape[0]):
        temp_final_w_x[:,i] = w_x[i,:,-1]
        temp_final_w_y[:,i] = w_y[i,:,-1]
       
    final_w_x = np.zeros(Cartesian_Points.shape[0])   
    
    final_w_y = np.zeros(Cartesian_Points.shape[0])
      
    final_w_x = np.mean(temp_final_w_x, axis = 1)
    
    final_w_y = np.mean(temp_final_w_y, axis = 1)
    
    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - final_w_x[i], Cartesian_Points[i,1] - final_w_y[i]]))
        
        #diff_error[i] = norm(np.array([Cartesian_Points[i,0] - temp_final_w_x[i,i], Cartesian_Points[i,1] - temp_final_w_y[i,i]]))
   
    return w_x, w_x[:,:,-1], w_y, w_y[:,:,-1], diff_error, C_2, MSD, new_Kmax, res, flag_for_res

def LMS(Kmax, L, b, q, A, init_Points, Cartesian_Points, lapl_Points, Deg):
    
    MSD = np.zeros(Kmax)
    
    w_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    w_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    w_x[:,0] = init_Points[:,0]
    w_y[:,0] = init_Points[:,1]
    
    G_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    G_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    #0.04*G_x[:,k+1]
    for k in range (Kmax-1):
        
        if (k == 0):
            
            G_x[:,k+1] = L.T@(b - L@w_x[:,k])
            
            G_y[:,k+1] = L.T@(q - L@w_y[:,k])
            
        else:
            
            G_x[:,k+1] = 0.3*G_x[:,k] + 0.7*(L.T@(b - L@w_x[:,k]))
            
            G_y[:,k+1] = 0.3*G_y[:,k] + 0.7*(L.T@(q - L@w_y[:,k]))
            
            #G_x[:,k+1] = L_bar.T@(b - L_bar@w_x[:,k])
            
            #G_y[:,k+1] = L_bar.T@(q - L_bar@w_y[:,k])
            
        #w_x[:,k+1] = w_x[:,k] + 0.04*G_x[:,k+1]
        
        #w_y[:,k+1] = w_y[:,k] + 0.04*G_y[:,k+1]
        
        fac = 2/np.real(np.max(np.linalg.eigvals(L.T@L)))
        
        #0.02
        w_x[:,k+1] = w_x[:,k] + min(fac, 0.02)*L.T@(b - L@w_x[:,k])
        
        w_y[:,k+1] = w_y[:,k] + min(fac, 0.02)*L.T@(q - L@w_y[:,k])
       
    for k in range (Kmax):
        
        MSD[k] = 10*np.log10((norm(np.array([Cartesian_Points[:,0] - w_x[:,k], Cartesian_Points[:,1] - w_y[:,k]]))**2)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2))
        

    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - w_x[i,-1], Cartesian_Points[i,1] - w_y[i,-1]]))
    
   
    return w_x, w_y, diff_error, MSD

def RLS(Kmax, L, b, q, A, init_Points, Cartesian_Points, lapl_Points, Deg):
    
    MSD = np.zeros(Kmax)
    
    w_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    w_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    w_x[:,0] = init_Points[:,0]
    w_y[:,0] = init_Points[:,1]
    
    P = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax))
   
    for i in range (Cartesian_Points.shape[0]):
        
        P[:,:,0] = 0.1*np.eye((Cartesian_Points.shape[0]))
        
    lam = 0.99
    
    g_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    g_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    cor_L = L.T@L
    
    g_x[:,0] = ((L.T@b) - cor_L@w_x[:,0])
    g_y[:,0] = ((L.T@q) - cor_L@w_y[:,0])
    
    p_x = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    p_y = np.zeros((Cartesian_Points.shape[0], Kmax))
    
    p_x[:,1] = g_x[:,0]
    p_y[:,1] = g_y[:,0]
    
    flag = 0
    
    res = np.zeros(Kmax)
   
    new_Kmax = Kmax
    
   
    #0.00003*
    for k in range (Kmax-1):
        
        
        a_x = (p_x[:,k+1].T@g_x[:,k]) / (p_x[:,k+1].T@cor_L@p_x[:,k+1])
        
        #print ('Iteration: ', k+1)
        #print (p_x[:,k+1].T@cor_L@p_x[:,k+1])
        
        w_x[:,k+1] = w_x[:,k] + a_x*p_x[:,k+1] 
        
        g_x_dot = (0.1*g_x[:,k] + ((L.T@b) - ((L.T@L))@w_x[:,k]))
        
        g_x[:,k+1] = (g_x_dot - a_x*(cor_L@p_x[:,k+1]))
        
        temp_g_x = g_x[:,k+1] - g_x[:,k]
        
        b_x = (temp_g_x.T@g_x[:,k+1]) / (g_x[:,k].T@g_x[:,k])
        
        if (k+2 <= Kmax-1):
            p_x[:,k+2] = (g_x[:,k+1] + b_x*p_x[:,k+1])
        
        a_y = (p_y[:,k+1].T@g_y[:,k]) / (p_y[:,k+1].T@cor_L@p_y[:,k+1])
        
        w_y[:,k+1] = w_y[:,k] + a_y*p_y[:,k+1]
        
        g_y_dot = (0.1*g_y[:,k] + ((L.T@q) - ((L.T@L))@w_y[:,k]))
        
        g_y[:,k+1] = (g_y_dot - a_y*(cor_L@p_y[:,k+1]))
        
        temp_g_y = g_y[:,k+1] - g_y[:,k]
        
        b_y = (temp_g_y.T@g_y[:,k+1]) / (g_y[:,k].T@g_y[:,k])
        
        if (k+2 <= Kmax-1):
            p_y[:,k+2] = (g_y[:,k+1] + b_y*p_y[:,k+1])
        
        cor_L = (L.T@L)
        
        '''
        w_x[:,k+1] = w_x[:,k] + (P[:,:,k]/(1 + (lam**(-1))*cor_L@P[:,:,k]@cor_L.T))@(L.T@b - cor_L@w_x[:,k])
        
        w_y[:,k+1] = w_y[:,k] + (P[:,:,k]/(1 + (lam**(-1))*cor_L@P[:,:,k]@cor_L.T))@(L.T@q - cor_L@w_y[:,k])
        
        P[:,:,k+1] = (lam**(-1))*(P[:,:,k] - ((lam**(-1))*P[:,:,k]@cor_L.T@cor_L@P[:,:,k])/(1 + (lam**(-1))*cor_L@P[:,:,k]@cor_L.T))
        '''
        res[k+1] = norm(np.array([w_x[:,k+1] - w_x[:,k], w_y[:,k+1] - w_y[:,k]]))**2
        
        if (flag == 0):
            
            if (abs(res[k+1] - res[k]) <= 0.001):
                new_Kmax = k+1
                flag = 1
        
        
    for k in range (Kmax):
        
        MSD[k] = 10*np.log10((norm(np.array([Cartesian_Points[:,0] - w_x[:,k], Cartesian_Points[:,1] - w_y[:,k]]))**2)/(norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2))
        

    diff_error = np.zeros(Cartesian_Points.shape[0])
     
    for i in range (Cartesian_Points.shape[0]):
       
        diff_error[i] = norm(np.array([Cartesian_Points[i,0] - w_x[i,-1], Cartesian_Points[i,1] - w_y[i,-1]]))
    
   
    return w_x, w_y, diff_error, MSD, res, Kmax

num_of_iter = 500
num_of_vehicles = 20

sigma_x = 3
sigma_y = 2.5
 
sigma_d = 1
sigma_a = 4

beta = 1
epsilon = 0.0

range_of_tranceivers = 20

number_of_connected_neighbours = 6

random_time_index = int(random.uniform(0,num_of_iter-1))


mean_GPS_error = np.zeros(num_of_iter)

mse_GPS_error = np.zeros(num_of_iter)
mse_lapl_error = np.zeros(num_of_iter)
mse_opt_aoa_error = np.zeros(num_of_iter)
mse_distr_lapl_error = np.zeros(num_of_iter)
mse_ATCCG_error = np.zeros(num_of_iter)
mse_ATCME_error = np.zeros(num_of_iter)
mse_ATC_error = np.zeros(num_of_iter)
mse_LMS_error = np.zeros(num_of_iter)

max_GPS_error = np.zeros(num_of_iter)
max_lapl_error = np.zeros(num_of_iter)
max_opt_aoa_error = np.zeros(num_of_iter)
max_distr_lapl_error = np.zeros(num_of_iter)
max_ATCCG_error = np.zeros(num_of_iter)
max_ATCME_error = np.zeros(num_of_iter)
max_ATC_error = np.zeros(num_of_iter)


Cartesian_Points = np.zeros((num_of_vehicles,3))    
    
for i in range(Cartesian_Points[:,0].size):
    Cartesian_Points[i,0] = random.uniform(0, 4*Cartesian_Points.shape[0])
    Cartesian_Points[i,1] = random.uniform(0, 6)


noisy_Cartesian = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[1]))
    
gps_noise_x = np.zeros((Cartesian_Points.shape[0]))
gps_noise_y = np.zeros((Cartesian_Points.shape[0]))

gps_noise_x = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points.shape[0])[0]
gps_noise_y = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points.shape[0])[1]


noisy_Cartesian[:,0] = Cartesian_Points[:,0] + gps_noise_x
noisy_Cartesian[:,1] = Cartesian_Points[:,1] + gps_noise_y
noisy_Cartesian[:,2] = Cartesian_Points[:,2] 

Dt = 0.1
tire_radius = 100

theta = np.zeros((Cartesian_Points.shape[0], num_of_iter)) 


for i in range(theta.shape[0]):
    theta[i,0] = random.uniform(math.radians(0), math.radians(6))

speed = np.zeros(num_of_vehicles)
speed = np.random.uniform(6, 7,num_of_vehicles)

        
array_of_time_lapl = np.zeros(num_of_iter)

array_of_time_distr_lapl = np.zeros(num_of_iter)

array_of_time_opt = np.zeros(num_of_iter)

true_X = np.zeros((num_of_vehicles, num_of_iter)) 
true_Y = np.zeros((num_of_vehicles, num_of_iter))

KM_true_X = np.zeros((num_of_vehicles, num_of_iter)) 
KM_true_Y = np.zeros((num_of_vehicles, num_of_iter))

traj_X = np.zeros((num_of_vehicles, num_of_iter)) 
traj_Y = np.zeros((num_of_vehicles, num_of_iter))

recon_X = np.zeros((num_of_vehicles, num_of_iter)) 
recon_Y = np.zeros((num_of_vehicles, num_of_iter))

recon_X_GD = np.zeros((num_of_vehicles, num_of_iter)) 
recon_Y_GD = np.zeros((num_of_vehicles, num_of_iter))
           
low_a = 70
high_a = 110


low_a_2 = 250
high_a_2 = 290

matrix_of_L_rank = np.zeros(num_of_iter)

max_iter = 150

Kmax_ATC = max_iter
Kmax_ATCCG = max_iter
Kmax_ATCME = max_iter
Kmax_LMS = 2
Kmax_RLS = 2

ATCCG_temp_MSD = np.zeros((Kmax_ATCCG, num_of_iter))
ATCCG_average_MSD = np.zeros(Kmax_ATCCG)

ATCME_temp_MSD = np.zeros((Kmax_ATCME, num_of_iter))
ATCME_average_MSD = np.zeros(Kmax_ATCME)

ATC_temp_MSD = np.zeros((Kmax_ATC, num_of_iter))
ATC_average_MSD = np.zeros(Kmax_ATC)

ATCCG_temp_MSD_wtr = np.zeros((Kmax_ATCCG, num_of_iter))
ATCCG_average_MSD_wtr = np.zeros(Kmax_ATCCG)

ATCME_temp_MSD_wtr = np.zeros((Kmax_ATCME, num_of_iter))
ATCME_average_MSD_wtr = np.zeros(Kmax_ATCME)

ATC_temp_MSD_wtr = np.zeros((Kmax_ATC, num_of_iter))
ATC_average_MSD_wtr = np.zeros(Kmax_ATC)

LMS_temp_MSD = np.zeros((Kmax_LMS, num_of_iter))
LMS_average_MSD = np.zeros(Kmax_LMS)

RLS_temp_MSD = np.zeros((Kmax_RLS, num_of_iter))
RLS_average_MSD = np.zeros(Kmax_RLS)

flag_for_deg_1 = np.zeros(num_of_iter)
flag_for_deg_2 = np.zeros(num_of_iter)

lapl_MSD = np.zeros(num_of_iter)

local_lapl_MSD = np.zeros(num_of_iter)

GPS_MSD = np.zeros(num_of_iter)

L_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATC_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATC_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATCME_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATCME_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATCCG_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATCCG_full = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATC_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATC_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATCME_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATCME_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATCCG_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))
y_ATCCG_full_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], num_of_iter))

x_ATC = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATC, num_of_iter))
y_ATC = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATC, num_of_iter))

x_ATCME = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATCME, num_of_iter))
y_ATCME = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATCME, num_of_iter))

x_ATCCG = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATCCG, num_of_iter))
y_ATCCG = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0], Kmax_ATCCG, num_of_iter))

new_Kmax_RLS = np.zeros(num_of_iter)
new_Kmax_ATC = np.zeros(num_of_iter)
new_Kmax_ATCME = np.zeros(num_of_iter)
new_Kmax_ATCCG = np.zeros(num_of_iter)

new_Kmax_ATC_wtr = np.zeros(num_of_iter)
new_Kmax_ATCME_wtr = np.zeros(num_of_iter)
new_Kmax_ATCCG_wtr = np.zeros(num_of_iter)

true_delta_X = np.zeros((Cartesian_Points.shape[0], num_of_iter))
true_delta_Y = np.zeros((Cartesian_Points.shape[0], num_of_iter))

est_delta_X = np.zeros((Cartesian_Points.shape[0], num_of_iter))
est_delta_Y = np.zeros((Cartesian_Points.shape[0], num_of_iter))

full_lapl_error = np.zeros((Cartesian_Points.shape[0], num_of_iter))
full_gps_error = np.zeros((Cartesian_Points.shape[0], num_of_iter))

full_C = np.zeros((num_of_iter, Cartesian_Points.shape[0], Cartesian_Points.shape[0]))

step_a_x = np.zeros((Cartesian_Points.shape[0], Kmax_ATCCG, num_of_iter))
step_a_y = np.zeros((Cartesian_Points.shape[0], Kmax_ATCCG, num_of_iter))

delta_error = np.zeros((Cartesian_Points.shape[0], num_of_iter))

array_of_time_ATC = np.zeros(num_of_iter)
array_of_time_ATCME = np.zeros(num_of_iter)
array_of_time_ATCCG = np.zeros(num_of_iter)

loc_error_true_KM = np.zeros((Cartesian_Points.shape[0], num_of_iter))
for v in range(num_of_iter):
    
    
    print ('Number of vehicles is: ' , Cartesian_Points[:,0].size)
    A = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    Deg = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    
    D, a = np.zeros((Cartesian_Points[:,0].size,Cartesian_Points[:,0].size)), np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    AoA = np.zeros((Cartesian_Points[:,0].size,Cartesian_Points[:,0].size))
    
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            D[i,j] = norm(np.array([Cartesian_Points[i][0]-Cartesian_Points[j][0], Cartesian_Points[i][1]-Cartesian_Points[j][1]]),2)
            
    
    noise_for_distances = np.zeros(Cartesian_Points[:,0].size**2)
    noise_for_distances = (1- epsilon)*np.random.normal(0, sigma_d, Cartesian_Points[:,0].size**2) + epsilon*np.random.gumbel(0, beta, Cartesian_Points[:,0].size**2)
    noise_for_distances = np.reshape(noise_for_distances, (Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    '''
    for i in range (Cartesian_Points[:,0].size):
        for j in range (Cartesian_Points[:,0].size):
            if (i>j):
                noise_for_distances[i,j] = noise_for_distances[j,i]
    
    '''       
    noisy_D = np.zeros((D[:,0].size,D[0,:].size))
    
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            noisy_D[i,j] = D[i,j] + np.random.normal(0, sigma_d, 1)
            
            
            
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (i!=j):
                if (D[i,j] <= range_of_tranceivers):
                    A[i,j] = 1
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (A[i,j] == 1 and A[j,i] == 0):
                A[j,i] = 1
            elif (A[i,j] == 0 and A[j,i] == 1):
                A[i,j] = 1
                    
                    
    
    for i in range(A[:,0].size):
        count = 0
        for j in range(A[0,:].size):
            if (A[i,j] == 1):
                count += 1
                if (count > number_of_connected_neighbours):
                    A[i,j] = 0
                    A[j,i] = 0
    
    for i in range (A[:,0].size):
        A[i,i] = 0
        Deg[i,i] = np.sum([A[i,:]])
        
  
    for i in range(a[:,0].size):
        for j in range(a[0,:].size):
            if (i == j):
                D[i,j] = 0
                noisy_D[i,j] = 0
           
            if (A[i,j] == 1):
                a[i,j] = CalculateAzimuthAngle(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])
                
                AoA[i,j] = CalculateAoA(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])[0]
                
            else:
                D[i,j] = 0
                noisy_D[i,j] = 0
            if (noisy_D[i,j] < 0 ):
                noisy_D[i,j] *= -1
          
                #print ('Bad')
                #flag_neg_d = 0
           
    list_for_del = []               
    for i in range (A[:,0].size):
        if (Deg[i,i] == 0):
            print ('Index for delete is: ' + str(i),'\n')
            list_for_del.append(i)
    
    list_for_rows = []
    list_for_cols = []        
    for i in range(noisy_D[:,0].size):
        for j in range(noisy_D[0,:].size):
            if (A[i,j] == 1 and noisy_D[i,j]<=3):
                list_for_rows.append(i)
                list_for_cols.append(j)
                
    list_for_rows = list(dict.fromkeys(list_for_rows))
    list_for_cols = list(dict.fromkeys(list_for_cols))
    comb_list = list_for_rows + list_for_cols
    comb_list = list(dict.fromkeys(comb_list))
        
    array_for_del = np.asarray(list_for_del)
    
    
    if (len(array_for_del) > 0):
        print ('Shape of Degree matrix is: ' , np.shape(Deg),'\n')
        Deg = np.delete(Deg, array_for_del, 0)
        A = np.delete(A, array_for_del, 0)
        Cartesian_Points = np.delete(Cartesian_Points, array_for_del, 0)
        noisy_Cartesian = np.delete(noisy_Cartesian, array_for_del, 0)
        D = np.delete(D, array_for_del, 0)
        noisy_D = np.delete(noisy_D, array_for_del, 0)
        a = np.delete(a, array_for_del, 0)
        AoA = np.delete(AoA, array_for_del, 0)
        #speed = np.delete(speed,array_for_del,0)
        traj_X = np.delete(traj_X,array_for_del,0)
        traj_Y = np.delete(traj_Y,array_for_del,0)
        recon_X = np.delete(recon_X,array_for_del,0)
        recon_Y = np.delete(recon_Y,array_for_del,0)
        recon_X_GD = np.delete(recon_X_GD,array_for_del,0)
        recon_Y_GD = np.delete(recon_Y_GD,array_for_del,0)
        true_X = np.delete(true_X,array_for_del,0)
        true_Y = np.delete(true_Y,array_for_del,0)
        #theta = np.delete(theta,array_for_del,0)
        Deg = np.delete(Deg, array_for_del, 1)
        A = np.delete(A, array_for_del, 1)
        D = np.delete(D, array_for_del, 1)
        noisy_D = np.delete(noisy_D, array_for_del, 1)
        a = np.delete(a, array_for_del, 1)
        AoA = np.delete(AoA, array_for_del, 1)
        print ('Shape of Degree matrix is: ' , np.shape(Deg),'\n')
    
    
    #print (np.array_equal(A,A.T))
    #L = np.zeros((A[:,0].size,A[0,:].size))
   
            
    L = Deg - A 
    
    if (np.sum(Deg) % Cartesian_Points.shape[0] == 0):
        print ('All to all')
    '''
    ######!!!!!!!!!!! Normalized symmetric Laplacian Matrix !!!!!!########
    L = inv(Deg)@(Deg-A)
    ######!!!!!!!!!!! Normalized symmetric Laplacian Matrix !!!!!!########
    '''
    '''
    ######!!!!!!!!!!! Random walk normalized Laplacian Matrix !!!!!!######## 
    new_Deg = np.zeros((Deg[:,0].size,Deg[:,0].size))
    
    for i in range(new_Deg[:,0].size):
        for j in range(new_Deg[0,:].size):
            if (Deg[i,j]!=0):
                new_Deg[i,j] = Deg[i,j]**(-1/2)
    
    L = new_Deg@(Deg-A)@new_Deg  
    ######!!!!!!!!!!! Random walk normalized Laplacian Matrix !!!!!!########
    '''
    '''
    ######!!!!!!!!!!! Transition Laplacian Matrix !!!!!!######## 
    L = inv(Deg)@A 
    ######!!!!!!!!!!! Transition Laplacian Matrix !!!!!!########
    '''
    
    
    eig = np.sort(np.linalg.eigvals(L))
    
    L_full[:,:,v] = L
    
    matrix_of_L_rank[v] = matrix_rank(L)
    '''
    if (np.argwhere(1 == Deg).size > 0):
        flag_for_deg_1[v] = 1
        print ('Possible problem 1')
    if (np.argwhere(2 == Deg).size > 0):
        flag_for_deg_2[v] = 1
        print ('Possible problem 2')
    '''
    if (matrix_of_L_rank[v] < L.shape[1]-1):
        print ('Rank problem')
        break;
    
    GPS_error = np.zeros((Cartesian_Points[:,0].size))
    GPS_error_x = np.zeros((Cartesian_Points[:,0].size))
    GPS_error_y = np.zeros((Cartesian_Points[:,0].size))
    for i in range(GPS_error.size):
        GPS_error[i] = norm(np.array([Cartesian_Points[i,0] - noisy_Cartesian[i,0] , Cartesian_Points[i,1] - noisy_Cartesian[i,1]]), 2)
        
        GPS_error_x[i] = norm(np.array([Cartesian_Points[i,0] - noisy_Cartesian[i,0]]), 2)
        GPS_error_y[i] = norm(np.array([Cartesian_Points[i,1] - noisy_Cartesian[i,1]]), 2)
    
    GPS_MSD[v] = 10*np.log10((norm(np.array([Cartesian_Points[:,0] - noisy_Cartesian[:,0], Cartesian_Points[:,1] - noisy_Cartesian[:,1]]))**2) / (norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2))
    
    full_gps_error[:,v] = GPS_error
    noisy_a = np.zeros((a[:,0].size,a[0,:].size))
    
    noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    
    deg_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    deg_true_AoA = np.zeros((a[:,0].size,a[0,:].size))
    tan_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    temp_tan_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
   
    
    noise_for_angles = np.zeros(Cartesian_Points[:,0].size**2)
    noise_for_angles = (1- epsilon)*np.random.normal(0, math.radians(sigma_a), Cartesian_Points[:,0].size**2) + epsilon*np.random.gumbel(0, beta, Cartesian_Points[:,0].size**2)
    noise_for_angles = np.reshape(noise_for_angles, (Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    '''
    for i in range (Cartesian_Points[:,0].size):
        for j in range (Cartesian_Points[:,0].size):
            if (i>j):
                noise_for_angles[i,j] = noise_for_angles[j,i]
                
    '''            
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (A[i,j] == 1):
                noisy_a[i,j] = a[i,j] + np.random.normal(0,math.radians(sigma_a),size = 1)
                
                #np.random.normal(0,math.radians(sigma_a),size = 1)
                noisy_AoA[i,j] = AoA[i,j] + np.random.normal(0,math.radians(sigma_a),size = 1)
                #np.random.normal(0,math.radians(sigma_a),size = 1)
                
                if (noisy_AoA[i,j] > math.radians(360)):
                    noisy_AoA[i,j] = math.radians(360)
                if (noisy_AoA[i,j] < math.radians(-360)):
                    noisy_AoA[i,j] = math.radians(-360)
                    
                deg_noisy_AoA[i,j] = math.degrees(noisy_AoA[i,j])
                
                deg_true_AoA[i,j] = math.degrees(AoA[i,j])
                
                tan_noisy_AoA[i,j] = (math.tan(noisy_AoA[i,j]))
    
                if (deg_noisy_AoA[i,j] <= 90 and deg_noisy_AoA[i,j] >= low_a):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(low_a)))
                    
                elif (deg_noisy_AoA[i,j] <= high_a and deg_noisy_AoA[i,j] >= 90):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(high_a)))
                    
                elif (deg_noisy_AoA[i,j] >= -90 and deg_noisy_AoA[i,j] <= -low_a):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(-low_a)))
                    
                elif (deg_noisy_AoA[i,j] >= -high_a and deg_noisy_AoA[i,j] <= -90):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(-high_a)))
                    
                else:    
                    temp_tan_noisy_AoA[i,j] = (math.tan(noisy_AoA[i,j]))
                    
                    
                if (deg_noisy_AoA[i,j] <= 90 and deg_noisy_AoA[i,j] >= low_a_2):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(low_a_2)))
                    
                elif (deg_noisy_AoA[i,j] <= high_a_2 and deg_noisy_AoA[i,j] >= 90):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(high_a_2)))
                    
                elif (deg_noisy_AoA[i,j] >= -90 and deg_noisy_AoA[i,j] <= -low_a_2):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(-low_a_2)))
                    
                elif (deg_noisy_AoA[i,j] >= -high_a_2 and deg_noisy_AoA[i,j] <= -90):
                    temp_tan_noisy_AoA[i,j] = (math.tan(math.radians(-high_a_2)))
                    
                else:    
                    temp_tan_noisy_AoA[i,j] = (math.tan(noisy_AoA[i,j]))    
   
    delta_X = np.zeros((Cartesian_Points[:,0].size))
    delta_Y = np.zeros((Cartesian_Points[:,0].size))
                 
    temp_delta_X = np.zeros((Cartesian_Points[:,0].size))
    temp_delta_Y = np.zeros((Cartesian_Points[:,0].size))
    for i in range(noisy_D[:,0].size):
        for j in range(noisy_D[0,:].size):
            if (A[i,j] == 1):
                delta_X[i] += (-noisy_D[i,j]*math.sin(noisy_a[i,j]))
                delta_Y[i] += (-noisy_D[i,j]*math.cos(noisy_a[i,j]))
                
                temp_delta_X[i] += (Cartesian_Points[i,0] - Cartesian_Points[j,0])
                temp_delta_Y[i] += (Cartesian_Points[i,1] - Cartesian_Points[j,1])
                
          
       
    true_delta_X[:,v] = temp_delta_X
    true_delta_Y[:,v] = temp_delta_Y
    
    est_delta_X[:,v] = delta_X
    est_delta_Y[:,v] = delta_Y
    
    for i in range (noisy_D[:,0].size):
        
        delta_error[i,v] = norm(np.array([true_delta_X[i,v] - est_delta_X[i,v] , true_delta_Y[i,v] - est_delta_Y[i,v]]), 2)
    #########!!!!!!! ONLY LAPLACIAN !!!!!!!!!############
    start_lapl = timer()
    '''
    list_random_anchors_idx=[]
    list_random_anchors=[]
    
    number_of_anchors = int(round(1.0*Cartesian_Points[:,0].size))
    for i in range(number_of_anchors):
        
        r=random.randint(0,Cartesian_Points[:,0].size-1)
        while (r in list_random_anchors_idx):
            r=random.randint(0,Cartesian_Points[:,0].size-1)
        list_random_anchors_idx.append(r)
        list_random_anchors.append(noisy_Cartesian[r,:])
        
    
    anchors_index = np.zeros((len(list_random_anchors_idx)))
    anchors = np.zeros((len(list_random_anchors)))
    
    anchors_index = np.asarray(list_random_anchors_idx)
    anchors = np.asarray(list_random_anchors)
    '''
    
    anchors_index = np.arange(Cartesian_Points.shape[0])
    anchors = np.copy(noisy_Cartesian[:,:2])
    
    Points = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
    
    lapl_error = np.zeros((Cartesian_Points[:,0].size))
    
    L_bar = np.zeros((Cartesian_Points[:,0].size + anchors_index.size, Cartesian_Points[0,:].size))
    
    Points, lapl_error, L_bar = Solve_The_System(Cartesian_Points, L, anchors_index, anchors, delta_X, delta_Y)
    
    lapl_error_x = np.zeros(Cartesian_Points.shape[0])
    lapl_error_y = np.zeros(Cartesian_Points.shape[0])
    
    for i in range (Cartesian_Points.shape[0]):
        
        lapl_error_x[i] = norm(Cartesian_Points[i,0] - Points[i,0])
        lapl_error_y[i] = norm(Cartesian_Points[i,1] - Points[i,1])
   
    end_lapl = timer()  
    
    full_lapl_error[:,v] = lapl_error
    
    lapl_MSD[v] = 10*np.log10((norm(np.array([Cartesian_Points[:,0] - Points[:,0], Cartesian_Points[:,1] - Points[:,1]]))**2) / (norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2))
    
    array_of_time_lapl[v] = end_lapl - start_lapl
    #########!!!!!!! ONLY LAPLACIAN !!!!!!!!!############
    
    
    #########!!!!!!! ONLY DISTRIBUTED LAPLACIAN !!!!!!!!!############
    start_distr_lapl = timer()
    x_distr_lapl, y_distr_lapl = np.zeros(Cartesian_Points[:,0].size), np.zeros(Cartesian_Points[:,0].size)
    
    x_distr_lapl, y_distr_lapl = Distributed_Lapl(Deg, A, delta_X, delta_Y, noisy_Cartesian[:,0], noisy_Cartesian[:,1], noisy_D, noisy_a)
    
    distributed_lapl_error = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (x_distr_lapl.size):
        distributed_lapl_error[i] = norm(np.array([x_distr_lapl[i] - Cartesian_Points[i,0], y_distr_lapl[i] - Cartesian_Points[i,1]]),2)  
     
    local_Points = np.zeros((Cartesian_Points.shape[0],2))
    
    local_Points[:,0] = np.copy(x_distr_lapl)
    local_Points[:,1] = np.copy(y_distr_lapl)
    
    local_lapl_MSD[v] = 10*np.log10((norm(np.array([Cartesian_Points[:,0] - local_Points[:,0], Cartesian_Points[:,1] - local_Points[:,1]]))**2) / (norm(np.array([Cartesian_Points[:,0], Cartesian_Points[:,1]]))**2))
    
    end_distr_lapl = timer()
    array_of_time_distr_lapl[v] = end_distr_lapl - start_distr_lapl    
    #########!!!!!!! ONLY DISTRIBUTED LAPLACIAN !!!!!!!!!############
    
    '''
    #########!!!!!!! OPTIMIZATION_AoA !!!!!!!!!############ 
    start_opt = timer()
    
    x_opt_aoa = np.zeros((Cartesian_Points[:,0].size))
    y_opt_aoa = np.zeros((Cartesian_Points[:,0].size))
   
    
    x_opt_aoa, y_opt_aoa = Optimization_AoA_GD(A, noisy_D, tan_noisy_AoA, noisy_Cartesian[:,0], noisy_Cartesian[:,1], deg_noisy_AoA)
    
    #x_opt_aoa, y_opt_aoa = Optimization_AoA(A, noisy_D, noisy_Cartesian[:,0], noisy_Cartesian[:,1], tan_noisy_AoA, deg_noisy_AoA)
    
    error_opt_aoa = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_opt_aoa.size):
        error_opt_aoa[i] = norm(np.array([Cartesian_Points[i,0] - x_opt_aoa[i], Cartesian_Points[i,1] - y_opt_aoa[i]]), 2)
    
    
    end_opt = timer()    
    array_of_time_opt[v] = end_opt-start_opt
    #########!!!!!!! OPTIMIZATION_AoA !!!!!!!!!############
    '''
    
    
    #########!!!!!!! Diffusion !!!!!!!!!############ 
    
   
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
   
    n_KM_x_comp, n_KM_y_comp = np.zeros(Cartesian_Points.shape[0]), np.zeros(Cartesian_Points.shape[0])
    
    n_speed, n_yaw_rate = np.zeros(Cartesian_Points.shape[0]), np.zeros(Cartesian_Points.shape[0])
    
    
    for i in range(Cartesian_Points.shape[0]):
        
        n_speed[i] = (speed[i] + np.random.normal(0, 0.3*speed[i]))
       
        n_KM_x_comp[i] =  (n_speed[i])*math.cos(theta[i,v] + np.random.normal(0, 0.5))*Dt 
        n_KM_y_comp[i] =  (n_speed[i])*math.sin(theta[i,v] + np.random.normal(0, 0.5))*Dt
       
        if (v > 0):
            KM_true_X[i,v] = KM_true_X[i,v-1] + n_KM_x_comp[i]
            KM_true_Y[i,v] = KM_true_Y[i,v-1] + n_KM_y_comp[i]
        else:
            KM_true_X[i,v] = noisy_Cartesian[i,0]
            KM_true_Y[i,v] = noisy_Cartesian[i,1]
            
        loc_error_true_KM[i,v] = norm([Cartesian_Points[i,0] - KM_true_X[i,v], Cartesian_Points[i,1] - KM_true_Y[i,v]])
            
    ATC_init_Points = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0],2))
    
    ATCME_init_Points = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0],2))
    
    ATCCG_init_Points = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0],2))
    
    init_Points_wtr = np.zeros((Cartesian_Points.shape[0], Cartesian_Points.shape[0],2))
    
    init_Points_wtr[:,:,0] = noisy_Cartesian[:,0]
    init_Points_wtr[:,:,1] = noisy_Cartesian[:,1]
    
    #if (v % 100 == 0):
        
        #ATCCG_init_Points[:,:,0] = noisy_Cartesian[:,0]
        #ATCCG_init_Points[:,:,1] = noisy_Cartesian[:,1]
        
    if (v == 0):
        
        ATC_init_Points[:,:,0] = noisy_Cartesian[:,0]
        ATC_init_Points[:,:,1] = noisy_Cartesian[:,1]
        
        ATCME_init_Points[:,:,0] = noisy_Cartesian[:,0]
        ATCME_init_Points[:,:,1] = noisy_Cartesian[:,1]
        
        ATCCG_init_Points[:,:,0] = noisy_Cartesian[:,0]
        ATCCG_init_Points[:,:,1] = noisy_Cartesian[:,1]
        
    else:
        
        for i in range (Cartesian_Points.shape[0]):
            
            ATC_init_Points[i,:,0] = x_ATC_full[i,:,v-1] + n_KM_x_comp[i]
            ATC_init_Points[i,:,1] = y_ATC_full[i,:,v-1] + n_KM_y_comp[i]
            
            ATCME_init_Points[i,:,0] = x_ATCME_full[i,:,v-1] 
            ATCME_init_Points[i,:,1] = y_ATCME_full[i,:,v-1] 
            
            ATCCG_init_Points[i,:,0] = x_ATCCG_full[i,:,v-1] + n_KM_x_comp[i]
            ATCCG_init_Points[i,:,1] = y_ATCCG_full[i,:,v-1] + n_KM_y_comp[i]
            
            '''
            
            ATC_init_Points[i,:,0] = x_ATC_full[i,:,v-1] 
            ATC_init_Points[i,:,1] = y_ATC_full[i,:,v-1] 
            
            ATCME_init_Points[i,:,0] = x_ATCME_full[i,:,v-1] 
            ATCME_init_Points[i,:,1] = y_ATCME_full[i,:,v-1] 
            
            ATCCG_init_Points[i,:,0] = x_ATCCG_full[i,:,v-1] 
            ATCCG_init_Points[i,:,1] = y_ATCCG_full[i,:,v-1]
            
            ATC_init_Points[i,:,0] = noisy_Cartesian[i,0]
            ATC_init_Points[i,:,1] = noisy_Cartesian[i,1]
            
            ATCME_init_Points[i,:,0] = noisy_Cartesian[i,0]
            ATCME_init_Points[i,:,1] = noisy_Cartesian[i,1]
            
            ATCCG_init_Points[i,:,0] = noisy_Cartesian[i,0]
            ATCCG_init_Points[i,:,1] = noisy_Cartesian[i,1]
            '''
   
    #x_CTA, x_CTA_full[:,:,v], y_CTA, y_CTA_full[:,:,v], temp_CTA_error, C_CTA, CTA_temp_MSD[:,v], new_Kmax_CTA[v], step_b_x[:,:,v], step_b_y[:,:,v], res_CTA, dif_res_CTA = CTA(Kmax_CTA, L, delta_X, delta_Y, A, CTA_init_Points, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
            
    #x_ATC, x_ATC_full[:,:,v], y_ATC, y_ATC_full[:,:,v], temp_ATC_error, C_ATC, ATC_temp_MSD[:,v], conc_L, step_a_x[:,:,v], step_a_y[:,:,v], new_Kmax_ATC[v], flag_for_psd, res_ATC, dif_res_ATC, flag_for_converg_ATC = ATC(Kmax_ATC, L, delta_X, delta_Y, A, ATC_init_Points, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    start_ATC = timer()
    
    x_ATC[:,:,:,v], x_ATC_full[:,:,v], y_ATC[:,:,:,v], y_ATC_full[:,:,v], temp_ATC_error, C_ATC, ATC_temp_MSD[:,v], new_Kmax_ATC[v], res_ATC, dif_res_ATC = LMSATC(Kmax_ATC, L, delta_X, delta_Y, A, ATC_init_Points, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    end_ATC = timer()
    array_of_time_ATC[v] = end_ATC-start_ATC
    
    start_ATCME = timer()
    
    x_ATCME[:,:,:,v], x_ATCME_full[:,:,v], y_ATCME[:,:,:,v], y_ATCME_full[:,:,v], temp_ATCME_error, C_ATCME, ATCME_temp_MSD[:,v], new_Kmax_ATCME[v], res_ATCME, dif_res_ATCME = ATCME(Kmax_ATCME, L, delta_X, delta_Y, A, ATCME_init_Points, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    end_ATCCG = timer()
    array_of_time_ATCME[v] = end_ATCCG-start_ATCME
    
    start_ATCCG = timer()
    
    x_ATCCG[:,:,:,v], x_ATCCG_full[:,:,v], y_ATCCG[:,:,:,v], y_ATCCG_full[:,:,v], temp_ATCCG_error, C_ATCCG, ATCCG_temp_MSD[:,v], new_Kmax_ATCCG_wtr[v], res_ATCCG, dif_res_ATCCG = ATCCG(Kmax_ATC, L_bar, b, q, A, ATCCG_init_Points, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    end_ATCCG = timer()
    array_of_time_ATCCG[v] = end_ATCCG-start_ATCCG
    
    x_ATC_wtr, x_ATC_full_wtr[:,:,v], y_ATC_wtr, y_ATC_full_wtr[:,:,v], temp_ATC_error_wtr, C_ATC_wtr, ATC_temp_MSD_wtr[:,v], new_Kmax_ATC[v], res_ATC_wtr, dif_res_ATC_wtr = LMSATC(Kmax_ATC, L, delta_X, delta_Y, A, init_Points_wtr, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    x_ATCME_wtr, x_ATCME_full_wtr[:,:,v], y_ATCME_wtr, y_ATCME_full_wtr[:,:,v], temp_ATCME_error_wtr, C_ATCME_wtr, ATCME_temp_MSD_wtr[:,v], new_Kmax_ATCME_wtr[v], res_ATCME_wtr, dif_res_ATCME_wtr = ATCME(Kmax_ATCME, L, delta_X, delta_Y, A, init_Points_wtr, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
            
    x_ATCCG_wtr, x_ATCCG_full_wtr[:,:,v], y_ATCCG_wtr, y_ATCCG_full_wtr[:,:,v], temp_ATCCG_error_wtr, C_ATCCG_wtr, ATCCG_temp_MSD_wtr[:,v], new_Kmax_ATCCG_wtr[v], res_ATCCG_wtr, dif_res_ATCCG_wtr = ATCCG(Kmax_ATC, L_bar, b, q, A, init_Points_wtr, Cartesian_Points, noisy_Cartesian, Points, Deg, v)
    
    x_LMS, y_LMS, temp_LMS_error, LMS_temp_MSD[:,v] = LMS(Kmax_LMS, L_bar, b, q, A, noisy_Cartesian, Cartesian_Points, Points, Deg)
    
    x_RLS, y_RLS, temp_RLS_error, RLS_temp_MSD[:,v], res, new_Kmax_RLS[v] = RLS(Kmax_RLS, L_bar, b, q, A, noisy_Cartesian, Cartesian_Points, Points, Deg)
    
    mean_res_ATC = np.mean(res_ATC, axis = 0)
    
    #diff_error = np.mean(temp_diff_error, axis = 1)
    
    conc_rank = np.zeros(Cartesian_Points.shape[0])
    
    #for i in range (Cartesian_Points.shape[0]):
        
        #conc_rank[i] = norm(conc_L[i,:,:])*norm(inv(conc_L[i,:,:]))
        
    full_C[v,:,:] = C_ATC
    ATC_error = np.copy(temp_ATC_error)
    ATCME_error = np.copy(temp_ATCME_error)
    ATCCG_error = np.copy(temp_ATCCG_error)
    #########!!!!!!! Diffusion !!!!!!!!!############
    
    
    print ('Iteration: ', v, '\n')
    
    mean_GPS_error[v] = np.mean(GPS_error)
    
    mse_GPS_error[v] = ((np.sum(np.power(GPS_error,2))/Cartesian_Points[:,0].size))
    mse_lapl_error[v] = ((np.sum(np.power(lapl_error,2))/Cartesian_Points[:,0].size))
    #mse_opt_aoa_error[v] = ((np.sum(np.power(error_opt_aoa,2))/Cartesian_Points[:,0].size))
    mse_distr_lapl_error[v] = ((np.sum(np.power(distributed_lapl_error,2))/Cartesian_Points[:,0].size))
    mse_ATC_error[v] = ((np.sum(np.power(ATC_error,2))/Cartesian_Points[:,0].size))
    mse_ATCME_error[v] = ((np.sum(np.power(ATCME_error,2))/Cartesian_Points[:,0].size))
    mse_ATCCG_error[v] = ((np.sum(np.power(ATCCG_error,2))/Cartesian_Points[:,0].size))
    
    #if (mse_ATC_error[v] > 50):
        #print ('Bad')
        #break
    
    max_GPS_error[v] = np.max(GPS_error)
    max_lapl_error[v] = np.max(lapl_error)
    #max_opt_aoa_error[v] = np.max(error_opt_aoa)
    max_distr_lapl_error[v] = np.max(distributed_lapl_error)
    max_ATC_error[v] = np.max(ATC_error)
    max_ATCME_error[v] = np.max(ATCME_error)
    max_ATCCG_error[v] = np.max(ATCCG_error)
    
    #if (math.isinf(mse_opt_aoa_error[v]) or math.isnan(mse_opt_aoa_error[v])):
        #break;
   
    true_X[:,v] = Cartesian_Points[:,0]
    true_Y[:,v] = Cartesian_Points[:,1]
    
    traj_X[:,v] = noisy_Cartesian[:,0]
    traj_Y[:,v] = noisy_Cartesian[:,1]
    
    #recon_X[:,v] = x_opt_aoa
    #recon_Y[:,v] = y_opt_aoa
    
    #recon_X_GD[:,v] = x_opt_aoa
    #recon_Y_GD[:,v] = y_opt_aoa
   
    
    if (v < num_of_iter-1):
        
        #if ((Dt*(v+1) % 50) == 0):
            #speed = np.zeros(Cartesian_Points[:,0].size)
            #speed = np.random.uniform(6,7,Cartesian_Points[:,0].size)
            #speed = np.random.uniform(28,29,Cartesian_Points[:,0].size)
            #print ('Time for speed change: ', Dt*(v+1),'\n')
        
        for i in range(Cartesian_Points[:,0].size):
            #speed = random.uniform(27,33)
            yaw_rate = speed[i]/tire_radius
             
            '''
            if (v % 20 == 0):
                theta[i,v] = (-20)
            elif (v % 30 == 0):
                theta [i,v]= (130)
            elif (v % 50 == 0):
                theta[i,v] = (50)
            elif (v % 70 == 0):
                theta[i,v] = (100)
            
            if (v % 30 == 0):
                theta[i,v] = math.radians(20)
            
            elif (v % 65 == 0):
                
                theta[i,v] = (-130)
                
            elif (v % 93 == 0):
                theta[i,v] = (50)
                
            elif (v % 129 == 0):
                theta[i,v] = (100)
            
            '''
            Cartesian_Points[i,0] +=  (-speed[i]/yaw_rate)*math.sin(theta[i,v]) + (speed[i]/yaw_rate)*math.sin(theta[i,v] + yaw_rate*Dt)
            Cartesian_Points[i,1] +=  (speed[i]/yaw_rate)*math.cos(theta[i,v])  + (-speed[i]/yaw_rate)*math.cos(theta[i,v] + yaw_rate*Dt)
            
            theta[i,v+1] = theta[i,v] + Dt*yaw_rate
            
            #theta[i,v+1] = theta[i,v] + 0.004
            
            '''
            Cartesian_Points[i,0] += (Dt)*speed[i]
            Cartesian_Points[i,1] += (Dt)*speed[i]
            
            Cartesian_Points[i,0] = Cartesian_Points[i,0] + (Dt)*speed[i]*math.cos(theta )
            Cartesian_Points[i,1] = Cartesian_Points[i,1] + (Dt)*speed[i]*math.sin(theta )
            
            
             
            if (v % 20 == 0):
                theta = math.radians(-20)
            elif (v % 30 == 0):
                theta = math.radians(130)
            elif (v % 50 == 0):
                theta = math.radians(50)
            elif (v % 70 == 0):
                theta = math.radians(100)
                    
            Cartesian_Points[i,0] = Cartesian_Points[i,0] + (Dt)*speed[i]*math.cos(theta + 1/(2*(Dt)*yaw_rate))
            Cartesian_Points[i,1] = Cartesian_Points[i,1] + (Dt)*speed[i]*math.sin(theta + 1/(2*(Dt)*yaw_rate))
            '''
            
        noisy_Cartesian = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
        
        gps_noise_x = np.zeros((Cartesian_Points[:,0].size))
        gps_noise_y = np.zeros((Cartesian_Points[:,0].size))
     
        
        gps_noise_x = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[0][:Cartesian_Points[:,0].size]
        gps_noise_y = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[1][:Cartesian_Points[:,1].size]
        
        noisy_Cartesian[:,0] = Cartesian_Points[:,0] + gps_noise_x
        noisy_Cartesian[:,1] = Cartesian_Points[:,1] + gps_noise_y
        noisy_Cartesian[:,2] = Cartesian_Points[:,2]
       


ATC_average_MSD = np.mean(ATC_temp_MSD, axis = 1)
ATCME_average_MSD = np.mean(ATCME_temp_MSD, axis = 1)
ATCCG_average_MSD = np.mean(ATCCG_temp_MSD, axis = 1)

ATC_average_MSD_wtr = np.mean(ATC_temp_MSD_wtr, axis = 1)
ATCME_average_MSD_wtr = np.mean(ATCME_temp_MSD_wtr, axis = 1)
ATCCG_average_MSD_wtr = np.mean(ATCCG_temp_MSD_wtr, axis = 1)

LMS_average_MSD = np.mean(LMS_temp_MSD, axis = 1)
RLS_average_MSD = np.mean(RLS_temp_MSD, axis = 1)

print ('Mean time of Lapl is: ', np.mean(array_of_time_lapl), '\n')
print ('Mean time of Distr Lapl is: ', np.mean(array_of_time_distr_lapl), '\n')
print ('Mean time of Opt is: ', np.mean(array_of_time_opt), '\n')

print ('\n')

print ('Mean of GPS noise is: ', np.mean(mean_GPS_error), '\n')

print ('\n')

print ('Mean of delta error is: ', np.mean(delta_error), '\n')
print ('Mean MSE of GPS is: ', np.mean(mse_GPS_error), '\n')
print ('Mean MSE of Laplacian is: ', np.mean(mse_lapl_error), '\n')
print ('Mean MSE of Distr Laplacian is: ', np.mean(mse_distr_lapl_error), '\n')
print ('Mean MSE of Optimization_AoA is: ', np.mean(mse_opt_aoa_error), '\n')


if (norm(mse_lapl_error) < norm(mse_GPS_error)):
    print ('MSE Laplacian reduction: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error))
    
else:
    print ('MSE Laplacian increment: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error))


if (norm(mse_distr_lapl_error) < norm(mse_GPS_error)):    
    print ('MSE Distr Laplacian reduction: ', norm(mse_distr_lapl_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE Distr Laplacian increment: ', norm(mse_distr_lapl_error- mse_GPS_error)/norm(mse_GPS_error))    

   
if (norm(mse_opt_aoa_error) < norm(mse_GPS_error)):    
    print ('MSE Opt reduction: ', norm(mse_opt_aoa_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE Opt increment: ', norm(mse_opt_aoa_error- mse_GPS_error)/norm(mse_GPS_error))
 
if (norm(mse_ATC_error) < norm(mse_GPS_error)):    
    print ('MSE ATC reduction: ', norm(mse_ATC_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE ATC increment: ', norm(mse_ATC_error- mse_GPS_error)/norm(mse_GPS_error))

if (norm(mse_ATCME_error) < norm(mse_GPS_error)):    
    print ('MSE ATCME reduction: ', norm(mse_ATCME_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE ATCME increment: ', norm(mse_ATCME_error- mse_GPS_error)/norm(mse_GPS_error))
    
if (norm(mse_ATCCG_error) < norm(mse_GPS_error)):    
    print ('MSE ATCCG reduction: ', norm(mse_ATCCG_error- mse_GPS_error)/norm(mse_GPS_error), '\n')

else:
    print ('MSE ATCCG increment: ', norm(mse_ATCCG_error- mse_GPS_error)/norm(mse_GPS_error), '\n')

if (norm(max_lapl_error) < norm(max_GPS_error)):
    print ('Max Laplacian reduction: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error))
    
else:
    print ('Max Laplacian increment: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error))


if (norm(max_distr_lapl_error) < norm(max_GPS_error)):    
    print ('Max Distr Laplacian reduction: ', norm(max_distr_lapl_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('Max Distr Laplacian increment: ', norm(max_distr_lapl_error- max_GPS_error)/norm(max_GPS_error))    

    
if (norm(max_opt_aoa_error) < norm(max_GPS_error)):    
    print ('Max Opt reduction: ', norm(max_opt_aoa_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('Max Opt increment: ', norm(max_opt_aoa_error- max_GPS_error)/norm(max_GPS_error))
    
if (norm(max_ATC_error) < norm(max_GPS_error)):    
    print ('Max ATC reduction: ', norm(max_ATC_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('Max ATC increment: ', norm(max_ATC_error- max_GPS_error)/norm(max_GPS_error))
    
if (norm(max_ATCME_error) < norm(max_GPS_error)):    
    print ('Max ATCME reduction: ', norm(max_ATCME_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('Max ATCME increment: ', norm(max_ATCME_error- max_GPS_error)/norm(max_GPS_error))

if (norm(max_ATCCG_error) < norm(max_GPS_error)):    
    print ('Max ATCCG reduction: ', norm(max_ATCCG_error- max_GPS_error)/norm(max_GPS_error), '\n')

else:
    print ('Max ATCCG increment: ', norm(max_ATCCG_error- max_GPS_error)/norm(max_GPS_error), '\n')
    

print ('Radius: ', np.min(np.max(D, axis = 1)))
print ('Diameter: ', np.max(np.max(D, axis = 1)))
print ('Connectivity: ', eig[1])
print ('Sum of eigenvalues: ', np.sum(eig))
print ('Last distance: ', norm(np.array([true_X[0][0]-true_X[0][-1], true_Y[0][0]-true_Y[0][-1]]),2))

print ('Mean time of ATC: ', np.mean(array_of_time_ATC))
print ('Mean time of ATCME: ', np.mean(array_of_time_ATCME))
print ('Mean time of ATCCG: ', np.mean(array_of_time_ATCCG))

vehicle_index = 0

vehicle_index_2 = 1

plt_9.figure(9)

for j in range (num_of_iter-1):
    #plt_9.plot(np.array([traj_X[vehicle_index,j],traj_X[vehicle_index,j+1]]),np.array([traj_Y[vehicle_index,j],traj_Y[vehicle_index,j+1]]), 'rs-')
    #plt_9.plot(np.array([recon_X[vehicle_index,j],recon_X[vehicle_index,j+1]]),np.array([recon_Y[vehicle_index,j],recon_Y[vehicle_index,j+1]]), 'b*-')
    plt_9.plot(np.array([true_X[vehicle_index,j],true_X[vehicle_index,j+1]]),np.array([true_Y[vehicle_index,j],true_Y[vehicle_index,j+1]]), 'go-', fillstyle='none')
    #plt_9.plot(np.array([true_X[vehicle_index_2,j],true_X[vehicle_index_2,j+1]]),np.array([true_Y[vehicle_index_2,j],true_Y[vehicle_index_2,j+1]]), 'ro-', fillstyle='none')
    
    plt_9.plot(np.array([KM_true_X[vehicle_index,j],KM_true_X[vehicle_index,j+1]]),np.array([KM_true_Y[vehicle_index,j],KM_true_Y[vehicle_index,j+1]]), 'm<-', fillstyle='none')
    #plt_9.plot(np.array([KM_true_X[vehicle_index_2,j],KM_true_X[vehicle_index_2,j+1]]),np.array([KM_true_Y[vehicle_index_2,j],KM_true_Y[vehicle_index_2,j+1]]), 'c<-', fillstyle='none')
    #plt_9.plot(np.array([true_X[vehicle_index_2+1,j],true_X[vehicle_index_2+1,j+1]]),np.array([true_Y[vehicle_index_2+1,j],true_Y[vehicle_index_2+1,j+1]]), 'bo-', fillstyle='none')
    
  
#plt_9.legend(('Veh 1', 'Veh 2', 'Veh 1-KM', 'Veh 2-KM'), facecolor='white', fontsize = 30 )
plt_9.legend(('Veh 1', 'Veh 1-KM'), facecolor='white', fontsize = 30 )
plt_9.xlabel('x-axis', fontsize = 35)
plt_9.ylabel('y-axis', fontsize = 35) 
plt_9.xticks(fontsize=28)
plt_9.yticks(fontsize=28)
plt_9.grid(b=True)
plt_9.title('True trajectories', fontsize = 35)   

plt_11.figure(11)
for i in range (A[:,0].size):
    for j in range(A[0,:].size):
        if (A[i,j] == 1):
            plt_11.plot(Cartesian_Points[i,0], Cartesian_Points[i,1], 'ko', markersize = 10)
            plt_11.plot(np.array([Cartesian_Points[i,0], Cartesian_Points[j,0]]), np.array([Cartesian_Points[i,1], Cartesian_Points[j,1]]), 'r-', linewidth=3.0, markersize = 10)
            plt_11.annotate(str(i), (Cartesian_Points[i,0], Cartesian_Points[i,1]), fontsize=20)
            #plt_11.plot(np.array([noisy_Cartesian[i,0], noisy_Cartesian[j,0]]), np.array([noisy_Cartesian[i,1], noisy_Cartesian[j,1]]), 'gd-', fillstyle='none', linewidth=3.0, markersize = 12)
            #plt_11.annotate(str(i), (noisy_Cartesian[i,0], noisy_Cartesian[i,1]), fontsize=20)
        else:
            plt_11.plot(Cartesian_Points[i,0], Cartesian_Points[i,1], 'ko', markersize = 10)

plt_11.xlabel('x-axis', fontsize = 35) 
plt_11.ylabel('y-axis', fontsize = 35)              
plt_11.title('VANET Graph at time instant t = ' + str(v), fontsize = 35) 
plt_11.xticks(fontsize=28)
plt_11.yticks(fontsize=28)
plt_11.grid(b=True)
#plt_11.tick_params(direction='out', length= 28)
#plt_11.legend(('True VANET', 'Noisy VANET'),facecolor='white', fontsize = 27 )
plt_11.show() 

sorted_x_mse_GPS_error = np.sort(mse_GPS_error)
sorted_x_mse_lapl_error = np.sort(mse_lapl_error)
sorted_x_mse_opt_aoa_error = np.sort(mse_opt_aoa_error)
sorted_x_mse_distr_lapl_error = np.sort(mse_distr_lapl_error)
sorted_x_mse_ATC_error = np.sort(mse_ATC_error)
sorted_x_mse_ATCME_error = np.sort(mse_ATCME_error)
sorted_x_mse_ATCCG_error = np.sort(mse_ATCCG_error)

sorted_y_mse_GPS_error = np.arange(len(np.sort(mse_GPS_error)))/float(len(mse_GPS_error))
sorted_y_mse_lapl_error = np.arange(len(np.sort(mse_lapl_error)))/float(len(mse_lapl_error))
sorted_y_mse_opt_aoa_error = np.arange(len(np.sort(mse_opt_aoa_error)))/float(len(mse_opt_aoa_error))
sorted_y_mse_distr_lapl_error = np.arange(len(np.sort(mse_distr_lapl_error)))/float(len(mse_distr_lapl_error))
sorted_y_mse_ATC_error = np.arange(len(np.sort(mse_ATC_error)))/float(len(mse_ATC_error))
sorted_y_mse_ATCME_error = np.arange(len(np.sort(mse_ATCME_error)))/float(len(mse_ATCME_error))
sorted_y_mse_ATCCG_error = np.arange(len(np.sort(mse_ATCCG_error)))/float(len(mse_ATCCG_error))

plt_12.figure(12)
plt_12.plot(sorted_x_mse_GPS_error, sorted_y_mse_GPS_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
#plt_12.plot(sorted_x_mse_opt_aoa_error, sorted_y_mse_opt_aoa_error, 'c*-',  label="TCL-MLE", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_distr_lapl_error, sorted_y_mse_distr_lapl_error, 'm*-', label="DGCL", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_ATC_error, sorted_y_mse_ATC_error, 'y*-', label="ATC", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_ATCME_error, sorted_y_mse_ATCME_error, 'g*-', label="ATCME", linewidth = 4, markersize = 6)
plt_12.plot(sorted_x_mse_ATCCG_error, sorted_y_mse_ATCCG_error, 'k*-', label="ATCCG", linewidth = 4, markersize = 6)
plt_12.xticks(fontsize=28)
plt_12.yticks(fontsize=28)
plt_12.tick_params(direction='out', length=8)
plt_12.grid(b=True)
plt_12.legend(facecolor='white', fontsize = 37 )
plt_12.xlabel('Localization Mean Square Error [m$^2$]', fontsize = 35)
plt_12.ylabel('CDF', fontsize = 35)
plt_12.show()


sorted_x_max_GPS_error = np.sort(max_GPS_error)
sorted_x_max_lapl_error = np.sort(max_lapl_error)
sorted_x_max_opt_aoa_error = np.sort(max_opt_aoa_error)
sorted_x_max_distr_lapl_error = np.sort(max_distr_lapl_error)
sorted_x_max_ATC_error = np.sort(max_ATC_error)
sorted_x_max_ATCME_error = np.sort(max_ATCME_error)
sorted_x_max_ATCCG_error = np.sort(max_ATCCG_error)

sorted_y_max_GPS_error = np.arange(len(np.sort(max_GPS_error)))/float(len(max_GPS_error))
sorted_y_max_lapl_error = np.arange(len(np.sort(max_lapl_error)))/float(len(max_lapl_error))
sorted_y_max_opt_aoa_error = np.arange(len(np.sort(max_opt_aoa_error)))/float(len(max_opt_aoa_error))
sorted_y_max_distr_lapl_error = np.arange(len(np.sort(max_distr_lapl_error)))/float(len(max_distr_lapl_error))
sorted_y_max_ATC_error = np.arange(len(np.sort(max_ATC_error)))/float(len(max_ATC_error))
sorted_y_max_ATCME_error = np.arange(len(np.sort(max_ATCME_error)))/float(len(max_ATCME_error))
sorted_y_max_ATCCG_error = np.arange(len(np.sort(max_ATCCG_error)))/float(len(max_ATCCG_error))

plt_13.figure(13)
plt_13.plot(sorted_x_max_GPS_error, sorted_y_max_GPS_error, 'r*-',  label="GPS", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b*-', label="CGCL", linewidth = 4, markersize = 6)
#plt_13.plot(sorted_x_max_opt_aoa_error, sorted_y_max_opt_aoa_error, 'c*-',  label="TCL-MLE", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_distr_lapl_error, sorted_y_max_distr_lapl_error, 'm*-', label="DGCL", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_ATC_error, sorted_y_max_ATC_error, 'y*-', label="ATC", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_ATCME_error, sorted_y_max_ATCME_error, 'g*-', label="ATCME", linewidth = 4, markersize = 6)
plt_13.plot(sorted_x_max_ATCCG_error, sorted_y_max_ATCCG_error, 'k*-', label="ATCCG", linewidth = 4, markersize = 6)
plt_13.xticks(fontsize=28)
plt_13.yticks(fontsize=28)
plt_13.tick_params(direction='out', length=8)
plt_13.grid(b=True)
plt_13.legend(facecolor='white', fontsize = 37 )
plt_13.xlabel('Localization Maximum Absolute Error [m]', fontsize = 35)
plt_13.ylabel('CDF', fontsize = 35)
plt_13.show()

plt_1.figure(1)
plt_1.plot(np.arange(Kmax_LMS), LMS_average_MSD, 'b*-',  label="LMS", linewidth = 4, markersize = 6)
plt_1.plot(np.arange(Kmax_RLS), RLS_average_MSD, 'r*-',  label="CG", linewidth = 4, markersize = 6)
plt_1.plot(np.arange(max(Kmax_LMS,Kmax_RLS)), np.mean(lapl_MSD)*(np.ones(max(Kmax_LMS,Kmax_RLS))), 'm*-', label="CGCL", linewidth = 4, markersize = 6)
plt_1.xticks(fontsize=28)
plt_1.yticks(fontsize=28)
plt_1.tick_params(direction='out', length=8)
plt_1.grid(b=True)
plt_1.legend(facecolor='white', fontsize = 37 )
plt_1.xlabel('Iterations', fontsize = 35)
plt_1.ylabel('Avg. MSD', fontsize = 35)
plt_1.show()

plt_2.figure(2)
#plt_2.plot(np.arange(Kmax_LMS), LMS_average_MSD, 'y*-',  label="Global LMS", linewidth = 4, markersize = 6)
plt_2.plot(np.arange(Kmax_ATC), ATC_average_MSD, 'y*-',  label="ATC", linewidth = 4, markersize = 6)
plt_2.plot(np.arange(Kmax_ATCME), ATCME_average_MSD, 'go-', label="ATCME", linewidth = 4, markersize = 6)
plt_2.plot(np.arange(Kmax_ATCCG), ATCCG_average_MSD, 'k<-', label="ATCCG", linewidth = 4, markersize = 6)
plt_2.plot(np.arange(Kmax_ATC), np.mean(lapl_MSD)*(np.ones(Kmax_ATC)), 'b-', label="CGCL", linewidth = 4, markersize = 6)
plt_2.plot(np.arange(Kmax_ATC), np.mean(local_lapl_MSD)*(np.ones(Kmax_ATC)), 'm-', label="DGCL", linewidth = 4, markersize = 6)
#plt_2.plot(np.arange(max(Kmax_CTA,Kmax_ATC)), np.mean(lapl_MSD)*(np.ones(max(Kmax_CTA,Kmax_ATC))), 'm*-', label="CGCL", linewidth = 4, markersize = 6)
#plt_2.plot(np.arange(max(Kmax_CTA,Kmax_ATC)), np.mean(local_lapl_MSD)*(np.ones(max(Kmax_CTA,Kmax_ATC))), 'g*-', label="DGCL", linewidth = 4, markersize = 6)
plt_2.xticks(fontsize=28)
plt_2.yticks(fontsize=28)
plt_2.tick_params(direction='out', length=8)
plt_2.grid(b=True)
plt_2.legend(facecolor='white', fontsize = 37 )
plt_2.xlabel('Iterations', fontsize = 35)
plt_2.ylabel('Avg. MSD (dB)', fontsize = 35)
plt_2.show()
plt_2.title('Tracking mode', fontsize = 35)

plt_3.figure(3)
#plt_3.plot(np.arange(Kmax_LMS), LMS_average_MSD, 'y*-',  label="Global LMS", linewidth = 4, markersize = 6)
plt_3.plot(np.arange(Kmax_ATC), ATC_average_MSD_wtr, 'y*-',  label="ATC", linewidth = 4, markersize = 6)
plt_3.plot(np.arange(Kmax_ATCME), ATCME_average_MSD_wtr, 'go-', label="ATCME", linewidth = 4, markersize = 6)
plt_3.plot(np.arange(Kmax_ATCCG), ATCCG_average_MSD_wtr, 'k<-', label="ATCCG", linewidth = 4, markersize = 6)
plt_3.plot(np.arange(Kmax_ATC), np.mean(lapl_MSD)*(np.ones(Kmax_ATC)), 'b-', label="CGCL", linewidth = 4, markersize = 6)
plt_3.plot(np.arange(Kmax_ATC), np.mean(local_lapl_MSD)*(np.ones(Kmax_ATC)), 'm-', label="DGCL", linewidth = 4, markersize = 6)
#plt_2.plot(np.arange(max(Kmax_CTA,Kmax_ATC)), np.mean(lapl_MSD)*(np.ones(max(Kmax_CTA,Kmax_ATC))), 'm*-', label="CGCL", linewidth = 4, markersize = 6)
#plt_2.plot(np.arange(max(Kmax_CTA,Kmax_ATC)), np.mean(local_lapl_MSD)*(np.ones(max(Kmax_CTA,Kmax_ATC))), 'g*-', label="DGCL", linewidth = 4, markersize = 6)
plt_3.xticks(fontsize=28)
plt_3.yticks(fontsize=28)
plt_3.tick_params(direction='out', length=8)
plt_3.grid(b=True)
plt_3.legend(facecolor='white', fontsize = 37 )
plt_3.xlabel('Iterations', fontsize = 35)
plt_3.ylabel('Avg. MSD (dB)', fontsize = 35)
plt_3.show()
plt_3.title('No tracking mode', fontsize = 35)


num_of_time_instances = 20

#num_of_iter

MSD_ATC_vector = np.reshape(ATC_temp_MSD[:,:num_of_time_instances], Kmax_ATC*num_of_time_instances, order = 'F')

MSD_ATCME_vector = np.reshape(ATCME_temp_MSD[:,:num_of_time_instances], Kmax_ATCME*num_of_time_instances, order = 'F')

MSD_ATCCG_vector = np.reshape(ATCCG_temp_MSD[:,:num_of_time_instances], Kmax_ATCCG*num_of_time_instances, order = 'F')

MSD_lapl_vector = np.zeros(Kmax_ATC*num_of_time_instances)

start = 0
stop = Kmax_ATC
for i in range (num_of_time_instances):
    
    MSD_lapl_vector[start:stop] = lapl_MSD[i]
    
    start = stop
    
    stop += Kmax_ATC
    
plt_4.figure(4)
plt_4.plot(np.arange(Kmax_ATC*num_of_time_instances), MSD_ATC_vector, 'y*-',  label="ATC", linewidth = 4, markersize = 6)
plt_4.plot(np.arange(Kmax_ATCME*num_of_time_instances), MSD_ATCME_vector, 'g*-',  label="ATCME", linewidth = 4, markersize = 6)
plt_4.plot(np.arange(Kmax_ATCCG*num_of_time_instances), MSD_ATCCG_vector, 'k*-',  label="ATCCG", linewidth = 4, markersize = 6)
plt_4.plot(np.arange(Kmax_ATC*num_of_time_instances), MSD_lapl_vector, 'b*-', label="CGCL", linewidth = 4, markersize = 9)
plt_4.xticks(fontsize=28)
plt_4.yticks(fontsize=28)
plt_4.tick_params(direction='out', length=8)
plt_4.grid(b=True)
plt_4.legend(facecolor='white', fontsize = 37 )
#plt_4.xlabel('Iterations', fontsize = 35)
plt_4.ylabel('Avg. MSD', fontsize = 35)
#plt_4.title('Time instant ' + str(random_t))
plt_4.show()

'''
plt_14.figure(14)
plt_14.plot(np.arange(num_of_iter), GPS_Vo_vector, 'r*--',  label="GPS")
plt_14.plot(np.arange(num_of_iter), Lapl_Vo_vector, 'b*--', label="Lapl")
plt_14.xticks(fontsize=28)
plt_14.yticks(fontsize=28)
plt_14.tick_params(direction='out', length=8)
plt_14.grid(b=True)
plt_14.legend(facecolor='white', fontsize = 37 )
plt_14.xlabel('Time instances', fontsize = 35)
plt_14.ylabel('Error', fontsize = 35)
plt_14.show()
'''
'''
plt_14.figure(14)
plt_14.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b-')
#plt_14.plot(sorted_x_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], sorted_y_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], 'bo')
plt_14.plot(sorted_x_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], sorted_y_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], 'b*')
#plt_14.plot(sorted_x_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], sorted_y_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], 'bo')
plt_14.plot(sorted_x_mse_opt_aoa_error, sorted_y_mse_opt_aoa_error, 'c-')
#plt_14.plot(sorted_x_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], sorted_y_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], 'co')
plt_14.plot(sorted_x_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], sorted_y_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], 'c*')
#plt_14.plot(sorted_x_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], sorted_y_mse_opt_aoa_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], 'co')
plt_14.plot(sorted_x_mse_distr_lapl_error, sorted_y_mse_distr_lapl_error, 'm-')
#plt_14.plot(sorted_x_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], sorted_y_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.78)], 'mo')
plt_14.plot(sorted_x_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], sorted_y_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], 'm*')
#plt_14.plot(sorted_x_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], sorted_y_mse_distr_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.82)], 'mo')
plt_14.xticks(fontsize=28)
plt_14.yticks(fontsize=28)
plt_14.tick_params(direction='out', length=8)
plt_14.grid(b=True)
plt_14.show()

plt_15.figure(15)
plt_15.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b-')
#plt_15.plot(sorted_x_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], sorted_y_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], 'bo')
plt_15.plot(sorted_x_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], sorted_y_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], 'b*')
#plt_15.plot(sorted_x_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], sorted_y_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], 'bo')
plt_15.plot(sorted_x_max_opt_aoa_error, sorted_y_max_opt_aoa_error, 'c-')
#plt_15.plot(sorted_x_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], sorted_y_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], 'co')
plt_15.plot(sorted_x_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], sorted_y_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], 'c*')
#plt_15.plot(sorted_x_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], sorted_y_max_opt_aoa_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], 'co')
plt_15.plot(sorted_x_max_distr_lapl_error, sorted_y_max_distr_lapl_error, 'm-')
#plt_15.plot(sorted_x_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], sorted_y_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.78)], 'mo')
plt_15.plot(sorted_x_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], sorted_y_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], 'm*')
#plt_15.plot(sorted_x_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], sorted_y_max_distr_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.82)], 'mo')
plt_15.xticks(fontsize=28)
plt_15.yticks(fontsize=28)
plt_15.tick_params(direction='out', length=8)
plt_15.grid(b=True)
plt_15.show()
'''
'''
data_3 = np.load('Diff_Results/3_mse_ATCCG_error.npy')

data_7 = np.load('Diff_Results/7_mse_ATCCG_error.npy')

data_10 = np.load('Diff_Results/10_mse_ATCCG_error.npy')

data_15 = np.load('Diff_Results/15_mse_ATCCG_error.npy')

sorted_x_3_mse_ATCCG_error = np.sort(data_3)

sorted_x_7_mse_ATCCG_error = np.sort(data_7)

sorted_x_10_mse_ATCCG_error = np.sort(data_10)

sorted_x_15_mse_ATCCG_error = np.sort(data_15)

sorted_y_3_mse_ATCCG_error = np.arange(len(np.sort(data_3)))/float(len(data_3))

sorted_y_7_mse_ATCCG_error = np.arange(len(np.sort(data_7)))/float(len(data_7))

sorted_y_10_mse_ATCCG_error = np.arange(len(np.sort(data_10)))/float(len(data_10))

sorted_y_15_mse_ATCCG_error = np.arange(len(np.sort(data_15)))/float(len(data_15))

plt_10.figure(10)
plt_10.plot(sorted_x_3_mse_ATCCG_error, sorted_y_3_mse_ATCCG_error, 'k*-', label="ATCCG-3 Veh", linewidth = 4, markersize = 6)
plt_10.plot(sorted_x_7_mse_ATCCG_error, sorted_y_7_mse_ATCCG_error, 'r*-', label="ATCCG-7 Veh", linewidth = 4, markersize = 6)
plt_10.plot(sorted_x_10_mse_ATCCG_error, sorted_y_10_mse_ATCCG_error, 'b*-', label="ATCCG-10 Veh", linewidth = 4, markersize = 6)
plt_10.plot(sorted_x_15_mse_ATCCG_error, sorted_y_15_mse_ATCCG_error, 'm*-', label="ATCCG-15 Veh", linewidth = 4, markersize = 6)
plt_10.xticks(fontsize=28)
plt_10.yticks(fontsize=28)
plt_10.tick_params(direction='out', length=8)
plt_10.grid(b=True)
plt_10.legend(facecolor='white', fontsize = 37 )
plt_10.xlabel('Localization Mean Square Error [m$^2$]', fontsize = 35)
plt_10.ylabel('CDF', fontsize = 35)
plt_10.show()
'''