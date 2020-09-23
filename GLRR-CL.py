import numpy as np
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
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.sparse.linalg import lsmr,svds
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes
from sklearn.cluster import KMeans

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



def CalculateDist_EUKL (x_1, x_2, y_1, y_2):
       
    return(math.sqrt((x_1 - x_2)**2 + (y_1-y_2)**2))


def CalculateAzimuthAngle (x_observer, x_target, y_observer, y_target):
    
    #flag = 0
    
    if (x_observer == x_target or y_observer == y_target):
        return 0
    
    if (x_target >= x_observer and y_target >= y_observer):
        
        a = (math.atan((x_target-x_observer)/(y_target-y_observer)))
        flag_1 = 0
        flag_2 = math.radians(0)
        
    elif (x_target >= x_observer and y_target <= y_observer):
        
        a = math.radians(90) + (math.atan((y_observer-y_target)/(x_target-x_observer)))
        #flag = math.radians(90)
        flag_1 = 1
        flag_2 = math.radians(90)
        
    elif (x_target <= x_observer and y_target <= y_observer):
        
        a = math.radians(180) + (math.atan((x_observer-x_target)/(y_observer-y_target)))
        #flag = math.radians(180)
        flag_1 = 2
        flag_2 = math.radians(180)
        
    elif (x_target <= x_observer and y_target >= y_observer):
        
        a = math.radians(270) + (math.atan((y_target-y_observer)/(x_observer-x_target)))
        #flag = math.radians(270)
        flag_1 = 3
        flag_2 = math.radians(270)
    
    return a, flag_1, flag_2

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

    Ident_Matrix = np.identity((L_bar[0,:].size))
    
    for i in range (Ident_Matrix[:,0].size):
        for j in range (Ident_Matrix[0,:].size):
            Ident_Matrix[i,j] *= 0.001
    
    
    if (np.linalg.matrix_rank(L_bar.T@L_bar) < (L_bar.T@L_bar)[0,:].size):
        print ('Problem with rank of L_bar')
        
    X = lsmr(L_bar,b)[0]
    Y = lsmr(L_bar,q)[0]

   
    Points = np.zeros((X.size, 3))

    Points[:,0] = X
    Points[:,1] = Y

    reconstr_error = np.zeros((X.size))

    for i in range(reconstr_error.size):
        #reconstr_error[i] = CalculateDist_EUKL(Cartesian_Points[i,0],X[i], Cartesian_Points[i,1],Y[i])
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


def Optimization_Laplacian(L, A, anchors, anchors_index, delta_X, delta_Y, z_d,z_c_x,z_c_y,z_a,a_flag, a_flag_2):
    
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


    x = cvx.Variable(shape = A[:,0].size)
    y = cvx.Variable(shape = A[:,0].size)
    
    f_1 = 0
    f_2 = 0
    f_3 = 0 
    
    for i in range(z_c_x.size):
        f_1 += (z_c_x[i] - x[i])**2 + (z_c_y[i] - y[i])**2
        
    for i in range(z_d[:,0].size):
        for j in range(z_d[0,:].size):
            if (A[i,j] == 1):
                
                f_2 += cvx.power(cvx.pos(-z_d[i,j] + (cvx.norm(cvx.vstack( [x[i] - x[j], y[i] - y[j] ] ),2))),2)
                
                if (a_flag[i,j] == 1):
                        
                    if (a_flag_2[i,j] == 0):
                            
                        f_3 += (z_a[i,j]*(x[i] - x[j]) - (y[j] - y[i]))**2
                                
                    else:
                        
                        f_3 += (z_a[i,j]*(x[j] - x[i]) - (y[i] - y[j]))**2
                            
                else:
                        
                    if (a_flag_2[i,j] == 0):
                            
                        f_3 += (z_a[i,j]*(y[j] - y[i]) - (x[j] - x[i]))**2
                                
                    else:
                            
                        f_3 += (z_a[i,j]*(y[i] - y[j]) - (x[i] - x[j]))**2
                         
    opt_prob = cvx.Problem(cvx.Minimize(f_1 + f_2 + f_3 + cvx.sum_squares(L_bar@x - b) + cvx.sum_squares(L_bar@y - q)))
    opt_prob.solve()
    
    return x.value, y.value

def Optimization_AoA_GD (A, noisy_D, tan_noisy_AoA, noise_X, noise_Y, deg_noisy_AoA):
    
    delta = 0.001
    
    low_a = 70
    high_a = 110
    
    low_a_2 = 250
    high_a_2 = 290
    
    num_of_iter_GD = 300
    
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
            
            temp_x = np.zeros(index.size   )   
            temp_y = np.zeros(index.size   )
            
            #temp_x = inv(L_local.T@L_local)@L_local.T@b
            #temp_y = inv(L_local.T@L_local)@L_local.T@q
            
            temp_x = lsmr(L_local,b)[0]
            temp_y = lsmr(L_local,q)[0]
            
            #print ('temp_x:', temp_x,'\n')
            #print ('temp_y:', temp_y,'\n')
            x_final[k,l] = temp_x[index.size-1]
            y_final[k,l] = temp_y[index.size-1]
       
    return x_final[:,num_of_iter_distr-1], y_final[:,num_of_iter_distr-1]
            

def shrink_operator(Matrix, v):
    '''
    #start = timer()
    for i in range (Matrix[:,0].size):
        for j in range (Matrix[0,:].size):
           
            Matrix[i,j] = np.sign(Matrix[i,j])*max(abs(Matrix[i,j]) - v, 0)
    
    #end = timer()
    
    #print (end-start)
    '''
    Zeros_Matrix = np.zeros((Matrix.shape[0], Matrix.shape[1]))
    
    return np.sign(Matrix)*np.maximum(abs(Matrix) - v, Zeros_Matrix)

def RPCA(Ex, Ey, num_of_iter):
    
    lamda = 1/(math.sqrt(max(Ex[:,0].size, num_of_iter)))

    #lamda = 0.1
    
    mhi = 0.0001
    
    RPCA_iter = 300
    
    Yx = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yx = Ex / max(norm(Ex), norm((lamda**(-1))*Ex, np.inf))
    
    Nx = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yy = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yy = Ey / max(norm(Ey), norm((lamda**(-1))*Ey, np.inf))
    
    Ny = np.zeros((Ex[:,0].size, num_of_iter))
    
    #time_array = np.zeros(RPCA_iter)
    
    for i in range (RPCA_iter):
        
        Ux, Sigmax, Vhx = svd((Ex - Nx + (mhi**(-1))*Yx), full_matrices=False)
        
        Sx = np.copy(Ux@shrink_operator(np.eye(Sigmax.size)*Sigmax, (mhi**(-1)))@Vhx)
        
        Nx = np.copy(shrink_operator(Ex - Sx + (mhi**(-1))*Yx ,lamda*(mhi**(-1))))
        
        Yx = np.copy(Yx + mhi*(Ex - Sx - Nx))
        
        Uy, Sigmay, Vhy = svd((Ey - Ny + (mhi**(-1))*Yy), full_matrices=False)
        
        Sy = np.copy(Uy@shrink_operator(np.eye(Sigmay.size)*Sigmay, (mhi**(-1)))@Vhy)
        
        Ny = np.copy(shrink_operator(Ey - Sy + (mhi**(-1))*Yy ,lamda*(mhi**(-1))))
        
        Yy = np.copy(Yy + mhi*(Ey - Sy - Ny))
        
        mhi = mhi*1.01
        
    #print (np.sum(time_array))
        
    '''    
    Yy = np.zeros((Ex[:,0].size, num_of_iter))
    
    Yy = Ey / max(norm(Ey), norm((lamda**(-1))*Ey, np.inf))
    
    Ny = np.zeros((Ex[:,0].size, num_of_iter))
    
    for i in range (RPCA_iter):
        
        Uy, Sigmay, Vhy = svd((Ey - Ny + (mhi**(-1))*Yy), full_matrices=False)
        
        Sy = np.copy(Uy@shrink_operator(np.eye(Sigmay.size)*Sigmay, (mhi**(-1)))@Vhy)
        
        Ny = np.copy(shrink_operator(Ey - Sy + (mhi**(-1))*Yy ,lamda*(mhi**(-1))))
        
        Yy = np.copy(Yy + mhi*(Ey - Sy - Ny))
        
        mhi = mhi*1.01      
        
     '''   
    return Sx, Sy, Nx, Ny

def LRMR(B_x, B_y, u, s, vh, rank):
    
    
    #num_of_LRMR_iter = 100
    
    #residual = []
    
    target_x = np.copy(B_x)
    target_y = np.copy(B_y)
    
    i = 0

    while (True):
        
        W_X = u.T@target_x
        
        D_r_X = np.eye(s.size)
        
        D_r_X = D_r_X*s
        
        u_W_X, s_W_X, vh_W_X = svd(W_X,  full_matrices=False)
        
        
        a = np.sum(s_W_X[:rank])/np.sum(s_W_X)
        #print (a,'\n')
        #print (np.min(s_W_X[1:])/np.max(s_W_X[1:]))
        s_W_X[rank:] = 0
        
        Theta_matrix_X = vh.T@(inv(D_r_X)@u_W_X@(np.eye(s_W_X.size)*s_W_X)@vh_W_X)
       
        W_Y = u.T@target_y
        
        D_r_Y = np.eye(s.size)
        
        D_r_Y = D_r_Y*s
        
        u_W_Y, s_W_Y, vh_W_Y = svd(W_Y,  full_matrices=False)
        
        b = np.sum(s_W_Y[:rank])/np.sum(s_W_Y)
        #print (b,'\n')
        #print (np.min(s_W_Y[1:])/np.max(s_W_Y[1:]))
        s_W_Y[rank:] = 0
        
        Theta_matrix_Y = vh.T@(inv(D_r_Y)@u_W_Y@(np.eye(s_W_Y.size)*s_W_Y)@vh_W_Y)
        
        i += 1
        if (i == 1):
            break;
    ''' 
    print (norm(target_x[int(target_x.shape[0]/2):,:], 'nuc'))
    print (norm(Theta_matrix_X), 'nuc')
    '''
    return Theta_matrix_X, Theta_matrix_Y, a, b

def IncSVD(U, Sigma, Vh, d):
    
    x = U.T@d
    
    zx = d - U@x
    
    rhox = norm(zx)
    
    p = (1/rhox)*zx
    
    
    #print (x)
    
    #print (Sigma)
    
    temp_matrix_1 = np.zeros((Sigma[:,0].size, Sigma[:,0].size+1))
    
    temp_matrix_1[:, :Sigma[:,0].size] = Sigma
    
    temp_matrix_1[:, Sigma[:,0].size] = x
    
    #print (np.zeros(Sigma[:,0].size).T)
    #print (rhox)
    
    #temp_matrix_2 = np.concatenate((np.zeros(p.size).T, rhox), axis = 1)
    
    temp_matrix_2 = np.zeros(Sigma[:,0].size+1)
    
    temp_matrix_2[:Sigma[:,0].size] = np.zeros(Sigma[:,0].size)
    temp_matrix_2[Sigma[:,0].size] = rhox
    
    #print (temp_matrix_2)
    #middle_matrix = np.concatenate((temp_matrix_1, temp_matrix_2))
    
    middle_matrix = np.zeros((Sigma[:,0].size+1, Sigma[:,0].size+1))
    
    middle_matrix[:Sigma[:,0].size] = temp_matrix_1
    middle_matrix[Sigma[:,0].size] = temp_matrix_2
    
    G, sigma_bar, Hh = svd(middle_matrix,  full_matrices=False)
    
    print (G, '\n')
    Sigma_bar = sigma_bar*np.eye(sigma_bar.size)
    
    #print (G, '\n')
    #print (Sigma_bar, '\n')
    #print (Hh, '\n')
    
    U_1 = U@G[:Sigma[:,0].size, :Sigma[:,0].size] + p[:Sigma[:,0].size]@G[Sigma[:,0].size, :Sigma[:,0].size]
    
    #print (U_1)
    Sigma_1 = Sigma_bar[:Sigma[:,0].size, :Sigma[:,0].size]
    #print (Hh[Sigma[:,0].size, :Sigma[:,0].size].T)
    
    V_1 = np.zeros((Vh.T[:,0].size+1, Sigma[:,0].size))
    
    V_1[:Vh.T[:,0].size, :] = Vh.T@Hh[:Sigma[:,0].size, :Sigma[:,0].size].T
    V_1[Vh.T[:,0].size, :] = Hh[Sigma[:,0].size, :Sigma[:,0].size].T
    
    #print (V_1)
    #V_1 = np.concatenate((Vh.T@Hh[:Sigma[:,0].size, :Sigma[:,0].size].T, Hh[Sigma[:,0].size, :Sigma[:,0].size].T), axis = 0)
    
    return U_1, Sigma_1, V_1

def PowerSVD(target_matrix):

    U = np.zeros((target_matrix.shape[0], target_matrix.shape[1]))

    Sigma = np.zeros((target_matrix.shape[1], target_matrix.shape[1]))

    Vh = np.zeros((target_matrix.shape[1], target_matrix.shape[1]))
    
    for j in range (target_matrix.shape[1]):
        
        i = 0
        
        vh_temp = 0.01*np.ones(target_matrix[0,:].size)
        
        while (i < 10):
            #start = vh_temp
            
            u_temp = np.copy(target_matrix@vh_temp / norm(target_matrix@vh_temp))
            
            vh_temp = np.copy(target_matrix.T@u_temp / norm(target_matrix.T@u_temp))
            
            s_temp = np.copy(norm(target_matrix.T@u_temp))
            
            #end = np.copy(vh_temp)
            
            #print (norm(end - start))
            i += 1
            
        U[:,j] = np.copy(u_temp)
        Sigma[j,j] = np.copy(s_temp)
        Vh[j,:] = np.copy(vh_temp)
        
        target_matrix = np.copy(target_matrix - s_temp*u_temp.reshape(-1,1)@vh_temp.reshape(-1,1).T)
        
    return U, Sigma, Vh

def NNM(L_bar, B_x, B_y):
    
    temp_Sx = cvx.Variable(shape = (L_bar[0,:].size, B_x[0,:].size))
    
    temp_Sy = cvx.Variable(shape = (L_bar[0,:].size, B_y[0,:].size))



    opt_prob_1 = cvx.Problem(cvx.Minimize(cvx.norm(temp_Sx, 'nuc') )
                             
                    , [cvx.sum_squares(L_bar@temp_Sx - B_x) <= 100]
                     #norm(L_bar@recon_X - B_x)**2
                             #,[L_bar[:int(L_bar.shape[0]/2),:]@temp_Sx == B_x[:int(B_x.shape[0]/2),:]]
                     )
                     
        
    opt_prob_1.solve()
    
    opt_prob_2 = cvx.Problem(cvx.Minimize(cvx.norm(temp_Sy, 'nuc'))
                    
                             , [cvx.sum_squares(L_bar@temp_Sy - B_y) <= 100]
                             #norm(L_bar@recon_Y - B_y)**2
                             #,[L_bar[:int(L_bar.shape[0]/2),:]@temp_Sy == B_y[:int(B_y.shape[0]/2),:]]
                    )
        
    opt_prob_2.solve()
    
    '''
    #print (matrix_rank(temp_Sx.value))
    #print (norm(L_bar@temp_Sx.value - B_x))
    print (norm(B_x[int(B_x.shape[0]/2):,:], 'nuc'))
    s = svd(temp_Sx.value, full_matrices = False)[1]
    print (norm(temp_Sx.value, 'nuc'))
    print (np.sum( s  ))
    '''
    return temp_Sx.value, temp_Sy.value

#Nx = temp_Nx.value
#Ny = temp_Ny.value
#mu, sigma_x, sigma_y = 0, 0.00001, 0.00001
#mu, sigma_x, sigma_y = 0, 7.49, 2.93 # CEP = 6m, HDOP = 1.2, UERE = 6.7
#mu, sigma_x, sigma_y = 0, 0.16, 4.68 # CEP = 3m, HDOP = 0.7, UERE = 6.7


    
num_of_iter = 500
num_of_vehicles = 20

random_time_index = int(random.uniform(0,num_of_iter-1))


mean_GPS_error = np.zeros(num_of_iter)

mse_GPS_error = np.zeros(num_of_iter)
mse_lapl_error = np.zeros(num_of_iter)
mse_local_lapl_error = np.zeros(num_of_iter)
mse_lapl_tracking_error = np.zeros(num_of_iter)
mse_local_lapl_tracking_error = np.zeros(num_of_iter)
mse_lapl_rpca_error = np.zeros(num_of_iter)
mse_local_lapl_rpca_error = np.zeros(num_of_iter)
mse_lapl_error_outliers = np.zeros(num_of_iter)

max_GPS_error = np.zeros(num_of_iter)
max_lapl_error = np.zeros(num_of_iter)
max_local_lapl_error = np.zeros(num_of_iter)
max_lapl_tracking_error = np.zeros(num_of_iter)
max_local_lapl_tracking_error = np.zeros(num_of_iter)
max_lapl_rpca_error = np.zeros(num_of_iter)
max_local_lapl_rpca_error = np.zeros(num_of_iter)
max_lapl_error_outliers = np.zeros(num_of_iter)

Cartesian_Points = np.zeros((num_of_vehicles,3))    
    
for i in range(Cartesian_Points[:,0].size):
    Cartesian_Points[i,0] = random.uniform(0, 4*Cartesian_Points[:,0].size)
    Cartesian_Points[i,1] = random.uniform(0, 12)


noisy_Cartesian = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
    
gps_noise_x = np.zeros((Cartesian_Points[:,0].size))
gps_noise_y = np.zeros((Cartesian_Points[:,0].size))

sigma_x = 3
sigma_y = 2.5

sigma_d = 1
sigma_a = 4

gps_noise_x = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[0][:Cartesian_Points[:,0].size]
gps_noise_y = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[1][:Cartesian_Points[:,1].size]

#0.45/math.sqrt(2)
#7.49, 2.93
#10, 10
#2, 1.5

noisy_Cartesian[:,0] = Cartesian_Points[:,0] + gps_noise_x
noisy_Cartesian[:,1] = Cartesian_Points[:,1] + gps_noise_y
noisy_Cartesian[:,2] = Cartesian_Points[:,2] 


Dt = 0.1 
tire_diam = 0.3

theta = np.zeros((Cartesian_Points[:,0].size, num_of_iter)) 

for i in range(theta.shape[0]):
    theta[i,0] = random.uniform(math.radians(5), math.radians(10))

speed = np.zeros(Cartesian_Points[:,0].size)
speed = np.random.uniform(28,29,Cartesian_Points[:,0].size)

traj_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter)) 
traj_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

recon_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter)) 
recon_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

local_recon_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter)) 
local_recon_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

true_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter)) 
true_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
           
low_a = 70
high_a = 110

low_a_2 = 250
high_a_2 = 290

L_full = np.zeros((num_of_iter, Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))

B_x = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))
B_y = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))

local_B_x = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))
local_B_y = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))

temp_B_x = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))
temp_B_y = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))

temp_local_B_x = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))
temp_local_B_y = np.zeros((2*Cartesian_Points[:,0].size, num_of_iter))

attack_vector_x = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
attack_vector_y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

true_outliers_label = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

TP = np.zeros(num_of_iter)
TN = np.zeros(num_of_iter)
FP = np.zeros(num_of_iter)
FN = np.zeros(num_of_iter)

TPR = np.zeros(num_of_iter)
FPR = np.zeros(num_of_iter)

Acc = np.zeros(num_of_iter)

array_of_L_rank = np.zeros(num_of_iter)

Sx = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
Nx = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

Sy = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
Ny = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

local_Sx = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
local_Nx = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

local_Sy = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
local_Ny = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

Theta_matrix_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
Theta_matrix_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

local_Theta_matrix_X = np.zeros((Cartesian_Points[:,0].size, num_of_iter))
local_Theta_matrix_Y = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

window_range = 9

ratios_x, ratios_y  = np.zeros(num_of_iter), np.zeros(num_of_iter)
for v in range(num_of_iter):
    
    
    print ('Number of vehicles is: ' , Cartesian_Points[:,0].size)
    
    Deg = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    
    D, a = np.zeros((Cartesian_Points[:,0].size,Cartesian_Points[:,0].size)), np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    AoA = np.zeros((Cartesian_Points[:,0].size,Cartesian_Points[:,0].size))
    
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            D[i,j] = norm(np.array([Cartesian_Points[i][0]-Cartesian_Points[j][0], Cartesian_Points[i][1]-Cartesian_Points[j][1]]),2)
            
    
    noise_for_distances = np.zeros(Cartesian_Points[:,0].size**2)
    noise_for_distances = np.random.normal(0, sigma_d, Cartesian_Points[:,0].size**2) 
    noise_for_distances = np.reshape(noise_for_distances, (Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    for i in range (Cartesian_Points[:,0].size):
        for j in range (Cartesian_Points[:,0].size):
            if (i>j):
                noise_for_distances[i,j] = noise_for_distances[j,i]
                
    noisy_D = np.zeros((D[:,0].size,D[0,:].size))
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (D[i,j] + noise_for_distances[i,j] >= 0):
                noisy_D[i,j] = D[i,j] + noise_for_distances[i,j]
            else:
                noisy_D[i,j] = D[i,j] + abs(noise_for_distances[i,j])
            
            #np.random.normal(0,sigma_d,size = 1)
    
    if (v == 0):
        flag_start = 1
    else:
        flag_start = 0
        
    if (flag_start == 1):
    #if (v % window_range == 0):
        A = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
        for i in range(D[:,0].size):
            for j in range(D[0,:].size):
                if (i!=j):
                    if (D[i,j] <= 20):
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
                    if (count > 6):
                        #if (np.sum(A[j,:]) >= 3):
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
                a[i,j] = CalculateAzimuthAngle(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])[0]
                
                AoA[i,j] = CalculateAoA(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])[0]
                
            else:
                D[i,j] = 0
                noisy_D[i,j] = 0
            if (noisy_D[i,j] < 0 ):
                print ('Bad')
           
    list_for_del = []               
    for i in range (A[:,0].size):
        if (Deg[i,i] == 0):
            print ('Index for delete is: ' + str(i),'\n')
            list_for_del.append(i)
    
    
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
        speed = np.delete(speed,array_for_del,0)
        traj_X = np.delete(traj_X,array_for_del,0)
        traj_Y = np.delete(traj_Y,array_for_del,0)
        recon_X = np.delete(recon_X,array_for_del,0)
        recon_Y = np.delete(recon_Y,array_for_del,0)
        local_recon_X = np.delete(recon_X,array_for_del,0)
        local_recon_Y = np.delete(recon_Y,array_for_del,0)
        true_X = np.delete(true_X,array_for_del,0)
        true_Y = np.delete(true_Y,array_for_del,0)
        theta = np.delete(theta,array_for_del,0)
        Deg = np.delete(Deg, array_for_del, 1)
        A = np.delete(A, array_for_del, 1)
        D = np.delete(D, array_for_del, 1)
        noisy_D = np.delete(noisy_D, array_for_del, 1)
        a = np.delete(a, array_for_del, 1)
        AoA = np.delete(AoA, array_for_del, 1)
        B_x = np.delete(B_x, array_for_del, 0)
        B_y = np.delete(B_x, array_for_del, 0)
        L_full = np.delete(L_full, array_for_del, 1)
        L_full = np.delete(L_full, array_for_del, 2)
        print ('Shape of Degree matrix is: ' , np.shape(Deg),'\n')
    
    
    #print (np.array_equal(A,A.T))
    #L = np.zeros((A[:,0].size,A[0,:].size))
        
    L = Deg - A
    
    L_full[v,:,:] = L
    
    array_of_L_rank[v] = matrix_rank(L)
    
    delta_X = np.zeros((Cartesian_Points[:,0].size))
    delta_Y = np.zeros((Cartesian_Points[:,0].size))
    
    
    
    GPS_error = np.zeros((Cartesian_Points[:,0].size))
    for i in range(GPS_error.size):
        GPS_error[i] = norm(np.array([Cartesian_Points[i,0] - noisy_Cartesian[i,0] , Cartesian_Points[i,1] - noisy_Cartesian[i,1]]), 2)
    
    
    noisy_a = np.zeros((a[:,0].size,a[0,:].size))
    
    noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    
    deg_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    deg_true_AoA = np.zeros((a[:,0].size,a[0,:].size))
    tan_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
    temp_tan_noisy_AoA = np.zeros((a[:,0].size,a[0,:].size))
   
    noise_for_angles = np.zeros(Cartesian_Points[:,0].size**2)
    noise_for_angles = np.random.normal(0, math.radians(sigma_a), Cartesian_Points[:,0].size**2) 
    noise_for_angles = np.reshape(noise_for_angles, (Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    for i in range (Cartesian_Points[:,0].size):
        for j in range (Cartesian_Points[:,0].size):
            if (i>j):
                noise_for_angles[i,j] = noise_for_angles[j,i]
                
                
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (A[i,j] == 1):
                noisy_a[i,j] = a[i,j] + noise_for_angles[i,j]
                
                #np.random.normal(0,math.radians(sigma_a),size = 1)
                noisy_AoA[i,j] = AoA[i,j] + noise_for_angles[i,j]
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
                    
    for i in range(noisy_D[:,0].size):
        for j in range(noisy_D[0,:].size):
            if (A[i,j] == 1):
                delta_X[i] += -noisy_D[i,j]*math.sin(noisy_a[i,j])
                delta_Y[i] += -noisy_D[i,j]*math.cos(noisy_a[i,j])
                    
       
    
    list_random_outliers_idx=[]
    
    number_of_outliers = int(round(0.0*Cartesian_Points[:,0].size))
    
    for i in range(number_of_outliers):
            
        r=random.randint(0, Cartesian_Points[:,0].size-1)
        while (r in list_random_outliers_idx):
            r=random.randint(0, Cartesian_Points[:,0].size-1)
        list_random_outliers_idx.append(r)
    
        
    for i in range (len(list_random_outliers_idx)):
        
        true_outliers_label[list_random_outliers_idx[i],v] = 1
    
    for i in range (len(list_random_outliers_idx)):
        attack_vector_x[list_random_outliers_idx[i],v] = random.uniform(5,40)
        attack_vector_y[list_random_outliers_idx[i],v] = random.uniform(5,40)
        
        
    for i in range (len(list_random_outliers_idx)):
    
        noisy_Cartesian[list_random_outliers_idx[i],0] += attack_vector_x[list_random_outliers_idx[i],v] 
        noisy_Cartesian[list_random_outliers_idx[i],1] += attack_vector_y[list_random_outliers_idx[i],v]
    

    
    #########!!!!!!! ONLY LAPLACIAN !!!!!!!!!############
    start_lapl = timer()
    
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
    
    #anchors_index = np.asarray(list_random_anchors_idx)
    #anchors = np.asarray(list_random_anchors)
            
    anchors_index = np.arange(Cartesian_Points[:,0].size)
    anchors = np.copy(noisy_Cartesian[:,:2])
    
    Points_lapl = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
    
    lapl_error = np.zeros((Cartesian_Points[:,0].size))
    
    L_bar = np.zeros((Cartesian_Points[:,0].size + anchors_index.size, Cartesian_Points[0,:].size))
    
    Points_lapl, lapl_error, L_bar = Solve_The_System(Cartesian_Points, L, anchors_index, anchors, delta_X, delta_Y)
    
    local_Points_lapl = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
    '''
    #local_Points_lapl[:,0], local_Points_lapl[:,1] = Distributed_Lapl(Deg, A, delta_X, delta_Y, noisy_Cartesian[:,0], noisy_Cartesian[:,1], noisy_D, noisy_a)
    
    local_Points_lapl[:,0], local_Points_lapl[:,1] = Optimization_AoA_GD(A, noisy_D, tan_noisy_AoA, noisy_Cartesian[:,0], noisy_Cartesian[:,1], deg_noisy_AoA)
    
    local_lapl_error = np.zeros(Cartesian_Points[:,0].size)
    for i in range (Points_lapl[:,0].size):
        local_lapl_error[i] = norm(np.array([local_Points_lapl[i,0] - Cartesian_Points[i,0], local_Points_lapl[i,1] - Cartesian_Points[i,1]]),2)  
    '''
    
    #noisy_Cartesian
    #Points_lapl
    j = 0
    for i in range (2*Cartesian_Points[:,0].size):
        if (i <= Cartesian_Points[:,0].size-1):
            B_x[i,v] = delta_X[i]
            B_y[i,v] = delta_Y[i]
            '''
            local_B_x[i,v] = delta_X[i]
            local_B_y[i,v] = delta_Y[i]
            
            temp_B_x[i,v] = delta_X[i]
            temp_B_y[i,v] = delta_Y[i]
            temp_local_B_x[i,v] = delta_X[i]
            temp_local_B_y[i,v] = delta_Y[i]
            '''
        else:
            B_x[i,v] = Points_lapl[j,0]
            B_y[i,v] = Points_lapl[j,1]
            '''
            local_B_x[i,v] = local_Points_lapl[j,0]
            local_B_y[i,v] = local_Points_lapl[j,1]
            
            temp_B_x[i,v] = Points_lapl[j,0]
            temp_B_y[i,v] = Points_lapl[j,1]
            temp_local_B_x[i,v] = local_Points_lapl[j,0]
            temp_local_B_y[i,v] = local_Points_lapl[j,1]
            '''
            j += 1
            
            
    end_lapl = timer()  
    
    #########!!!!!!! ONLY LAPLACIAN !!!!!!!!!############
    
    print ('Iteration: ', v, '\n')
    
    mse_GPS_error[v] = ((np.sum(np.power(GPS_error,2))/Cartesian_Points[:,0].size))
    mse_lapl_error[v] = ((np.sum(np.power(lapl_error,2))/Cartesian_Points[:,0].size))
    #mse_local_lapl_error[v] = ((np.sum(np.power(local_lapl_error,2))/Cartesian_Points[:,0].size))
    max_GPS_error[v] = np.max(GPS_error)
    max_lapl_error[v] = np.max(lapl_error)
    #max_local_lapl_error[v] = np.max(local_lapl_error)
    
    traj_X[:,v] = noisy_Cartesian[:,0]
    traj_Y[:,v] = noisy_Cartesian[:,1]
    
    recon_X[:,v] = Points_lapl[:,0]
    recon_Y[:,v] = Points_lapl[:,1]
    '''
    local_recon_X[:,v] = local_Points_lapl[:,0]
    local_recon_Y[:,v] = local_Points_lapl[:,1]
    '''
    true_X[:,v] = Cartesian_Points[:,0]
    true_Y[:,v] = Cartesian_Points[:,1]
    
    if (v == 0):
        u, s, vh = svd(L_bar, full_matrices=False)
        
    if (v <  window_range):
       
        Theta_matrix_X[:,v] = recon_X[:,v]
        Theta_matrix_Y[:,v] = recon_Y[:,v]
        '''
        local_Theta_matrix_X[:,v] = local_recon_X[:,v]
        local_Theta_matrix_Y[:,v] = local_recon_Y[:,v]
        '''
    else:
        
        temp_theta_matrix_x = np.zeros((Cartesian_Points[:,0].size,  window_range+1))
        temp_theta_matrix_y = np.zeros((Cartesian_Points[:,0].size,  window_range+1))
        
        temp_theta_matrix_x, temp_theta_matrix_y, ratios_x[v], ratios_y[v] = LRMR(B_x[:,v-window_range:v+1], B_y[:,v-window_range:v+1], u, s, vh, rank = 3)
        
        #print (matrix_rank(true_X[:,v-window_range:v+1]), '\n')
        
        #temp_theta_matrix_x, temp_theta_matrix_y = NNM(L_bar, B_x[:,v-window_range:v+1], B_y[:,v-window_range:v+1])
        
        Theta_matrix_X[:,v] = np.copy(temp_theta_matrix_x[:,-1])
        Theta_matrix_Y[:,v] = np.copy(temp_theta_matrix_y[:,-1])
        
        #B_x[Cartesian_Points[:,0].size:,:-1] = np.copy(Theta_matrix_X[:,:-1])
        #B_y[Cartesian_Points[:,0].size:,:-1] = np.copy(Theta_matrix_Y[:,:-1])
        '''
        local_temp_theta_matrix_x = np.zeros((Cartesian_Points[:,0].size,  window_range+1))
        local_temp_theta_matrix_y = np.zeros((Cartesian_Points[:,0].size,  window_range+1))
        
        #local_temp_theta_matrix_x, local_temp_theta_matrix_y = LRMR(local_B_x[:,v-window_range:v+1], local_B_y[:,v-window_range:v+1], u, s, vh, rank = 2)
        
        local_temp_theta_matrix_x, local_temp_theta_matrix_y = NNM(L_bar, B_x[:,v-window_range:v+1], B_y[:,v-window_range:v+1])
        
        local_Theta_matrix_X[:,v] = np.copy(local_temp_theta_matrix_x[:,-1])
        local_Theta_matrix_Y[:,v] = np.copy(local_temp_theta_matrix_y[:,-1])
        
        #local_B_x[Cartesian_Points[:,0].size:,:-1] = np.copy(local_Theta_matrix_X[:,:-1])
        #local_B_y[Cartesian_Points[:,0].size:,:-1] = np.copy(local_Theta_matrix_Y[:,:-1])
        '''
        
    if (v < num_of_iter-1):
        '''
        if ((Dt*(v+1) % 15) == 0):
            speed = np.zeros(Cartesian_Points[:,0].size)
            speed = np.random.uniform(31,33,Cartesian_Points[:,0].size)
            print ('Time for speed change: ', Dt*(v+1),'\n')
        '''
        if ((Dt*(v+1) % 1) == 0):
            speed = np.zeros(Cartesian_Points[:,0].size)
            speed = np.random.uniform(28,29,Cartesian_Points[:,0].size)
            print ('Time for speed change: ', Dt*(v+1),'\n')
        
        for i in range(Cartesian_Points[:,0].size):
            
            yaw_rate = speed[i]/tire_diam
            '''    
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
            Cartesian_Points[i,0] +=  (-speed[i]/yaw_rate)*math.sin(theta[i,v]) + (speed[i]/yaw_rate)*math.sin(theta[i,v] + yaw_rate*Dt)
            Cartesian_Points[i,1] +=  (speed[i]/yaw_rate)*math.cos(theta[i,v])  + (-speed[i]/yaw_rate)*math.cos(theta[i,v] + yaw_rate*Dt)
            
            theta[i,v] = theta[i,v] + Dt*yaw_rate
            
        noisy_Cartesian = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[0,:].size))
        
        gps_noise_x = np.zeros((Cartesian_Points[:,0].size))
        gps_noise_y = np.zeros((Cartesian_Points[:,0].size))
     
        
        gps_noise_x = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[0][:Cartesian_Points[:,0].size]
        gps_noise_y = GPS_noise(0, sigma_x, sigma_y, Cartesian_Points[:,0].size)[1][:Cartesian_Points[:,1].size]
        
        noisy_Cartesian[:,0] = Cartesian_Points[:,0] + gps_noise_x
        noisy_Cartesian[:,1] = Cartesian_Points[:,1] + gps_noise_y
        noisy_Cartesian[:,2] = Cartesian_Points[:,2]
     
    if (array_of_L_rank[v] < Cartesian_Points[:,0].size - 1):
        print ('Not connected')
        break;
        

'''
Batch_Theta_matrix_X, Batch_Theta_matrix_Y = LRMR(temp_B_x, temp_B_y, u, s, vh, rank = matrix_rank(true_X))

Batch_local_Theta_matrix_X, Batch_local_Theta_matrix_Y = LRMR(temp_local_B_x, temp_local_B_y, u, s, vh, rank = matrix_rank(true_X))
'''

for i in range(num_of_iter):
    
    error_lapl_tracking = np.zeros(Cartesian_Points[:,0].size)
    
    error_local_lapl_tracking = np.zeros(Cartesian_Points[:,0].size)
    
    for j in range (Cartesian_Points[:,0].size):
        error_lapl_tracking[j] = norm(np.array([ true_X[j,i] - Theta_matrix_X[j,i], true_Y[j,i] - Theta_matrix_Y[j,i] ]), 2)
        error_local_lapl_tracking[j] = norm(np.array([ true_X[j,i] - local_Theta_matrix_X[j,i], true_Y[j,i] - local_Theta_matrix_Y[j,i] ]), 2)
        
    mse_lapl_tracking_error[i] =  ((np.sum(np.power(error_lapl_tracking,2))/Cartesian_Points[:,0].size)) 
    max_lapl_tracking_error[i] =  np.max(error_lapl_tracking)
    
    mse_local_lapl_tracking_error[i] =  ((np.sum(np.power(error_local_lapl_tracking,2))/Cartesian_Points[:,0].size)) 
    max_local_lapl_tracking_error[i] =  np.max(error_local_lapl_tracking)


'''
Ex = Batch_Theta_matrix_X
Ey = Batch_Theta_matrix_Y

Sx, Sy, Nx, Ny = RPCA(Ex, Ey, num_of_iter)


local_Ex = Batch_local_Theta_matrix_X
local_Ey = Batch_local_Theta_matrix_Y

local_Sx, local_Sy, local_Nx, local_Ny = RPCA(local_Ex, local_Ey, num_of_iter)
'''
'''
uo, so, vho = svd(L_bar, full_matrices = False)

Uo = np.zeros((L_bar.shape[0], L_bar.shape[1]))

Sigmao = np.zeros((L_bar.shape[1], L_bar.shape[1]))

Vho = np.zeros((L_bar.shape[1], L_bar.shape[1]))

target_matrix = np.copy(L_bar)

for j in range (L_bar.shape[1]):
    i = 0
    
    vho_temp = 0.01*np.ones(L_bar[0,:].size)
    
    while (i < 20):
        start = vho_temp
        
        uo_temp = np.copy(target_matrix@vho_temp / norm(target_matrix@vho_temp))
        
        vho_temp = np.copy(target_matrix.T@uo_temp / norm(target_matrix.T@uo_temp))
        
        so_temp = np.copy(norm(target_matrix.T@uo_temp))
        
        end = np.copy(vho_temp)
        
        print (norm(end - start))
        i += 1
        
    Uo[:,j] = np.copy(uo_temp)
    Sigmao[j,j] = np.copy(so_temp)
    Vho[j,:] = np.copy(vho_temp)
    
    target_matrix = np.copy(target_matrix - so_temp*uo_temp.reshape(-1,1)@vho_temp.reshape(-1,1).T)
    
'''
'''
U_o, s_o, V_oh = svd(recon_X[:,:5],  full_matrices=False)
U_1, Sigma_1, V_1 = IncSVD(U_o, s_o*np.eye(s_o.size), V_oh, recon_X[:,5])

U_2, s_2, V_2h = svd(recon_X[:,:6],  full_matrices=False)

print (U_1 @ Sigma_1 @ V_1.T)
'''
'''
difference = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

array_of_outliers = np.zeros((Cartesian_Points[:,0].size, num_of_iter))

for i in range (num_of_iter):
    
    for j in range (Cartesian_Points[:,0].size):
        
        difference[j,i] = norm(np.array([ traj_X[j,i] - Sx[j,i], traj_Y[j,i] - Sy[j,i] ]), 2)
        
        if (difference[j,i] <= 10):
            difference[j,i] = 0
    
    if (np.sum(difference[:,i]) != 0):   
        
        kmeans = KMeans(n_clusters=2).fit(difference[:,i].reshape(-1,1))
            
        centers = np.zeros(1)
            
        centers[0] = np.argwhere(kmeans.cluster_centers_ == np.max(kmeans.cluster_centers_))[0][0]
           
        for j in range(difference[:,i].size):
                
            if (kmeans.labels_[j] == centers[0]): 
                array_of_outliers[j,i] = 1


for i in range (num_of_iter):
    
    for j in range (Cartesian_Points[:,0].size):
        
        if (true_outliers_label[j,i] == 1 and array_of_outliers[j,i] == 1):
            TP[i] += 1
                
        elif (true_outliers_label[j,i] == 0 and array_of_outliers[j,i] == 0):
            TN[i] += 1
                
        elif (true_outliers_label[j,i] == 1 and array_of_outliers[j,i] == 0):
            FN[i] += 1
                
        elif (true_outliers_label[j,i] == 0 and array_of_outliers[j,i] == 1):
            FP[i] += 1
            
    TP[i] /= Cartesian_Points[:,0].size
    TN[i] /= Cartesian_Points[:,0].size
    FN[i] /= Cartesian_Points[:,0].size
    FP[i] /= Cartesian_Points[:,0].size

        
    if (TP[i] > 0 or FN[i] > 0):
        TPR[i] = TP[i]/(TP[i] + FN[i])
    else:
        TPR[i] = 1
    if (FP[i] > 0 or TN[i] > 0):
        FPR[i] = FP[i]/(FP[i] + TN[i])
    else:
        FPR[i] = 1
         
    Acc[i] = (TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i])
    
for i in range(num_of_iter):
    
    error_lapl_rpca = np.zeros(Cartesian_Points[:,0].size)
    
    error_local_lapl_rpca = np.zeros(Cartesian_Points[:,0].size)
    
    for j in range (Cartesian_Points[:,0].size):
        error_lapl_rpca[j] = norm(np.array([ true_X[j,i] - Sx[j,i], true_Y[j,i] - Sy[j,i] ]), 2)
        error_local_lapl_rpca[j] = norm(np.array([ true_X[j,i] - local_Sx[j,i], true_Y[j,i] - local_Sy[j,i] ]), 2)
        
    mse_lapl_rpca_error[i] =  (np.sum(np.power(error_lapl_rpca,2))/Cartesian_Points[:,0].size)             
    max_lapl_rpca_error[i] =  np.max(error_lapl_rpca)
    
    mse_local_lapl_rpca_error[i] =  (np.sum(np.power(error_local_lapl_rpca,2))/Cartesian_Points[:,0].size)             
    max_local_lapl_rpca_error[i] =  np.max(error_local_lapl_rpca)

temp_traj_X = np.copy(traj_X)
temp_traj_Y = np.copy(traj_Y)

temp_true_X = np.copy(true_X)
temp_true_Y = np.copy(true_Y)

for i in range (num_of_iter):
    
    for j in range (traj_X[:,0].size):
        
        if (array_of_outliers[j,i] == 1):
            temp_traj_X[j,i] -= Nx[j,i]
            temp_traj_Y[j,i] -= Ny[j,i]
            
print ('Accuracy: ', np.mean(Acc), '\n')
'''
'''
Ex_outliers = np.copy(Ex - array_of_outliers*Nx)
Ey_outliers = np.copy(Ey - array_of_outliers*Ny)
    
Sx_outliers = RPCA(Ex_outliers, Ey_outliers)[0]
Sy_outliers = RPCA(Ex_outliers, Ey_outliers)[2]
    
for i in range (num_of_iter):
    
    lapl_error_outliers = np.zeros(Cartesian_Points[:,0].size)
     
    for j in range (lapl_error_outliers.size):
        lapl_error_outliers[j] = norm(np.array([ true_X[j,i] - Sx_outliers[j,i], true_Y[j,i] - Sy_outliers[j,i] ]), 2)
        
    mse_lapl_error_outliers[i] = ((np.sum(np.power(lapl_error_outliers,2))/traj_X[:,0].size))
    max_lapl_error_outliers[i] = np.max(lapl_error_outliers)
'''

vehicle_index = int(random.uniform(0,Cartesian_Points[:,0].size))

plt_9.figure(9)

for j in range (num_of_iter-1):
    #plt_9.plot(np.array([traj_X[vehicle_index,j],traj_X[vehicle_index,j+1]]),np.array([traj_Y[vehicle_index,j],traj_Y[vehicle_index,j+1]]), 'rs-')
    #plt_9.plot(np.array([recon_X[vehicle_index,j],recon_X[vehicle_index,j+1]]),np.array([recon_Y[vehicle_index,j],recon_Y[vehicle_index,j+1]]), 'b*-')
    plt_9.plot(np.array([true_X[vehicle_index,j],true_X[vehicle_index,j+1]]),np.array([true_Y[vehicle_index,j],true_Y[vehicle_index,j+1]]), 'go-', fillstyle='none')
    
    
#plt_9.legend(('Vehicle ' + str(vehicle_index) + ' noisy trajectory', 'Vehicle ' + str(vehicle_index) + ' reconstructed trajectory', 'Vehicle ' + str(vehicle_index) + ' true trajectory'))    
plt_9.xlabel('x-axis')
plt_9.ylabel('y-axis') 
plt_9.title('Trajectories')

plt_11.figure(11)
for i in range (A[:,0].size):
    for j in range(A[0,:].size):
        if (A[i,j] == 1):
            plt_11.plot(np.array([Cartesian_Points[i,0], Cartesian_Points[j,0]]), np.array([Cartesian_Points[i,1], Cartesian_Points[j,1]]), 'ro-')
            plt_11.annotate(str(i), (Cartesian_Points[i,0], Cartesian_Points[i,1]), fontsize=12)
plt_11.xlabel('x-axis') 
plt_11.ylabel('y-axis')              
plt_11.title('True VANET topology at time instant t = ' + str(v+1)) 
plt_11.show()
        
sorted_x_mse_GPS_error = np.sort(mse_GPS_error)
sorted_x_mse_lapl_error = np.sort(mse_lapl_error)
#sorted_x_mse_local_lapl_error = np.sort(mse_local_lapl_error)
sorted_x_mse_lapl_tracking_error = np.sort(mse_lapl_tracking_error)
'''
sorted_x_mse_local_lapl_tracking_error = np.sort(mse_local_lapl_tracking_error)
sorted_x_mse_lapl_rpca_error = np.sort(mse_lapl_rpca_error)
sorted_x_mse_local_lapl_rpca_error = np.sort(mse_local_lapl_rpca_error)
#sorted_x_mse_lapl_error_outliers = np.sort(mse_lapl_error_outliers)
'''

sorted_y_mse_GPS_error = np.arange(len(np.sort(mse_GPS_error)))/float(len(mse_GPS_error))
sorted_y_mse_lapl_error = np.arange(len(np.sort(mse_lapl_error)))/float(len(mse_lapl_error))
#sorted_y_mse_local_lapl_error = np.arange(len(np.sort(mse_local_lapl_error)))/float(len(mse_local_lapl_error))
sorted_y_mse_lapl_tracking_error = np.arange(len(np.sort(mse_lapl_tracking_error)))/float(len(mse_lapl_tracking_error))
'''
sorted_y_mse_local_lapl_tracking_error = np.arange(len(np.sort(mse_local_lapl_tracking_error)))/float(len(mse_local_lapl_tracking_error))
sorted_y_mse_lapl_rpca_error = np.arange(len(np.sort(mse_lapl_rpca_error)))/float(len(mse_lapl_rpca_error))
sorted_y_mse_local_lapl_rpca_error = np.arange(len(np.sort(mse_local_lapl_rpca_error)))/float(len(mse_local_lapl_rpca_error))
#sorted_y_mse_lapl_error_outliers = np.arange(len(np.sort(mse_lapl_error_outliers)))/float(len(mse_lapl_error_outliers))
'''

plt_12.figure(12)
plt_12.plot(sorted_x_mse_GPS_error, sorted_y_mse_GPS_error, 'r*--',  label="GPS")

plt_12.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b*--', label="C-GCL")
#plt_12.plot(sorted_x_mse_local_lapl_error, sorted_y_mse_local_lapl_error, 'g*--', label="GCL-L")
plt_12.plot(sorted_x_mse_lapl_tracking_error, sorted_y_mse_lapl_tracking_error, 'y*--', label="C-LRGCL")

'''
plt_12.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b*--', label="Second approach")
#plt_12.plot(sorted_x_mse_local_lapl_error, sorted_y_mse_local_lapl_error, 'g*--', label="First approach")
plt_12.plot(sorted_x_mse_lapl_tracking_error, sorted_y_mse_lapl_tracking_error, 'y*--', label="Third approach")
#plt_12.plot(sorted_x_mse_local_lapl_tracking_error, sorted_y_mse_local_lapl_tracking_error, 'c*--', label="GCL-L Tracking")
#plt_12.plot(sorted_x_mse_lapl_rpca_error, sorted_y_mse_lapl_rpca_error, 'm*--', label="GCL-C RPCA")
#plt_12.plot(sorted_x_mse_local_lapl_rpca_error, sorted_y_mse_local_lapl_rpca_error, 'k*--', label="GCL-L RPCA")
'''
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
#sorted_x_max_local_lapl_error = np.sort(max_local_lapl_error)
sorted_x_max_lapl_tracking_error = np.sort(max_lapl_tracking_error)
'''
sorted_x_max_local_lapl_tracking_error = np.sort(max_local_lapl_tracking_error)
sorted_x_max_lapl_rpca_error = np.sort(max_lapl_rpca_error)
sorted_x_max_local_lapl_rpca_error = np.sort(max_local_lapl_rpca_error)
#sorted_x_max_lapl_error_outliers = np.sort(max_lapl_error_outliers)
'''

sorted_y_max_GPS_error = np.arange(len(np.sort(max_GPS_error)))/float(len(max_GPS_error))
sorted_y_max_lapl_error = np.arange(len(np.sort(max_lapl_error)))/float(len(max_lapl_error))
#sorted_y_max_local_lapl_error = np.arange(len(np.sort(max_local_lapl_error)))/float(len(max_local_lapl_error))
sorted_y_max_lapl_tracking_error = np.arange(len(np.sort(max_lapl_tracking_error)))/float(len(max_lapl_tracking_error))
'''
sorted_y_max_local_lapl_tracking_error = np.arange(len(np.sort(max_local_lapl_tracking_error)))/float(len(max_local_lapl_tracking_error))
sorted_y_max_lapl_rpca_error = np.arange(len(np.sort(max_lapl_rpca_error)))/float(len(max_lapl_rpca_error))
sorted_y_max_local_lapl_rpca_error = np.arange(len(np.sort(max_local_lapl_rpca_error)))/float(len(max_local_lapl_rpca_error))
#sorted_y_max_lapl_error_outliers = np.arange(len(np.sort(max_lapl_error_outliers)))/float(len(max_lapl_error_outliers))
'''
fpr = np.zeros(3)
tpr = np.zeros(3)
 
plt_13.figure(13)
plt_13.plot(sorted_x_max_GPS_error, sorted_y_max_GPS_error, 'r*--',  label="GPS")
plt_13.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b*--', label="C-GCL")
#plt_13.plot(sorted_x_max_local_lapl_error, sorted_y_max_local_lapl_error, 'g*--', label="GCL-L")
plt_13.plot(sorted_x_max_lapl_tracking_error, sorted_y_max_lapl_tracking_error, 'y*--', label="C-LRGCL")
#plt_13.plot(sorted_x_max_local_lapl_tracking_error, sorted_y_max_local_lapl_tracking_error, 'c*--', label="GCL-L Tracking")
#plt_13.plot(sorted_x_max_lapl_rpca_error, sorted_y_max_lapl_rpca_error, 'm*--', label="GCL-C RPCA")
#plt_13.plot(sorted_x_max_local_lapl_rpca_error, sorted_y_max_local_lapl_rpca_error, 'k*--', label="GCL-L RPCA")
plt_13.xticks(fontsize=28)
plt_13.yticks(fontsize=28)
plt_13.tick_params(direction='out', length=8)
plt_13.grid(b=True)
plt_13.legend(facecolor='white', fontsize = 37 )
plt_13.xlabel('Maximum absolute localization error [m]', fontsize = 35)
plt_13.ylabel('CDF', fontsize = 35)
plt_13.show()
'''   
fpr[2] = 1
fpr[1] = np.mean(FPR)
    
tpr[2] = 1
tpr[1] = np.mean(TPR)
    
plt_14.figure(14)
plt_14.plot(fpr, tpr, 'ms-', label = 'k-means, RPCA')
plt_14.plot([0,1], [0,1], 'r--', linewidth=3.0)
plt_14.xticks(fontsize=28)
plt_14.yticks(fontsize=28)
plt_14.xlabel('False Positive Rate', fontsize = 35)
plt_14.ylabel('True Positive Rate', fontsize = 35)
plt_14.tick_params(direction='out', length=8)
plt_14.legend(facecolor='white', fontsize = 30)
plt_14.title('ROC Curve',fontsize = 35)
plt_14.show()

print ('AUC for RPCA is: ', np.trapz(tpr, fpr), '\n')
'''
if (norm(mse_lapl_error) < norm(mse_GPS_error)):
    print ('MSE GCL-C reduction: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error))
    
else:
    print ('MSE GCL-C increment: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error))

'''
if (norm(mse_local_lapl_error) < norm(mse_GPS_error)):
    print ('MSE GCL-L reduction: ', norm(mse_local_lapl_error- mse_GPS_error)/norm(mse_GPS_error))
    
else:
    print ('MSE GCL-L increment: ', norm(mse_local_lapl_error- mse_GPS_error)/norm(mse_GPS_error))
'''    
if (norm(mse_lapl_tracking_error) < norm(mse_GPS_error)):    
    print ('MSE GCL-C tracking reduction: ', norm(mse_lapl_tracking_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE GCL-C tracking increment: ', norm(mse_lapl_tracking_error- mse_GPS_error)/norm(mse_GPS_error))
'''    
if (norm(mse_local_lapl_tracking_error) < norm(mse_GPS_error)):    
    print ('MSE GCL-L tracking reduction: ', norm(mse_local_lapl_tracking_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE GCL-L tracking increment: ', norm(mse_local_lapl_tracking_error- mse_GPS_error)/norm(mse_GPS_error))
    
if (norm(mse_lapl_rpca_error) < norm(mse_GPS_error)):    
    print ('MSE GCL-C RPCA reduction: ', norm(mse_lapl_rpca_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE GCL-C RPCA increment: ', norm(mse_lapl_rpca_error- mse_GPS_error)/norm(mse_GPS_error))

if (norm(mse_local_lapl_rpca_error) < norm(mse_GPS_error)):    
    print ('MSE GCL-L RPCA reduction: ', norm(mse_local_lapl_rpca_error- mse_GPS_error)/norm(mse_GPS_error), '\n')

else:
    print ('MSE GCL-L RPCA increment: ', norm(mse_local_lapl_rpca_error- mse_GPS_error)/norm(mse_GPS_error), '\n')
'''

if (norm(max_lapl_error) < norm(max_GPS_error)):
    print ('MAX GCL-C reduction: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error))
    
else:
    print ('MAX GCL-C increment: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error))
'''    
if (norm(max_local_lapl_error) < norm(max_GPS_error)):
    print ('MAX GCL-L reduction: ', norm(max_local_lapl_error- max_GPS_error)/norm(max_GPS_error))
    
else:
    print ('MAX GCL-L increment: ', norm(max_local_lapl_error- max_GPS_error)/norm(max_GPS_error))
'''    
if (norm(max_lapl_tracking_error) < norm(max_GPS_error)):    
    print ('MAX GCL-C tracking reduction: ', norm(max_lapl_tracking_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('MAX GCL-C tracking increment: ', norm(max_lapl_tracking_error- max_GPS_error)/norm(max_GPS_error))
'''    
if (norm(max_local_lapl_tracking_error) < norm(max_GPS_error)):    
    print ('MAX GCL-L tracking reduction: ', norm(max_local_lapl_tracking_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('MAX GCL-L tracking increment: ', norm(max_local_lapl_tracking_error- max_GPS_error)/norm(max_GPS_error))

if (norm(max_lapl_rpca_error) < norm(max_GPS_error)):    
    print ('MAX GCL-C RPCA reduction: ', norm(max_lapl_rpca_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('MAX GCL-C RPCA increment: ', norm(max_lapl_rpca_error- max_GPS_error)/norm(max_GPS_error))
    
if (norm(max_local_lapl_rpca_error) < norm(max_GPS_error)):    
    print ('MAX GCL-L RPCA reduction: ', norm(max_local_lapl_rpca_error- max_GPS_error)/norm(max_GPS_error), '\n')

else:
    print ('MAX GCL-L RPCA increment: ', norm(max_local_lapl_rpca_error- max_GPS_error)/norm(max_GPS_error), '\n')
'''
'''
print ('Reduction of GCL-based scheme errors:','\n')
if (norm(mse_lapl_tracking_error) < norm(mse_lapl_error)):    
    print ('MSE tracking reduction: ', norm(mse_lapl_tracking_error- mse_lapl_error)/norm(mse_lapl_error))

else:
    print ('MSE tracking increment: ', norm(mse_lapl_tracking_error- mse_lapl_error)/norm(mse_lapl_error))
    

if (norm(mse_lapl_rpca_error) < norm(mse_lapl_error)):    
    print ('MSE RPCA reduction: ', norm(mse_lapl_rpca_error- mse_lapl_error)/norm(mse_lapl_error), '\n')

else:
    print ('MSE RPCA increment: ', norm(mse_lapl_rpca_error- mse_lapl_error)/norm(mse_lapl_error), '\n')
    
if (norm(max_lapl_tracking_error) < norm(max_lapl_error)):    
    print ('MAX tracking reduction: ', norm(max_lapl_tracking_error- max_lapl_error)/norm(max_lapl_error))

else:
    print ('MAX tracking increment: ', norm(max_lapl_tracking_error- max_lapl_error)/norm(max_lapl_error))
    

if (norm(max_lapl_rpca_error) < norm(max_lapl_error)):    
    print ('MAX RPCA reduction: ', norm(max_lapl_rpca_error- max_lapl_error)/norm(max_lapl_error), '\n')

else:
    print ('MAX RPCA increment: ', norm(max_lapl_rpca_error- max_lapl_error)/norm(max_lapl_error), '\n')    
'''    
print ('Rank of matrix of true locations:', matrix_rank(true_X),'\n')
print ('Rank of matrix of GPS locations:', matrix_rank(traj_X),'\n')
print ('Rank of L_bar matrix:', matrix_rank(L_bar),'\n')
print ('Rank of Theta_matrix:', matrix_rank(Theta_matrix_X),'\n')

print ('Ratio for x:', np.mean(ratios_x[9:]))

print ('Ratio for y:',np.mean(ratios_y[9:]))
'''
print ('Rank of S matrix:', matrix_rank(Sx),'\n')
print ('Rank of Local Theta_matrix:', matrix_rank(local_Theta_matrix_X),'\n')
print ('Rank of Local S matrix:', matrix_rank(local_Sx),'\n')
'''

'''
plt_14.figure(14)
plt_14.plot(sorted_x_mse_lapl_error, sorted_y_mse_lapl_error, 'b-')
plt_14.plot(sorted_x_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], sorted_y_mse_lapl_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], 'b*')
plt_14.plot(sorted_x_mse_lapl_tracking_error, sorted_y_mse_lapl_tracking_error, 'y-')
plt_14.plot(sorted_x_mse_lapl_tracking_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], sorted_y_mse_lapl_tracking_error[np.argwhere(sorted_y_mse_lapl_error == 0.8)], 'y*')
plt_14.xticks(fontsize=28)
plt_14.yticks(fontsize=28)
plt_14.tick_params(direction='out', length=8)
plt_14.grid(b=True)
plt_14.show()
'''
'''
plt_15.figure(15)
plt_15.plot(sorted_x_max_lapl_error, sorted_y_max_lapl_error, 'b-')
plt_15.plot(sorted_x_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], sorted_y_max_lapl_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], 'b*')
plt_15.plot(sorted_x_max_lapl_tracking_error, sorted_y_max_lapl_tracking_error, 'y-')
plt_15.plot(sorted_x_max_lapl_tracking_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], sorted_y_max_lapl_tracking_error[np.argwhere(sorted_y_max_lapl_error == 0.8)], 'y*')
plt_15.xticks(fontsize=28)
plt_15.yticks(fontsize=28)
plt_15.tick_params(direction='out', length=8)
plt_15.grid(b=True)
'''