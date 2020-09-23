import numpy as np
from timeit import default_timer as timer
#import matplotlib.pyplot as plt_hist_1
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
#import matplotlib.pyplot as plt_hist_2
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

import math
import random
from numpy.linalg import inv,cholesky,norm,matrix_rank,eig,svd,pinv
import cvxpy as cvx
from scipy.sparse.linalg import lsmr,svds
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial import ConvexHull, convex_hull_plot_2d

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
   
    if (x_observer == x_target or y_observer == y_target):
        return 0
    
    if (x_target >= x_observer and y_target >= y_observer):
        
        a = (math.atan((x_target-x_observer)/(y_target-y_observer)))
        
        
    elif (x_target >= x_observer and y_target <= y_observer):
        
        a = math.radians(90) + (math.atan((y_observer-y_target)/(x_target-x_observer)))
        
        
    elif (x_target <= x_observer and y_target <= y_observer):
        
        a = math.radians(180) + (math.atan((x_observer-x_target)/(y_observer-y_target)))
        
        
    elif (x_target <= x_observer and y_target >= y_observer):
        
        a = math.radians(270) + (math.atan((y_target-y_observer)/(x_observer-x_target)))
        
    
    return a

def CalculateAoA (x_observer, x_target, y_observer, y_target):
    
    
    
    a = (math.atan((y_target-y_observer)/(x_target-x_observer)))
    
    return a

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
        '''
        L_chol = cholesky(L_bar.T@L_bar + Ident_Matrix)
        ksi_x = np.zeros((L_chol.T[:,0].size,1))
        ksi_y = np.zeros((L_chol.T[:,0].size,1))
        ksi_x = inv(L_chol)@L_bar.T@b
        ksi_y = inv(L_chol)@L_bar.T@q
        X = inv(L_chol.T)@ksi_x
        Y = inv(L_chol.T)@ksi_y
        '''
        #X = inv(L_bar.T@L_bar + Ident_Matrix)@L_bar.T@b
        #Y = inv(L_bar.T@L_bar + Ident_Matrix)@L_bar.T@q
    #else:
        '''
        L_chol = cholesky(L_bar.T@L_bar)
        ksi_x = np.zeros((L_chol.T[:,0].size,1))
        ksi_y = np.zeros((L_chol.T[:,0].size,1))
        ksi_x = inv(L_chol)@L_bar.T@b
        ksi_y = inv(L_chol)@L_bar.T@q
        X = inv(L_chol.T)@ksi_x
        Y = inv(L_chol.T)@ksi_y
        '''
        #X = inv(L_bar.T@L_bar)@L_bar.T@b
        #Y = inv(L_bar.T@L_bar)@L_bar.T@q
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



def Optimization_AoA(A,z_d,z_c_x,z_c_y,z_a):
   
    y = cvx.Variable(shape = A[:,0].size)
    x = cvx.Variable(shape = A[:,0].size)
             
    f_1 = 0
    f_2 = 0
    f_3 = 0
    
    for i in range(z_c_x.size):
        f_1 += (((z_c_x[i]) - x[i])**2) + (((z_c_y[i]) - y[i])**2)
   
             
    for i in range(z_d[:,0].size):
        for j in range(z_d[0,:].size):
            if (A[i,j] == 1):
                            
                f_2 += (cvx.power(cvx.pos(-z_d[i,j] + (cvx.norm(cvx.vstack( [x[i] - x[j], y[i] - y[j] ] ),2))),2))
                
                     
                f_3 += ((z_a[i,j]*(x[j] - x[i]) - (y[j] - y[i]))**2)
                                
               
            
    opt_prob = cvx.Problem(cvx.Minimize(f_1 + f_2 + f_3))
    opt_prob.solve()
    return x.value, y.value
        

def Optimization_Outliers(A,z_d,z_c_x,z_c_y,z_a,L, anchors_index, delta_X, delta_Y):

    
    anchors = np.zeros((A[:,0].size, 3))
    
    for i in range (anchors[:,0].size):
        
        anchors[anchors_index[i],0] = z_c_x[anchors_index[i]]
        anchors[anchors_index[i],1] = z_c_y[anchors_index[i]]
    
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
    out_x = cvx.Variable(shape = b.size)    
    out_y = cvx.Variable(shape = b.size)
    
    
    opt_prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_bar@x - (b - out_x)) + cvx.sum_squares(L_bar@y - (q - out_y)) + 0.5*cvx.norm(out_x[A[:,0].size:],1) + 0.5*cvx.norm(out_y[A[:,0].size:],1) )
    
                           ,[out_x[:A[:,0].size] == np.zeros(A[:,0].size),
                            out_y[:A[:,0].size] == np.zeros(A[:,0].size),
                                   ]
                                   )
    
    
    #opt_prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_bar@x - (b )) + cvx.sum_squares(L_bar@y - (q ))))        
    opt_prob.solve()
    
    return x.value, y.value, out_x.value, out_y.value
    #return x.value, y.value, np.zeros(b.size), np.zeros(b.size)

def Optimization_Outliers_BS(A,z_d,z_c_x,z_c_y,z_a,L, anchors_index, delta_X, delta_Y):

    anchors = np.zeros((A[:,0].size, 3))
    
    for i in range (anchors[:,0].size):
        
        anchors[anchors_index[i],0] = z_c_x[anchors_index[i]]
        anchors[anchors_index[i],1] = z_c_y[anchors_index[i]]
    
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
    out_x = cvx.Variable(shape = b.size)    
    out_y = cvx.Variable(shape = b.size)
    
    g_x = 0
    g_y = 0
    
    for i in range (A[:,0].size):
        
        
        temp_A = np.zeros(A[:,0].size)
        
        temp_A = np.copy(A[i,:])
        
        temp_A[i] = 1
        
        '''
        sum_x_1 = 0
        sum_y_1 = 0
        #sum_x_2 = 0
        #sum_y_2 = 0
        
        for j in range (temp_A.size):
            
            if (temp_A[j] == 1):
                #print (j)
                #print ('\t')
                sum_x_1 += out_x[A[:,0].size + j]
                sum_y_1 += out_y[A[:,0].size + j]
                
                #sum_x_1 += b[A[:,0].size + j] - out_x[A[:,0].size + j]
                #sum_y_1 += q[A[:,0].size + j] - out_y[A[:,0].size + j]
                
                #sum_x_2 += out_x[A[:,0].size + j]
                #sum_y_2 += out_y[A[:,0].size + j]
        #print ('\n')   
        ''' 
        #(20**(-9))*
        g_x += (20**(-9))*cvx.norm(temp_A*(out_x[A[:,0].size:]) ,2)
        g_y += (20**(-9))*cvx.norm(temp_A*(out_y[A[:,0].size:]) ,2)
   
    opt_prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_bar@x - (b - out_x)) + cvx.sum_squares(L_bar@y - (q - out_y)) + g_x + g_y )
                           
                           ,[out_x[:A[:,0].size] == np.zeros(A[:,0].size),
                            out_y[:A[:,0].size] == np.zeros(A[:,0].size),
                           ]
                           )
    
    #opt_prob.solve(solver=cvx.OSQP)
    opt_prob.solve(solver=cvx.SCS)
    
    return x.value, y.value, out_x.value, out_y.value

def Optimization_Outliers_CF(A,z_d,z_c_x,z_c_y,z_a, deg_noisy_AoA):
    
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
    return x.value, y.value, out_x.value, out_y.value

def Optimization_Distr_Outliers(A,Deg, z_c_x,z_c_y,delta_X, delta_Y):

    x = np.zeros(z_c_x.size)
    y = np.zeros(z_c_x.size)
    
    out_x = np.zeros(z_c_x.size)
    out_y = np.zeros(z_c_x.size)
    
    
    for k in range (A[:,0].size):
        
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
            
             b[i] = z_c_x[index[i]]
             q[i] = z_c_y[index[i]]
                
                
        b[i+1] = delta_X[index[i+1]]
        q[i+1] = delta_Y[index[i+1]]  
            
        b[i+2] = z_c_x[k]
        q[i+2] = z_c_y[k]
        
        
        temp_x = cvx.Variable(shape = index.size)
        temp_y = cvx.Variable(shape = index.size)
        temp_out_x = cvx.Variable(shape = b.size)    
        temp_out_y = cvx.Variable(shape = b.size)
        
        
        g_x = 0
        g_y = 0
        
        for i in range (b.size):
            g_x += temp_out_x[i]
            g_y += temp_out_y[i]
        
        
        #0.5*cvx.norm(temp_out_x,1)
        opt_prob_1 = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_local@temp_x - (b - temp_out_x)) + 0.25*cvx.norm(g_x,1))
                               
                               ,[temp_out_x[index.size-1] == 0]
                               )
        
        opt_prob_1.solve()
        
        opt_prob_2 = cvx.Problem(cvx.Minimize(cvx.sum_squares(L_local@temp_y - (q - temp_out_y)) + 0.25*cvx.norm(g_y,1))
                               
                               ,[temp_out_y[index.size-1] == 0]
                               )
        
        opt_prob_2.solve()
        
        #print ('Index size:', index.size,'\n')
        #print ('temp_Out x: ', temp_out_x.value,'\n')
        #print ('temp_Out y: ', temp_out_y.value,'\n')
        
        #print ('temp_x: ', temp_x.value,'\n')
        #print ('temp_y: ', temp_y.value,'\n')
       
            
        x[k] = temp_x.value[index.size-1]
        y[k] = temp_y.value[index.size-1]
        
        out_x[k] = temp_out_x.value[index.size]
        out_y[k] = temp_out_y.value[index.size]
    

    return x, y, out_x, out_y

    
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
            
            #temp_x = inv(L_local.T@L_local)@L_local.T@b
            #temp_y = inv(L_local.T@L_local)@L_local.T@q
            
            temp_x = lsmr(L_local,b)[0]
            temp_y = lsmr(L_local,q)[0]
            
            x_final[k,l] = temp_x[index.size-1]
            y_final[k,l] = temp_y[index.size-1]
            
        
    return x_final[:,num_of_iter_distr-1], y_final[:,num_of_iter_distr-1]

def Optimization_AoA_GD (A, noisy_D, tan_noisy_AoA, noise_X, noise_Y, deg_noisy_AoA):
    
    delta = 0.001
    
    low_a = 70
    high_a = 110
    
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
                            
                            if (deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a):
                                
                                sum_x_a += 2*( (noise_X[j] - noise_X[i])*(tan_noisy_AoA[i,j]**2) - (noise_Y[j] - noise_Y[i])*tan_noisy_AoA[i,j])
                                
                                sum_y_a += 2*( (noise_Y[j] - noise_Y[i]) - (noise_X[j] - noise_X[i])*tan_noisy_AoA[i,j])
                        else:
                            
                            if (deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a):
                                
                                sum_x_a += 2*( (noise_X[j] - noise_X[i])*(tan_noisy_AoA[i,j]**2) - (noise_Y[j] - noise_Y[i])*tan_noisy_AoA[i,j])
                                
                                sum_y_a += 2*( (noise_Y[j] - noise_Y[i]) - (noise_X[j] - noise_X[i])*tan_noisy_AoA[i,j])
                      
                        
                    else:
                        
                        sum_x_d += 2*((noisy_D[i,j] - norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))*( (x_final[i,l-1] - x_final[j,l-1]) / (norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))))
                        
                        sum_y_d += 2*((noisy_D[i,j] - norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))*( (y_final[i,l-1] - y_final[j,l-1]) / (norm(np.array([x_final[i,l-1] - x_final[j,l-1], y_final[i,l-1] - y_final[j,l-1]]),2))))
                        
                        
                        if (deg_noisy_AoA[i,j] > 0):
                            
                            if (deg_noisy_AoA[i,j] <= low_a or deg_noisy_AoA[i,j] >= high_a):
                                
                                sum_x_a += 2*( (x_final[j,l-1] - x_final[i,l-1])*(tan_noisy_AoA[i,j]**2) - (y_final[j,l-1] - y_final[i,l-1])*tan_noisy_AoA[i,j])
                                
                                sum_y_a += 2*( (y_final[j,l-1] - y_final[i,l-1]) - (x_final[j,l-1] - x_final[i,l-1])*tan_noisy_AoA[i,j])
                        
                        else:
                            
                            if (deg_noisy_AoA[i,j] >= -low_a or deg_noisy_AoA[i,j] <= -high_a):
                                
                                sum_x_a += 2*( (x_final[j,l-1] - x_final[i,l-1])*(tan_noisy_AoA[i,j]**2) - (y_final[j,l-1] - y_final[i,l-1])*tan_noisy_AoA[i,j])
                                
                                sum_y_a += 2*( (y_final[j,l-1] - y_final[i,l-1]) - (x_final[j,l-1] - x_final[i,l-1])*tan_noisy_AoA[i,j])
                       
                        
            if (l==1):
                
                x_final[i,l] = noise_X[i] + delta*(sum_x_d + sum_x_a)
                
                y_final[i,l] = noise_Y[i] + delta*(sum_y_d + sum_y_a)
                
                
            else:
                
                x_final[i,l] =  x_final[i,l-1] + delta*(sum_x_d + sum_x_a + 2*(noise_X[i] - x_final[i,l-1])) 
                
                y_final[i,l] =  y_final[i,l-1] + delta*(sum_y_d + sum_y_a + 2*(noise_Y[i] - y_final[i,l-1])) 
                
        
    return x_final[:,num_of_iter_GD-1], y_final[:,num_of_iter_GD-1] 
   
#mu, sigma_x, sigma_y = 0, 0.00001, 0.00001
#mu, sigma_x, sigma_y = 0, 7.49, 2.93 # CEP = 6m, HDOP = 1.2, UERE = 6.7
#mu, sigma_x, sigma_y = 0, 0.16, 4.68 # CEP = 3m, HDOP = 0.7, UERE = 6.7


    
num_of_iter = 100
num_of_vehicles = 20

flag_for_outliers = np.zeros(num_of_iter)
mean_GPS_error = np.zeros(num_of_iter)

mse_GPS_error = np.zeros(num_of_iter)
mse_opt_error = np.zeros(num_of_iter)

mse_GPS_error_outliers = np.zeros(num_of_iter)
mse_opt_error_outliers = np.zeros(num_of_iter)

GPS_error_full = np.zeros((num_of_vehicles, num_of_iter))
GPS_error_outliers_full = np.zeros((num_of_vehicles, num_of_iter))
lapl_error_full = np.zeros((num_of_vehicles, num_of_iter))

max_GPS_error = np.zeros(num_of_iter)
max_opt_error = np.zeros(num_of_iter)

max_GPS_error_outliers = np.zeros(num_of_iter)
max_opt_error_outliers = np.zeros(num_of_iter)

mse_lapl_error = np.zeros(num_of_iter)
max_lapl_error = np.zeros(num_of_iter)

mse_lapl_BS_error = np.zeros(num_of_iter)
max_lapl_BS_error = np.zeros(num_of_iter)

mse_MLE_error = np.zeros(num_of_iter)
max_MLE_error = np.zeros(num_of_iter)

mse_local_error = np.zeros(num_of_iter)
max_local_error = np.zeros(num_of_iter)

zero_flag = np.zeros(num_of_iter)

zero_flag_grand = np.zeros(num_of_iter)

lapl_TP = np.zeros(num_of_iter)
lapl_TN = np.zeros(num_of_iter)
lapl_FP = np.zeros(num_of_iter)
lapl_FN = np.zeros(num_of_iter)

lapl_TPR = np.zeros(num_of_iter)
lapl_FPR = np.zeros(num_of_iter)

BS_TP = np.zeros(num_of_iter)
BS_TN = np.zeros(num_of_iter)
BS_FP = np.zeros(num_of_iter)
BS_FN = np.zeros(num_of_iter)

BS_TPR = np.zeros(num_of_iter)
BS_FPR = np.zeros(num_of_iter)

MLE_TP = np.zeros(num_of_iter)
MLE_TN = np.zeros(num_of_iter)
MLE_FP = np.zeros(num_of_iter)
MLE_FN = np.zeros(num_of_iter)

MLE_TPR = np.zeros(num_of_iter)
MLE_FPR = np.zeros(num_of_iter)

local_TP = np.zeros(num_of_iter)
local_TN = np.zeros(num_of_iter)
local_FP = np.zeros(num_of_iter)
local_FN = np.zeros(num_of_iter)

local_TPR = np.zeros(num_of_iter)
local_FPR = np.zeros(num_of_iter)

Acc = np.zeros(num_of_iter)
BS_Acc = np.zeros(num_of_iter)
MLE_Acc = np.zeros(num_of_iter)
local_Acc = np.zeros(num_of_iter)

thresholds = np.zeros(10)

start = 0
step = 5

for i in range (thresholds.size):
    thresholds[i] = start+step
    step += 5
    
thresh_TP = np.zeros((num_of_iter,thresholds.size))
thresh_TN = np.zeros((num_of_iter,thresholds.size))
thresh_FP = np.zeros((num_of_iter,thresholds.size))
thresh_FN = np.zeros((num_of_iter,thresholds.size))

thresh_TPR = np.zeros((num_of_iter,thresholds.size))
thresh_FPR = np.zeros((num_of_iter,thresholds.size))

thresh_Acc = np.zeros((num_of_iter,thresholds.size))

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
        
low_a = 70
high_a = 110
   
list_of_GPS_outliers = []

L_full = np.zeros((num_of_iter, Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))

array_of_L_rank = np.zeros(num_of_iter)

list_random_outliers_idx = [6, 13]

for v in range(num_of_iter):
    
    print ('Number of vehicles is: ' , Cartesian_Points[:,0].size)
    A = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size), dtype=int)
    Deg = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size), dtype=int)
    
    tmp_D = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    D, a = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size)), np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    AoA = np.zeros((Cartesian_Points[:,0].size,Cartesian_Points[:,0].size))
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            D[i,j] = norm(np.array([Cartesian_Points[i][0]-Cartesian_Points[j][0], Cartesian_Points[i][1]-Cartesian_Points[j][1]]),2)
           
   
    noise_for_distances = np.zeros(Cartesian_Points[:,0].size**2)
    noise_for_distances = np.random.normal(0, sigma_d, Cartesian_Points[:,0].size**2) 
    noise_for_distances = np.reshape(noise_for_distances, (Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))
    
    noisy_D = np.zeros((D[:,0].size,D[0,:].size))
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (D[i,j] + noise_for_distances[i,j] >= 0):
                noisy_D[i,j] = D[i,j] + noise_for_distances[i,j]
            else:
                noisy_D[i,j] = D[i,j] + abs(noise_for_distances[i,j])
            
    
    tmp_D = np.copy(noisy_D)
    
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
                    if (np.sum(A[j,:]) >= 3):
                        A[i,j] = 0
                        A[j,i] = 0
    
         
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (i == j):
                D[i,j] = 0
                noisy_D[i,j] = 0
            if (A[i,j] == 1):
                a[i,j] = CalculateAzimuthAngle(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])
                AoA[i,j] = CalculateAoA(Cartesian_Points[i][0], Cartesian_Points[j][0], Cartesian_Points[i][1], Cartesian_Points[j][1])
                
            else:
                D[i,j] = 0
                noisy_D[i,j] = 0
           
    
    list_for_del = []               
    for i in range (A[:,0].size):
        A[i,i] = 0
        Deg[i,i] = np.sum([A[i,:]])
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
        theta = np.delete(theta,array_for_del,0)
        Deg = np.delete(Deg, array_for_del, 1)
        A = np.delete(A, array_for_del, 1)
        D = np.delete(D, array_for_del, 1)
        noisy_D = np.delete(noisy_D, array_for_del, 1)
        a = np.delete(a, array_for_del, 1)
        AoA = np.delete(AoA, array_for_del, 1)
        L_full = np.delete(L_full, array_for_del, 1)
        L_full = np.delete(L_full, array_for_del, 2)
        print ('Shape of Degree matrix is: ' , np.shape(Deg),'\n')
    
    list_random_outliers_idx=[]
    
    number_of_outliers = int(round(0.2*Cartesian_Points[:,0].size))
    
    for i in range(number_of_outliers):
            
        r=random.randint(0, Cartesian_Points[:,0].size-1)
        while (r in list_random_outliers_idx):
            r=random.randint(0, Cartesian_Points[:,0].size-1)
        list_random_outliers_idx.append(r)
    
    temp_list_random_outliers_idx = list_random_outliers_idx
    list_random_outliers_idx.sort()
    
    
    temp_anchors_idx = []
    for i in range(Cartesian_Points[:,0].size):
        if ( (i in list_random_outliers_idx) == 0):
            temp_anchors_idx.append(i)
            
    L = Deg - A
    
    L_full[v,:,:] = L
    
    array_of_L_rank[v] = matrix_rank(L)
        
    true_outliers_label = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (len(list_random_outliers_idx)):
        
        true_outliers_label[list_random_outliers_idx[i]] = 1
        
    
    true_noisy_Cartesian = np.copy(noisy_Cartesian)
    
    temp_noisy_Cartesian = np.zeros((Cartesian_Points[:,0].size, Cartesian_Points[:,0].size))

    temp_noisy_Cartesian = np.copy(noisy_Cartesian) 

    attack_vector_x = np.zeros(number_of_outliers)
    attack_vector_y = np.zeros(number_of_outliers)
    
    for i in range (len(list_random_outliers_idx)):
        
        attack_vector_x[i] = random.uniform(5,40)
        attack_vector_y[i] = random.uniform(5,40)
        
        temp_noisy_Cartesian[list_random_outliers_idx[i],0] += attack_vector_x[i] 
        temp_noisy_Cartesian[list_random_outliers_idx[i],1] += attack_vector_y[i]
       
        #temp_noisy_Cartesian[list_random_outliers_idx[i],0] *= -1 
        #temp_noisy_Cartesian[list_random_outliers_idx[i],1] *= -1
        
    temp_noisy_Cartesian_2 = np.copy(temp_noisy_Cartesian)
    
    GPS_error = np.zeros((Cartesian_Points[:,0].size))
    GPS_error_outliers = np.zeros((Cartesian_Points[:,0].size))
    
    for i in range(GPS_error.size):
        GPS_error[i] = norm(np.array([Cartesian_Points[i,0] - noisy_Cartesian[i,0] , Cartesian_Points[i,1] - noisy_Cartesian[i,1]]), 2)
        GPS_error_outliers[i] = norm(np.array([Cartesian_Points[i,0] - temp_noisy_Cartesian[i,0] , Cartesian_Points[i,1] - temp_noisy_Cartesian[i,1]]), 2)
        
    GPS_error_full[:,v] = GPS_error
    GPS_error_outliers_full[:,v] = GPS_error_outliers
    
    for i in range (len(list_random_outliers_idx)):
        
        list_of_GPS_outliers.append(GPS_error_outliers[list_random_outliers_idx[i]])
        
    noisy_AoA = np.zeros((D[:,0].size,D[0,:].size))
    noisy_a = np.zeros((a[:,0].size,a[0,:].size))
    
    deg_noisy_AoA = np.zeros((D[:,0].size,D[0,:].size))
    tan_noisy_AoA = np.zeros((D[:,0].size,D[0,:].size))
   
    
    for i in range(D[:,0].size):
        for j in range(D[0,:].size):
            if (A[i,j] == 1):
                
                noisy_a[i,j] = a[i,j] + np.random.normal(0,math.radians(sigma_a),size = 1)
                
                noisy_AoA[i,j] = AoA[i,j] + np.random.normal(0,math.radians(sigma_a),size = 1)
              
                deg_noisy_AoA[i,j] = math.degrees(noisy_AoA[i,j])
                  
                tan_noisy_AoA[i,j] = (math.tan(noisy_AoA[i,j]))
                
                '''
                if ((math.degrees(noisy_AoA[i,j])) <= 90 and math.degrees(noisy_AoA[i,j]) >= low_a):
                    tan_noisy_AoA[i,j] = (math.tan(math.radians(low_a)))
                    
                elif (math.degrees(noisy_AoA[i,j]) <= high_a and math.degrees(noisy_AoA[i,j]) >= 90):
                    tan_noisy_AoA[i,j] = (math.tan(math.radians(high_a)))
                    
                elif ((math.degrees(noisy_AoA[i,j])) >= -90 and math.degrees(noisy_AoA[i,j]) <= -low_a):
                    tan_noisy_AoA[i,j] = (math.tan(math.radians(-low_a)))
                    
                elif (math.degrees(noisy_AoA[i,j]) >= -high_a and math.degrees(noisy_AoA[i,j]) <= -90):
                    tan_noisy_AoA[i,j] = (math.tan(math.radians(-high_a)))
                    
                else:    
                    tan_noisy_AoA[i,j] = (math.tan(noisy_AoA[i,j]))
                '''   
                    
    delta_X = np.zeros((Cartesian_Points[:,0].size))
    delta_Y = np.zeros((Cartesian_Points[:,0].size))
   
    for i in range(noisy_D[:,0].size):
        for j in range(noisy_D[0,:].size):
            if (A[i,j] == 1):
                delta_X[i] += -noisy_D[i,j]*math.sin(noisy_a[i,j])
                delta_Y[i] += -noisy_D[i,j]*math.cos(noisy_a[i,j])
    
    '''         
    #########!!!!!!! OPTIMIZATION WITHOUT OUTLIERS!!!!!!!!!############ 
    
    x_opt = np.zeros((Cartesian_Points[:,0].size))
    y_opt = np.zeros((Cartesian_Points[:,0].size))
   
    #true_noisy_Cartesian[:,0]
    #true_noisy_Cartesian[:,1]
    
    #temp_noisy_Cartesian_2[:,0]
    #temp_noisy_Cartesian_2[:,1]
    
    x_opt, y_opt= Optimization_AoA_GD (A, noisy_D, tan_noisy_AoA, true_noisy_Cartesian[:,0], true_noisy_Cartesian[:,1], deg_noisy_AoA)
    
    error_opt = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_opt.size):
        error_opt[i] = norm(np.array([Cartesian_Points[i,0] - x_opt[i], Cartesian_Points[i,1] - y_opt[i]]), 2)
    
    #########!!!!!!! OPTIMIZATION WITHOUT OUTLIERS!!!!!!!!!############ 
    '''
    
    #########!!!!!!! OPTIMIZATION WITH MLE L1 SPARSITY!!!!!!!!!############
    x_MLE = np.zeros((Cartesian_Points[:,0].size))
    y_MLE = np.zeros((Cartesian_Points[:,0].size))
    
    x_MLE, y_MLE, outliers_MLE_x, outliers_MLE_y = Optimization_Outliers_CF(A,noisy_D,temp_noisy_Cartesian_2[:,0],temp_noisy_Cartesian_2[:,1],tan_noisy_AoA,deg_noisy_AoA)
    
        
    error_MLE = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_MLE.size):
        error_MLE[i] = norm(np.array([Cartesian_Points[i,0] - x_MLE[i], Cartesian_Points[i,1] - y_MLE[i]]), 2)
        
    #########!!!!!!! OPTIMIZATION WITH MLE L1 SPARSITY!!!!!!!!!############
    
    '''
    #########!!!!!!! OPTIMIZATION WITH BLOCK SPARSITY!!!!!!!!!############
    x_BS = np.zeros((Cartesian_Points[:,0].size))
    y_BS = np.zeros((Cartesian_Points[:,0].size))

    
    x_BS, y_BS, outliers_BS_x, outliers_BS_y = Optimization_Outliers_BS(A, noisy_D, temp_noisy_Cartesian_2[:,0], temp_noisy_Cartesian_2[:,1], tan_noisy_AoA, L, np.arange(Cartesian_Points[:,0].size), delta_X, delta_Y)
    
        
    error_BS = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_BS.size):
        error_BS[i] = norm(np.array([Cartesian_Points[i,0] - x_BS[i], Cartesian_Points[i,1] - y_BS[i]]), 2)
        
    #########!!!!!!! OPTIMIZATION WITH BLOCK SPARSITY!!!!!!!!!############  
    '''
    '''   
    #########!!!!!!! OPTIMIZATION WITH LOCAL LAPL L1 SPARSITY !!!!!!!!!############
    x_local = np.zeros((Cartesian_Points[:,0].size))
    y_local = np.zeros((Cartesian_Points[:,0].size))
   
    
    x_local, y_local, outliers_local_x, outliers_local_y = Optimization_Distr_Outliers(A,Deg, temp_noisy_Cartesian_2[:,0],temp_noisy_Cartesian_2[:,1],delta_X, delta_Y)
    
        
    error_local = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_local.size):
        error_local[i] = norm(np.array([Cartesian_Points[i,0] - x_local[i], Cartesian_Points[i,1] - y_local[i]]), 2)
        
    #########!!!!!!! OPTIMIZATION WITH LOCAL LAPL L1 SPARSITY !!!!!!!!!############
    '''
    
    #########!!!!!!! OPTIMIZATION WITH LAPL L1 SPARSITY!!!!!!!!!############
    x_opt_outliers = np.zeros((Cartesian_Points[:,0].size))
    y_opt_outliers = np.zeros((Cartesian_Points[:,0].size))
   
    outliers_x = np.zeros((Cartesian_Points[:,0].size))
    outliers_y = np.zeros((Cartesian_Points[:,0].size))
    
    
    x_opt_outliers, y_opt_outliers, outliers_x, outliers_y = Optimization_Outliers(A, noisy_D, temp_noisy_Cartesian[:,0], temp_noisy_Cartesian[:,1], tan_noisy_AoA, L, np.arange(Cartesian_Points[:,0].size), delta_X, delta_Y)
   
    
    error_opt_outliers = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (error_opt_outliers.size):
        error_opt_outliers[i] = norm(np.array([Cartesian_Points[i,0] - x_opt_outliers[i], Cartesian_Points[i,1] - y_opt_outliers[i]]), 2)
    
    lapl_error_full[:,v] =  error_opt_outliers   
    #########!!!!!!! OPTIMIZATION WITH LAPL L1 SPARSITY!!!!!!!!!############

    #########!!!!!!! FIND THE OUTLIERS, REMOVE THEM AND PERFORM LAPLACIAN !!!!!!!!!############
    difference = np.zeros(Cartesian_Points[:,0].size)
    
    tmp_difference = np.zeros(Cartesian_Points[:,0].size)
    
    array_of_outliers = []
    
    for i in range(difference.size):
        
        difference[i] = (norm(np.array([(temp_noisy_Cartesian[i,0] ) - x_opt_outliers[i], (temp_noisy_Cartesian[i,1] ) - y_opt_outliers[i]]), 2))
        
        tmp_difference[i] = difference[i]
        
        if (difference[i] <= 10):
            difference[i] = 0
            
    for i in range (thresholds.size):

        thresh_array_of_outliers = []

        for j in range (tmp_difference.size):
            
            if (tmp_difference[j] > thresholds[i]):
                thresh_array_of_outliers.append(j)
          
        thresh_est_outliers_label = np.zeros(Cartesian_Points[:,0].size)
        
        for j in range (len(thresh_array_of_outliers)):
        
            thresh_est_outliers_label[thresh_array_of_outliers[j]] = 1
            
        for j in range (Cartesian_Points[:,0].size):
            
            if (true_outliers_label[j] == 1 and thresh_est_outliers_label[j] == 1):
                thresh_TP[v][i] += 1
                
            elif (true_outliers_label[j] == 0 and thresh_est_outliers_label[j] == 0):
                thresh_TN[v][i] += 1
                
            elif (true_outliers_label[j] == 1 and thresh_est_outliers_label[j] == 0):
                thresh_FN[v][i] += 1
                
            elif (true_outliers_label[j] == 0 and thresh_est_outliers_label[j] == 1):
                thresh_FP[v][i] += 1
                
        thresh_TP[v][i] /= Cartesian_Points[:,0].size
        thresh_TN[v][i] /= Cartesian_Points[:,0].size
        thresh_FN[v][i] /= Cartesian_Points[:,0].size
        thresh_FP[v][i] /= Cartesian_Points[:,0].size
        
        if (thresh_TP[v][i] > 0 or thresh_FN[v][i] > 0):
            thresh_TPR[v][i] = thresh_TP[v][i]/(thresh_TP[v][i] + thresh_FN[v][i])
        else:
            thresh_TPR[v][i] = 1
        if (thresh_FP[v][i] > 0 or thresh_TN[v][i] > 0):
            thresh_FPR[v][i] = thresh_FP[v][i]/(thresh_FP[v][i] + thresh_TN[v][i])
        else:
            thresh_FPR[v][i] = 1
     
        thresh_Acc[v][i] = (thresh_TP[v][i] + thresh_TN[v][i])/(thresh_TP[v][i] + thresh_TN[v][i] + thresh_FP[v][i] + thresh_FN[v][i])
   
    if (np.sum(difference) != 0):
        
        kmeans = KMeans(n_clusters=2).fit(difference.reshape(-1,1))
        
        centers = np.zeros(1)
        
        centers[0] = np.argwhere(kmeans.cluster_centers_ == np.max(kmeans.cluster_centers_))[0][0]
       
        for i in range(difference.size):
            
            if (kmeans.labels_[i] == centers[0]): 
            #if (difference[i] > 10):
                array_of_outliers.append(i)
            
    for i in range (len(array_of_outliers)):
         
        if (outliers_x.size == 2*Cartesian_Points[:,0].size):
            temp_noisy_Cartesian[array_of_outliers[i],0] -= outliers_x[A[:,0].size + array_of_outliers[i]]
            temp_noisy_Cartesian[array_of_outliers[i],1] -= outliers_y[A[:,0].size + array_of_outliers[i]]
        else:
            temp_noisy_Cartesian[array_of_outliers[i],0] -= outliers_x[array_of_outliers[i]]
            temp_noisy_Cartesian[array_of_outliers[i],1] -= outliers_y[array_of_outliers[i]]
            
    '''        
    list_random_anchors_idx=[]
    list_random_anchors=[]
       
    for i in range(int(round(1.0*Cartesian_Points[:,0].size))):
            
        r=random.randint(0,Cartesian_Points[:,0].size-1)
        while (r in list_random_anchors_idx):
            r=random.randint(0,Cartesian_Points[:,0].size-1)
        list_random_anchors_idx.append(r)
        list_random_anchors.append(temp_noisy_Cartesian[r,:])
             
        
    anchors_index = np.zeros((len(list_random_anchors_idx)))
    anchors = np.zeros((len(list_random_anchors)))
        
    anchors_index = np.asarray(list_random_anchors_idx)
    anchors = np.asarray(list_random_anchors)    
    
    if (len(array_of_outliers)>0):
       
        list_of_anchors_for_del = []
        for i in range (len(array_of_outliers)):
            list_of_anchors_for_del.append(np.argwhere(array_of_outliers[i] == anchors_index))
        
        anchors_index = np.delete(anchors_index, list_of_anchors_for_del)
        anchors = np.delete(anchors, list_of_anchors_for_del, 0)
    
    lapl_error = np.zeros((Cartesian_Points[:,0].size))
        
    Points_lapl =  np.zeros((Cartesian_Points[:,0].size, 3))
        
    L_bar = np.zeros((Cartesian_Points[:,0].size + anchors_index.size, Cartesian_Points[0,:].size))
    
    Points_lapl, lapl_error, L_bar = Solve_The_System(Cartesian_Points, L, anchors_index, anchors, delta_X, delta_Y)
    '''
    #########!!!!!!! FIND THE OUTLIERS, REMOVE THEM AND PERFORM LAPLACIAN !!!!!!!!!############
    
    '''
    #########!!!!!!! FIND THE OUTLIERS FROM BLOCK SPARSITY !!!!!!!!!############
    difference_BS = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range(difference_BS.size):
        
        difference_BS[i] = (norm(np.array([(temp_noisy_Cartesian_2[i,0] ) - x_BS[i], (temp_noisy_Cartesian_2[i,1] ) - y_BS[i]]), 2))
        
        if (difference_BS[i] <= 10):
            difference_BS[i] = 0
            
    array_of_outliers_BS = []
    
    if (np.sum(difference_BS) != 0):
        
        kmeans_BS = KMeans(n_clusters=2).fit(difference_BS.reshape(-1,1))
        
        centers_BS = np.zeros(1)
        
        centers_BS[0] = np.argwhere(kmeans_BS.cluster_centers_ == np.max(kmeans_BS.cluster_centers_))[0][0]
        
        for i in range(difference.size):
            
            if (kmeans_BS.labels_[i] == centers_BS[0]):
           
                array_of_outliers_BS.append(i) 
                
    #########!!!!!!! FIND THE OUTLIERS FROM BLOCK SPARSITY !!!!!!!!!############ 
    '''
    '''
    #########!!!!!!! FIND THE OUTLIERS FROM MLE L1 SPARSITY !!!!!!!!!############
    difference_MLE = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range(difference_MLE.size):
        
        difference_MLE[i] = (norm(np.array([(temp_noisy_Cartesian_2[i,0] ) - x_MLE[i], (temp_noisy_Cartesian_2[i,1] ) - y_MLE[i]]), 2))
        
        if (difference_MLE[i] <= 10):
            difference_MLE[i] = 0
            
    array_of_outliers_MLE = []
    
    if (np.sum(difference_MLE) != 0):
        
        kmeans_MLE = KMeans(n_clusters=2).fit(difference_MLE.reshape(-1,1))
        
        centers_MLE = np.zeros(1)
        
        centers_MLE[0] = np.argwhere(kmeans_MLE.cluster_centers_ == np.max(kmeans_MLE.cluster_centers_))[0][0]
        
        for i in range(difference_MLE.size):
            
            if (kmeans_MLE.labels_[i] == centers_MLE[0]):
           
                array_of_outliers_MLE.append(i) 
                
    #########!!!!!!! FIND THE OUTLIERS FROM MLE L1 SPARSITY !!!!!!!!!############
    '''
    '''
    #########!!!!!!! FIND THE OUTLIERS FROM LAPL LOCAL L1 SPARSITY !!!!!!!!!############
    difference_local = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range(difference_local.size):
        
        difference_local[i] = (norm(np.array([(temp_noisy_Cartesian_2[i,0] ) - x_local[i], (temp_noisy_Cartesian_2[i,1] ) - y_local[i]]), 2))
        
        if (difference_local[i] <= 10):
            difference_local[i] = 0
            
    array_of_outliers_local = []
    
    if (np.sum(difference_local) != 0):
        
        kmeans_local = KMeans(n_clusters=2).fit(difference_local.reshape(-1,1))
        
        centers_local = np.zeros(1)
        
        centers_local[0] = np.argwhere(kmeans_local.cluster_centers_ == np.max(kmeans_local.cluster_centers_))[0][0]
        
        for i in range(difference_local.size):
            
            if (kmeans_local.labels_[i] == centers_local[0]):
           
                array_of_outliers_local.append(i) 
                
    #########!!!!!!! FIND THE OUTLIERS FROM LAPL LOCAL L1 SPARSITY !!!!!!!!!############
    '''
    
    #########!!!!!!! TP, FP, ... USING LAPLACIAN !!!!!!!!!############
    est_outliers_label = np.zeros(Cartesian_Points[:,0].size)
   
    for i in range (len(array_of_outliers)):
        
        est_outliers_label[array_of_outliers[i]] = 1
        
    for i in range (Cartesian_Points[:,0].size):
        
        if (true_outliers_label[i] == 1 and est_outliers_label[i] == 1):
            lapl_TP[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_label[i] == 0):
            lapl_TN[v] += 1
            
        elif (true_outliers_label[i] == 1 and est_outliers_label[i] == 0):
            lapl_FN[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_label[i] == 1):
            lapl_FP[v] += 1
            
    lapl_TP[v] /= Cartesian_Points[:,0].size
    lapl_TN[v] /= Cartesian_Points[:,0].size
    lapl_FN[v] /= Cartesian_Points[:,0].size
    lapl_FP[v] /= Cartesian_Points[:,0].size
    
    if (v == 0):
        full_true_outliers_label = np.copy(true_outliers_label) 
        full_est_outliers_label = np.copy(est_outliers_label) 
    else:
        full_true_outliers_label = np.concatenate([full_true_outliers_label, true_outliers_label]) 
        full_est_outliers_label = np.concatenate([full_est_outliers_label, est_outliers_label]) 
        
    if (lapl_TP[v] > 0 or lapl_FN[v] > 0):
        lapl_TPR[v] = lapl_TP[v]/(lapl_TP[v] + lapl_FN[v])
    else:
        lapl_TPR[v] = 1
    if (lapl_FP[v] > 0 or lapl_TN[v] > 0):
        lapl_FPR[v] = lapl_FP[v]/(lapl_FP[v] + lapl_TN[v])
    else:
        lapl_FPR[v] = 1
     
    Acc[v] = (lapl_TP[v] + lapl_TN[v])/(lapl_TP[v] + lapl_TN[v] + lapl_FP[v] + lapl_FN[v])
   
    #########!!!!!!! TP, FP, ... USING LAPLACIAN !!!!!!!!!############
    
    '''
    #########!!!!!!! TP, FP, ... USING BLOCK SPARSITY !!!!!!!!!############
    
    est_outliers_BS_label = np.zeros(Cartesian_Points[:,0].size)
    
    for i in range (len(array_of_outliers_BS)):
        
        est_outliers_BS_label[array_of_outliers_BS[i]] = 1
        
    for i in range (Cartesian_Points[:,0].size):
        
        if (true_outliers_label[i] == 1 and est_outliers_BS_label[i] == 1):
            BS_TP[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_BS_label[i] == 0):
            BS_TN[v] += 1
            
        elif (true_outliers_label[i] == 1 and est_outliers_BS_label[i] == 0):
            BS_FN[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_BS_label[i] == 1):
            BS_FP[v] += 1
            
    BS_TP[v] /= Cartesian_Points[:,0].size
    BS_TN[v] /= Cartesian_Points[:,0].size
    BS_FN[v] /= Cartesian_Points[:,0].size
    BS_FP[v] /= Cartesian_Points[:,0].size
    
    if (v == 0):
         
        full_est_outliers_BS_label = np.copy(est_outliers_BS_label) 
    else:
        
        full_est_outliers_BS_label = np.concatenate([full_est_outliers_BS_label, est_outliers_BS_label]) 
        
    if (BS_TP[v] > 0 or BS_FN[v] > 0):
        BS_TPR[v] = BS_TP[v]/(BS_TP[v] + BS_FN[v])
    else:
        BS_TPR[v] = 1
    if (BS_FP[v] > 0 or BS_TN[v] > 0):    
        BS_FPR[v] = BS_FP[v]/(BS_FP[v] + BS_TN[v])
    else:
        BS_FPR[v] = 1
     
    BS_Acc[v] = (BS_TP[v] + BS_TN[v])/(BS_TP[v] + BS_TN[v] + BS_FP[v] + BS_FN[v])
    
    if (len(list_random_outliers_idx) == len(array_of_outliers)):
        
        if (list_random_outliers_idx == array_of_outliers):
            flag_for_outliers[v] = 1
            
    #########!!!!!!! TP, FP, ... USING BLOCK SPARSITY !!!!!!!!!############
    '''
    '''
    #########!!!!!!! TP, FP, ... USING MLE L1 SPARSITY !!!!!!!!!############
    est_outliers_MLE_label = np.zeros(Cartesian_Points[:,0].size)
   
    for i in range (len(array_of_outliers_MLE)):
        
        est_outliers_MLE_label[array_of_outliers_MLE[i]] = 1
        
    for i in range (Cartesian_Points[:,0].size):
        
        if (true_outliers_label[i] == 1 and est_outliers_MLE_label[i] == 1):
            MLE_TP[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_MLE_label[i] == 0):
            MLE_TN[v] += 1
            
        elif (true_outliers_label[i] == 1 and est_outliers_MLE_label[i] == 0):
            MLE_FN[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_MLE_label[i] == 1):
            MLE_FP[v] += 1
            
    MLE_TP[v] /= Cartesian_Points[:,0].size
    MLE_TN[v] /= Cartesian_Points[:,0].size
    MLE_FN[v] /= Cartesian_Points[:,0].size
    MLE_FP[v] /= Cartesian_Points[:,0].size
    
    if (v == 0): 
        full_est_outliers_MLE_label = np.copy(est_outliers_MLE_label) 
    else:
        full_est_outliers_MLE_label = np.concatenate([full_est_outliers_MLE_label, est_outliers_MLE_label]) 
        
    if (MLE_TP[v] > 0 or MLE_FN[v] > 0):
        MLE_TPR[v] = MLE_TP[v]/(MLE_TP[v] + MLE_FN[v])
    else:
        MLE_TPR[v] = 1
    if (MLE_FP[v] > 0 or MLE_TN[v] > 0):
        MLE_FPR[v] = MLE_FP[v]/(MLE_FP[v] + MLE_TN[v])
    else:
        MLE_FPR[v] = 1
     
    MLE_Acc[v] = (MLE_TP[v] + MLE_TN[v])/(MLE_TP[v] + MLE_TN[v] + MLE_FP[v] + MLE_FN[v])
   
    #########!!!!!!! TP, FP, ... USING MLE L1 SPARSITY !!!!!!!!!############
    '''
    '''
    #########!!!!!!! TP, FP, ... USING LAPL LOCAL L1 SPARSITY !!!!!!!!!############
    est_outliers_local_label = np.zeros(Cartesian_Points[:,0].size)
   
    for i in range (len(array_of_outliers_local)):
        
        est_outliers_local_label[array_of_outliers_local[i]] = 1
        
    for i in range (Cartesian_Points[:,0].size):
        
        if (true_outliers_label[i] == 1 and est_outliers_local_label[i] == 1):
            local_TP[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_local_label[i] == 0):
            local_TN[v] += 1
            
        elif (true_outliers_label[i] == 1 and est_outliers_local_label[i] == 0):
            local_FN[v] += 1
            
        elif (true_outliers_label[i] == 0 and est_outliers_local_label[i] == 1):
            local_FP[v] += 1
            
    local_TP[v] /= Cartesian_Points[:,0].size
    local_TN[v] /= Cartesian_Points[:,0].size
    local_FN[v] /= Cartesian_Points[:,0].size
    local_FP[v] /= Cartesian_Points[:,0].size
    
    if (v == 0): 
        full_est_outliers_local_label = np.copy(est_outliers_local_label) 
    else:
        full_est_outliers_local_label = np.concatenate([full_est_outliers_local_label, est_outliers_local_label]) 
        
    if (local_TP[v] > 0 or local_FN[v] > 0):
        local_TPR[v] = local_TP[v]/(local_TP[v] + local_FN[v])
    else:
        local_TPR[v] = 1
    if (local_FP[v] > 0 or local_TN[v] > 0):
        local_FPR[v] = local_FP[v]/(local_FP[v] + local_TN[v])
    else:
        local_FPR[v] = 1
     
    local_Acc[v] = (local_TP[v] + local_TN[v])/(local_TP[v] + local_TN[v] + local_FP[v] + local_FN[v])
   
    #########!!!!!!! TP, FP, ... USING LAPL LOCAL L1 SPARSITY !!!!!!!!!############
    '''
    end_opt = timer()
    
    
    
    print ('Iteration: ', v, '\n')
    
    mean_GPS_error[v] = np.mean(GPS_error)
    
    mse_GPS_error[v] = (np.sum(np.power(GPS_error,2))/Cartesian_Points[:,0].size)
    mse_GPS_error_outliers[v] = (np.sum(np.power(GPS_error_outliers,2))/Cartesian_Points[:,0].size)
    
    #mse_opt_error[v] = (np.sum(np.power(error_opt,2))/Cartesian_Points[:,0].size)
    
    mse_opt_error_outliers[v] = (np.sum(np.power(error_opt_outliers,2))/Cartesian_Points[:,0].size)
    #mse_lapl_error[v] = (np.sum(np.power(lapl_error,2))/Cartesian_Points[:,0].size)
    #mse_lapl_BS_error[v] = (np.sum(np.power(error_BS,2))/Cartesian_Points[:,0].size)
    #mse_MLE_error[v] = (np.sum(np.power(error_MLE,2))/Cartesian_Points[:,0].size)
    #mse_local_error[v] = (np.sum(np.power(error_local,2))/Cartesian_Points[:,0].size)
    
    max_GPS_error[v] = np.max(GPS_error)
    max_GPS_error_outliers[v] = np.max(GPS_error_outliers)
    
    #max_opt_error[v] = np.max(error_opt)
    
    max_opt_error_outliers[v] = np.max(error_opt_outliers)
    #max_lapl_error[v] = np.max(lapl_error)
    #max_lapl_BS_error[v] = np.max(error_BS)
    max_MLE_error[v] = np.max(error_MLE)
    #max_local_error[v] = np.max(error_local)
    
    '''
    if (flag_for_outliers[v] == 0):
        print ('Zero flag','\n')
        zero_flag[v] = 1
        #break;
        if ( mse_lapl_error[v] > mse_GPS_error_outliers[v]):
            print ('Issue')
            zero_flag_grand[v] = 1
            #break;
    
    #if (lapl_TP[v] == 0):
        #break;
        
    if (Acc[v] <= 0.8 or lapl_TP[v] == 0):
        print ('Bad accuracy','\n')
        #break;
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
        
    if (array_of_L_rank[v] < Cartesian_Points[:,0].size-1):
        print ('Rank issue')
   

thresh_mean_FPR = np.zeros(thresholds.size)
thresh_mean_TPR = np.zeros(thresholds.size)
thresh_mean_Acc = np.zeros(thresholds.size)

for i in range (thresholds.size):
    thresh_mean_FPR[i] = np.mean(thresh_FPR[:,i])
    thresh_mean_TPR[i] = np.mean(thresh_TPR[:,i])
    thresh_mean_Acc[i] = np.mean(thresh_Acc[:,i])

Points_for_ROC = np.zeros((thresholds.size+2,2))


Points_for_ROC[0,0] = 1
Points_for_ROC[0,1] = 1
Points_for_ROC[1:thresholds.size+1,0] = np.copy(thresh_mean_FPR)
Points_for_ROC[1:thresholds.size+1,1] = np.copy(thresh_mean_TPR) 
Points_for_ROC[thresholds.size+1,0] = 0
Points_for_ROC[thresholds.size+1,1] = 0


print ('Mean of GPS noise is: ', np.mean(mean_GPS_error), '\n')

print ('\n')

'''
print ('Mean MSE of GPS is: ', np.mean(mse_GPS_error), '\n')
print ('Mean MSE of Optimization is: ', np.mean(mse_opt_aoa_error), '\n')

print ('Mean MSE of GPS_outlier is: ', np.mean(mse_GPS_error_outliers), '\n')
print ('Mean MSE of Optimization_outlier is: ', np.mean(mse_opt_aoa_error_outliers), '\n')
'''

'''
print ('Avg True Positive Rate', np.mean(lapl_TPR), '\n')
print ('Avg False Positive Rate', np.mean(lapl_FPR), '\n')
print ('Avg Lapl Accuracy', np.mean(Acc), '\n')
print ('Avg Lapl_BS Accuracy', np.mean(BS_Acc), '\n')
'''

if (len(list_of_GPS_outliers) > 0):
    print ('Min outlier distance: ', np.min(np.asarray(list_of_GPS_outliers)), '\n')
    print ('Max outlier distance: ', np.max(np.asarray(list_of_GPS_outliers)), '\n')
    print ('Avg outlier distance: ', np.mean(np.asarray(list_of_GPS_outliers)), '\n')

'''
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
'''

plt_12.figure(12)
plt_12.plot(np.sort(mse_GPS_error), np.arange(len(np.sort(mse_GPS_error)))/float(len(mse_GPS_error)), 'r*--', label = 'GPS')
#plt_12.plot(np.sort(mse_opt_error), np.arange(len(np.sort(mse_opt_error)))/float(len(mse_opt_error)), 'k*--', label = 'TCL-MLE without outliers')
#plt_12.plot(np.sort(mse_lapl_BS_error), np.arange(len(np.sort(mse_lapl_BS_error)))/float(len(mse_lapl_BS_error)), 'c*--', label = 'GCL-BS with outliers')
plt_12.plot(np.sort(mse_GPS_error_outliers), np.arange(len(np.sort(mse_GPS_error_outliers)))/float(len(mse_GPS_error_outliers)), 'y*--', label = 'Spoofed GPS')
plt_12.plot(np.sort(mse_opt_error_outliers), np.arange(len(np.sort(mse_opt_error_outliers)))/float(len(mse_opt_error_outliers)), 'm*--', label = 'GCL')
#plt_12.plot(np.sort(mse_MLE_error), np.arange(len(np.sort(mse_MLE_error)))/float(len(mse_MLE_error)), 'g*--', label = 'TCL-MLE')
plt_12.xticks(fontsize=28)
plt_12.yticks(fontsize=28)
plt_12.tick_params(direction='out', length=8)
plt_12.legend(facecolor='white', fontsize = 25 )
plt_12.grid(b=True)
plt_12.xlabel('Mean Square Localization Error [m$^2$]', fontsize = 35)
plt_12.ylabel('CDF', fontsize = 35)
plt_12.show

'''
plt_12.figure(12)
plt_12.plot(np.sort(mse_GPS_error), np.arange(len(np.sort(mse_GPS_error)))/float(len(mse_GPS_error)), 'r*--', label = 'GPS')
#plt_12.plot(np.sort(mse_opt_error), np.arange(len(np.sort(mse_opt_error)))/float(len(mse_opt_error)), 'k*--', label = 'TCL-MLE without outliers')
#plt_12.plot(np.sort(mse_lapl_BS_error), np.arange(len(np.sort(mse_lapl_BS_error)))/float(len(mse_lapl_BS_error)), 'c*--', label = 'GCL-BS with outliers')
plt_12.plot(np.sort(mse_GPS_error_outliers), np.arange(len(np.sort(mse_GPS_error_outliers)))/float(len(mse_GPS_error_outliers)), 'y*--', label = 'Spoofed GPS')
plt_12.plot(np.sort(mse_opt_error_outliers), np.arange(len(np.sort(mse_opt_error_outliers)))/float(len(mse_opt_error_outliers)), 'm*--', label = 'GCL-L1 with outliers')
plt_12.plot(np.sort(mse_lapl_error), np.arange(len(np.sort(mse_lapl_error)))/float(len(mse_lapl_error)), 'b*--', label = 'GCL after the removal of outliers')
#plt_12.plot(np.sort(mse_MLE_error), np.arange(len(np.sort(mse_MLE_error)))/float(len(mse_MLE_error)), 'g*--', label = 'TCL-MLE with outliers')
#plt_12.plot(np.sort(mse_local_error), np.arange(len(np.sort(mse_local_error)))/float(len(mse_local_error)), '-', c = 'lightpink' , marker = '*', label = 'GCL-LOCAL with outliers')
plt_12.xticks(fontsize=28)
plt_12.yticks(fontsize=28)
plt_12.tick_params(direction='out', length=8)
plt_12.legend(facecolor='white', fontsize = 25 )
plt_12.grid(b=True)
plt_12.xlabel('Localization Mean Square Error [m$^2$]', fontsize = 35)
plt_12.ylabel('CDF', fontsize = 35)
plt_12.show
'''

plt_13.figure(13)
plt_13.plot(np.sort(max_GPS_error), np.arange(len(np.sort(max_GPS_error)))/float(len(max_GPS_error)), 'r*--', linewidth = 7.0, label = 'GPS')
#plt_13.plot(np.sort(max_opt_error), np.arange(len(np.sort(max_opt_error)))/float(len(max_opt_error)), 'k*--', label = 'TCL-MLE without outliers')
#plt_13.plot(np.sort(max_lapl_BS_error), np.arange(len(np.sort(max_lapl_BS_error)))/float(len(max_lapl_BS_error)), 'c*--', label = 'GCL-BS with outliers')
#plt_13.plot(np.sort(max_GPS_error_outliers), np.arange(len(np.sort(max_GPS_error_outliers)))/float(len(max_GPS_error_outliers)), 'y*--', label = 'Spoofed GPS')
plt_13.plot(np.sort(max_GPS_error_outliers), np.arange(len(np.sort(max_GPS_error_outliers)))/float(len(max_GPS_error_outliers)), 'y*--', linewidth = 7.0, label = 'GPS-4 attacked nodes')
#plt_13.plot(np.sort(max_opt_error_outliers), np.arange(len(np.sort(max_opt_error_outliers)))/float(len(max_opt_error_outliers)), 'm*--', label = 'GCL-L1 with outliers')
plt_13.plot(np.sort(max_opt_error_outliers), np.arange(len(np.sort(max_opt_error_outliers)))/float(len(max_opt_error_outliers)), 'm*--', linewidth = 7.0, label = 'RGCL-4 attacked nodes')
#plt_13.plot(np.sort(max_lapl_error), np.arange(len(np.sort(max_lapl_error)))/float(len(max_lapl_error)), 'b*--', label = 'GCL after the removal of outliers')
#plt_12.plot(np.sort(max_MLE_error), np.arange(len(np.sort(max_MLE_error)))/float(len(max_MLE_error)), 'g*--', label = 'TCL-MLE with outliers')
plt_12.plot(np.sort(max_MLE_error), np.arange(len(np.sort(max_MLE_error)))/float(len(max_MLE_error)), 'g*--', linewidth = 7.0, label = 'RTCL-MLE-4 attacked nodes')
#plt_12.plot(np.sort(max_local_error), np.arange(len(np.sort(max_local_error)))/float(len(max_local_error)), '-', c = 'lightpink' , marker = '*', label = 'GCL-LOCAL with outliers')
plt_13.xticks(fontsize=28)
plt_13.yticks(fontsize=28)
plt_13.tick_params(direction='out', length=8)
plt_13.legend(facecolor='white', fontsize = 25 )
plt_13.grid(b=True)
plt_13.xlabel('Maximum localization error [m]', fontsize = 35)
plt_13.ylabel('CDF', fontsize = 35)
plt_13.show()


if (number_of_outliers != 0):
    fpr, tpr, _ = metrics.roc_curve(full_true_outliers_label, full_est_outliers_label)
else:
    fpr = np.zeros(3)
    fpr[0] = 0
    fpr[2] = 1
    fpr[1] = np.mean(lapl_FPR)
    
    tpr = np.zeros(3)
    tpr[0] = 0
    tpr[2] = 1
    tpr[1] = np.mean(lapl_TPR)

'''
if (number_of_outliers != 0):
    fpr_BS, tpr_BS, _ = metrics.roc_curve(full_true_outliers_label, full_est_outliers_BS_label)
else:
    fpr_BS = np.zeros(3)
    tpr_BS = np.zeros(3)
    
    fpr_BS[2] = 1
    fpr_BS[1] = np.mean(BS_FPR)
    
    tpr_BS[2] = 1
    tpr_BS[1] = np.mean(BS_TPR)
'''
'''
if (number_of_outliers != 0):
    fpr_MLE, tpr_MLE, _ = metrics.roc_curve(full_true_outliers_label, full_est_outliers_MLE_label)
else:
    fpr_MLE = np.zeros(3)
    tpr_MLE = np.zeros(3)
    
    fpr_MLE[2] = 1
    fpr_MLE[1] = np.mean(MLE_FPR)
    
    tpr_MLE[2] = 1
    tpr_MLE[1] = np.mean(MLE_TPR)
'''    
'''
if (number_of_outliers != 0):
    fpr_local, tpr_local, _ = metrics.roc_curve(full_true_outliers_label, full_est_outliers_local_label)
else:
    fpr_local = np.zeros(3)
    tpr_local = np.zeros(3)
    
    fpr_local[2] = 1
    fpr_local[1] = np.mean(local_FPR)
    
    tpr_local[2] = 1
    tpr_local[1] = np.mean(local_TPR)
'''  
#plt_9.figure(9)
#plt_9.hist(thresh_mean_FPR, label='FPR from different thresholds')
#plt_9.hist(thresh_mean_TPR, label='TPR from different thresholds')
#plt_9.legend(facecolor='white', fontsize = 20)

'''
plt_16.figure(16)
plt_16.plot(Points_for_ROC[:,0], Points_for_ROC[:,1], 'bo-', label = 'different thresholds, GCL-L1 ')
plt_16.plot(fpr, tpr, 'ms-', label = 'k-means, GCL-L1')
plt_16.plot(fpr_BS, tpr_BS, 'g<-', label = 'k-means, GCL-BS')
#plt_16.plot(fpr_MLE, tpr_MLE, 'ks-', label = 'k-means, TCL-MLE')
#plt_16.plot(fpr_local, tpr_local, 'yd-', label = 'k-means, GCL-LOCAL-L1')
plt_16.plot([0,1], [0,1], 'r--', linewidth=3.0)
plt_16.xticks(fontsize=28)
plt_16.yticks(fontsize=28)
plt_16.xlabel('False Positive Rate', fontsize = 35)
plt_16.ylabel('True Positive Rate', fontsize = 35)
plt_16.tick_params(direction='out', length=8)
plt_16.legend(facecolor='white', fontsize = 30)
plt_16.title('ROC Curve',fontsize = 35)
plt_16.show()


print ('AUC for GCL-L1 is: ', np.trapz(tpr, fpr), '\n')
print ('AUC for thresholds in GCL-L1 is: ', -np.trapz(Points_for_ROC[:,1], Points_for_ROC[:,0]), '\n')
print ('AUC for GCL-BS is: ', np.trapz(tpr_BS, fpr_BS), '\n')
#print ('AUC for TCL-MLE is: ', np.trapz(tpr_MLE, fpr_MLE), '\n')
#print ('AUC for GCL-LOCAL-L1 is: ', np.trapz(tpr_local, fpr_local), '\n')
'''

plt_16.figure(16)
plt_16.plot(fpr, tpr, 'ms-', label = 'GCL')
#plt_16.plot(fpr_MLE, tpr_MLE, 'ks-', label = 'TCL-MLE')
plt_16.plot([0,1], [0,1], 'r--', linewidth=3.0)
plt_16.xticks(fontsize=28)
plt_16.yticks(fontsize=28)
plt_16.xlabel('False Positive Rate', fontsize = 35)
plt_16.ylabel('True Positive Rate', fontsize = 35)
plt_16.tick_params(direction='out', length=8)
plt_16.legend(facecolor='white', fontsize = 30)
plt_16.title('ROC Curve',fontsize = 35)
plt_16.show()

vehicle_idx = list_random_outliers_idx[0]
plt_6.figure(6)
plt_6.plot(np.sort(GPS_error_full[vehicle_idx, :]), np.arange(len(np.sort(GPS_error_full[vehicle_idx, :])))/float(len(GPS_error_full[vehicle_idx, :])), 'r*-', label="GPS", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(lapl_error_full[vehicle_idx, :]), np.arange(len(np.sort(lapl_error_full[vehicle_idx, :])))/float(len(lapl_error_full[vehicle_idx, :])), 'b*-', label="CGCL", linewidth = 4, markersize = 6)
plt_6.plot(np.sort(GPS_error_outliers_full[vehicle_idx, :]), np.arange(len(np.sort(GPS_error_outliers_full[vehicle_idx, :])))/float(len(GPS_error_outliers_full[vehicle_idx, :])), 'c*-', label="Spoofed GPS", linewidth = 4, markersize = 6)
plt_6.xticks(fontsize=28)
plt_6.yticks(fontsize=28)
plt_6.tick_params(direction='out', length=8)
plt_6.grid(b=True)
plt_6.legend(facecolor='white', fontsize = 37 )
plt_6.xlabel('Localization Error [m]', fontsize = 35)
plt_6.ylabel('CDF', fontsize = 35)
plt_6.title('Vehicle ' + str(vehicle_idx), fontsize = 35)
plt_6.show()


print ('AUC for GCL is: ', np.trapz(tpr, fpr), '\n')
#print ('AUC for TCL-MLE is: ', np.trapz(tpr_MLE, fpr_MLE), '\n')

if (norm(mse_opt_error_outliers) < norm(mse_GPS_error)):
    print ('MSE Laplacian reduction: ', norm(mse_opt_error_outliers- mse_GPS_error)/norm(mse_GPS_error))
    
else:
    print ('MSE Laplacian increment: ', norm(mse_opt_error_outliers- mse_GPS_error)/norm(mse_GPS_error))


'''   
if (norm(mse_lapl_BS_error) < norm(mse_GPS_error)):    
    print ('MSE BS reduction: ', norm(mse_lapl_BS_error- mse_GPS_error)/norm(mse_GPS_error))

else:
    print ('MSE BS increment: ', norm(mse_lapl_BS_error- mse_GPS_error)/norm(mse_GPS_error))

    
if (norm(mse_lapl_error) < norm(mse_GPS_error)):    
    print ('MSE reduction after outliers removal: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error), '\n')

else:
    print ('MSE increment after outliers removal: ', norm(mse_lapl_error- mse_GPS_error)/norm(mse_GPS_error), '\n')    

'''
'''
if (norm(mse_MLE_error) < norm(mse_GPS_error)):    
    print ('MSE MLE reduction: ', norm(mse_MLE_error- mse_GPS_error)/norm(mse_GPS_error), '\n')

else:
    print ('MSE MLE increment: ', norm(mse_MLE_error- mse_GPS_error)/norm(mse_GPS_error), '\n')
    
'''
'''    
if (norm(max_opt_error_outliers) < norm(max_GPS_error)):
    print ('Max Laplacian reduction: ', norm(max_opt_error_outliers- max_GPS_error)/norm(max_GPS_error))
    
else:
    print ('Max Laplacian increment: ', norm(max_opt_error_outliers- max_GPS_error)/norm(max_GPS_error))


    
if (norm(max_lapl_BS_error) < norm(max_GPS_error)):    
    print ('Max BS reduction: ', norm(max_lapl_BS_error- max_GPS_error)/norm(max_GPS_error))

else:
    print ('Max BS increment: ', norm(max_lapl_BS_error- max_GPS_error)/norm(max_GPS_error))


if (norm(max_lapl_error) < norm(max_GPS_error)):    
    print ('MAX reduction after outliers removal: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error), '\n')

else:
    print ('MAX increment after outliers removal: ', norm(max_lapl_error- max_GPS_error)/norm(max_GPS_error), '\n') 
    
if (norm(max_MLE_error) < norm(max_GPS_error)):    
    print ('Max MLE reduction: ', norm(max_MLE_error- max_GPS_error)/norm(max_GPS_error), '\n')

else:
    print ('Max MLE increment: ', norm(max_MLE_error- max_GPS_error)/norm(max_GPS_error), '\n')    
'''    