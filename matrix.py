#from mpl_toolkits.mplot3d import Axes3D    #Library required for projected 3d plots

import math
import numpy as np                 #here we load numpy, calling it 'np' from now on
from scipy.sparse import csc_matrix, csr_matrix, linalg as sla, identity
import config

Ly,Lx,ny,nx,dy,dx,nu,rho,Ulid = config.sharedparameters()
#parameters
shapePDP = (nx+1)*(ny+1), (nx+1)*(ny+1)

def poisson_permanent():
    
    nonzeros = 5*(ny-1)*(nx-1) + 4*(ny-1) + 4*(nx-1) + 4 #+ 4

    data = np.zeros(nonzeros)
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0 
    # bulk
    for j in range(1,ny):
        for i in range(1,nx):
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = (j+1)*(nx+1)+i
                n += 1 
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = (j-1)*(nx+1)+i
                n += 1
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+(i+1)
                n += 1
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+(i-1)
                n += 1
                data[n] = -4.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+i
                n += 1 
    data[:] *= -0.25

    #periodic
    i = 0 
    for j in range(1,ny):
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+i
                n += 1
                data[n] = -1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+(nx-1)
                n += 1
    i = nx 
    for j in range(1,ny):
                  data[n] = 1.
                  I[n] = j*(nx+1)+i
                  J[n] = j*(nx+1)+i
                  n += 1
                  data[n] = -1.
                  I[n] = j*(nx+1)+i
                  J[n] = j*(nx+1)+1
                  n += 1
    j = 0 
    for i in range(1,nx):
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+i
                n += 1
                data[n] = -1.
                I[n] = j*(nx+1)+i
                J[n] = (ny-1)*(nx+1)+i
                n += 1
    j = ny
    for i in range(1,nx):
                data[n] = 1.
                I[n] = j*(nx+1)+i
                J[n] = j*(nx+1)+i
                n += 1
                data[n] = -1.
                I[n] = j*(nx+1)+i
                J[n] = (1)*(nx+1)+i
                n += 1
    for i in [0,nx]:
     for j in  [0,ny]:
      data[n] = 1.
      I[n] = j*(nx+1)+i
      J[n] = j*(nx+1)+i
      n += 1
    #i,j=0,0
    #data[n] = -1.
    #I[n] = j*(nx+1)+i
    #J[n] = (ny-1)*(nx+1)+(nx-1)
    #n += 1
    #i,j=0,ny
    #data[n] = -1.
    #I[n] = j*(nx+1)+i
    #J[n] = (1)*(nx+1)+(nx-1)
    #n += 1
    #i,j=nx,0
    #data[n] = -1.
    #I[n] = j*(nx+1)+i
    #J[n] = (ny-1)*(nx+1)+(1)
    #n += 1
    #i,j=nx,ny
    #data[n] = -1.
    #I[n] = j*(nx+1)+i
    #J[n] = (1)*(nx+1)+(1)
    #n += 1

#    return csr_matrix((np.zeros(0), np.zeros((2,0))), shape=shapePDP)+identity((nx+1)*(ny+1))
    return csr_matrix((data, IJ), shape=shapePDP)

def poisson_c(csrCC, csrGC, csrGC_penal,local_sphere_numbers):
    mat = csr_matrix(shapePDP, dtype = np.float)
    for n in local_sphere_numbers:
      for csr in [csrCC[n], csrGC[n], csrGC_penal[n]]: mat = mat + csr_matrix(csr, shape=shapePDP)
    return mat

def add_csr(csrs,shapexy):
    mat = csr_matrix((shapexy,shapexy), dtype = np.float)
    for csr in csrs: 
      mat = mat + csr_matrix(csr, shape=(shapexy,shapexy))
    return mat

def matrixUt():
    
    nonzeros = 5*(ny-1)*(nx-2) + 2*5*(ny-1)
            
    data = np.zeros(nonzeros) 
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0
  
# bulk
    for j in range(1,ny):
        for i in range(1,nx-1):
            data[n] = -4.
            I[n] = j*nx+i
            J[n] = j*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j+1)*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j-1)*nx+i
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i+1)
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i-1)
            n += 1
# bcs periodic
    i = 0
    for j in range(1,ny):
            data[n] = -4.
            I[n] = j*nx+i
            J[n] = j*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j+1)*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j-1)*nx+i
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i+1)
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(nx-2)
            n += 1
    i = nx-1
    for j in range(1,ny):
            data[n] = -4.
            I[n] = j*nx+i
            J[n] = j*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j+1)*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j-1)*nx+i
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(1)
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i-1)
            n += 1
    
    shape = nx*(ny+1),nx*(ny+1)
    At = csc_matrix((data, IJ), shape=shape)
    return At

def Fbcu_permanent():

    nonzeros = 5*1*(ny-1)+5*(ny-1)*(nx-2) #+ 5*(ny-1)
    data = np.zeros(nonzeros) 
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0

    # bulk
    for j in range(1,ny):
        for i in range(1,nx-1):
            data[n] = -4.
            I[n] = j*nx+i
            J[n] = j*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j+1)*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j-1)*nx+i
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i+1)
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i-1)
            n += 1
    # bcs periodic
    i = 0
    for j in range(1,ny):
            data[n] = -4.
            I[n] = j*nx+i
            J[n] = j*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j+1)*nx+i
            n += 1
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = (j-1)*nx+i
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(i+1)
            n += 1 
            data[n] = 1.
            I[n] = j*nx+i
            J[n] = j*nx+(nx-2)
            n += 1
    #i = nx-1
    #for j in range(1,ny):
    #        data[n] = -4.
    #        I[n] = j*nx+i
    #        J[n] = j*nx+i
    #        n += 1
    #        data[n] = 1.
    #        I[n] = j*nx+i
    #        J[n] = (j+1)*nx+i
    #        n += 1
    #        data[n] = 1.
    #        I[n] = j*nx+i
    #        J[n] = (j-1)*nx+i
    #        n += 1 
    #        data[n] = 1.
    #        I[n] = j*nx+i
    #        J[n] = j*nx+(1)
    #        n += 1 
    #        data[n] = 1.
    #        I[n] = j*nx+i
    #        J[n] = j*nx+(i-1)
    #        n += 1

    data *= -0.5*nu/dx**2

    # periodicity
    nonzeros_per = 2*(nx-1) + (ny+1)
    data_per = np.zeros(nonzeros_per)
    IJ_per = np.zeros((2,nonzeros_per),dtype=np.uint64)
    I = IJ_per[0]
    J = IJ_per[1]
    n = 0
    j = 0
    for i in range(0,nx-1):
        data_per[n] = +1.
        I[n] = j*nx+i
        J[n] = (ny-1)*nx+i
        n += 1
    j = ny
    for i in range(0,nx-1):
        data_per[n] = +1.
        I[n] = j*nx+i
        J[n] = (1)*nx+i
        n += 1
    i = nx-1
    for j in range(0,ny+1):
        data_per[n] = +1.
        I[n] = j*nx+i
        J[n] = j*nx+(0)
        n += 1
    
    shape = nx*(ny+1),nx*(ny+1)
    At = csc_matrix((data, IJ), shape=shape)
    A_per = identity(shape[0]) - csc_matrix((data_per, IJ_per), shape=shape)

    return At, A_per

def matrixVt():
    
    nonzeros = 5*(ny)*(nx-1)
 
    data = np.zeros(nonzeros) 
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0  

    # bulk
    for j in range(1,ny-1):
        for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1      
# bcs
    j=0
    for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (ny-2)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1      
    j=ny-1
    for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1      

    shape = ny*(nx+1),ny*(nx+1)
    At = csc_matrix((data, IJ), shape=shape) 

    return At

def Fbcv_permanent():
    
    nonzeros = 5*(nx-1)+5*(ny-2)*(nx-1) #+ 5*(nx-1)

    data = np.zeros(nonzeros) 
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0
    # bulk
    for j in range(1,ny-1):
        for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1      
# bcs
    j=0
    for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (ny-2)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1      
    #j=ny-1
    #for i in range(1,nx):
    #        data[n] = -4.
    #        I[n] = j*(nx+1)+i
    #        J[n] = j*(nx+1)+i
    #        n+=1
    #        data[n] = 1.
    #        I[n] = j*(nx+1)+i
    #        J[n] = (1)*(nx+1)+i
    #        n+=1  
    #        data[n] = 1.
    #        I[n] = j*(nx+1)+i
    #        J[n] = (j-1)*(nx+1)+i
    #        n+=1  
    #        data[n] = 1.
    #        I[n] = j*(nx+1)+i
    #        J[n] = j*(nx+1)+(i+1)
    #        n+=1  
    #        data[n] = 1.
    #        I[n] = j*(nx+1)+i
    #        J[n] = j*(nx+1)+(i-1)
    #        n+=1      

    data *= -0.5*nu/dx**2

    nonzeros_per = 2*(ny-1) + (nx+1)
    data_per = np.zeros(nonzeros_per)
    IJ_per = np.zeros((2,nonzeros_per),dtype=np.uint64)
    I = IJ_per[0]
    J = IJ_per[1]
    n = 0
    # periodicity 
    i = 0
    for j in range(0,ny-1):
        data_per[n] = +1.
        I[n] = j*(nx+1)+i
        J[n] = j*(nx+1)+(nx-1)
        n+=1
    i = nx
    for j in range(0,ny-1):
        data_per[n] = +1.
        I[n] = j*(nx+1)+i
        J[n] = j*(nx+1)+(1)
        n+=1
    j = ny-1
    for i in range(0,nx+1):
        data_per[n] = +1.
        I[n] = j*(nx+1)+i
        J[n] = (0)*(nx+1)+i
        n+=1
    
    shape = ny*(nx+1),ny*(nx+1)
    At = csc_matrix((data, IJ), shape=shape)    
    A_per = identity(shape[0]) - csc_matrix((data_per, IJ_per), shape=shape)
    return At, A_per

def laplacianPsi():
    
    nonzeros = 5*(ny-1)*(nx-1) + 2*5*(ny-1) + 2*5*(nx-1)
 
    data = np.zeros(nonzeros)
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0  

    # bulk
    for j in range(1,ny):
        for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(i-1)
            n+=1
# bcs
    for j,jeff in zip([0,ny],[ny-1,1]):
        for i in range(1,nx):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (jeff+1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (jeff-1)*(nx+1)+i
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+(i+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+(i-1)
            n+=1

    for i,ieff in zip([0,nx],[nx-1,1]):
        for j in range(1,ny):
            data[n] = -4.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+ieff
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+ieff
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+ieff
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(ieff+1)
            n+=1  
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(ieff-1)
            n+=1     
    shape = (ny+1)*(nx+1),(ny+1)*(nx+1)
    At = csc_matrix((data, IJ), shape=shape) 

    return At

def BCSlaplacianPsi():

    nonzeros = 2*6*(ny-1) + 2*6*(nx-1) + 4

    data = np.zeros(nonzeros)
    IJ = np.zeros((2,nonzeros),dtype=np.uint64)
    I = IJ[0]
    J = IJ[1]
    n = 0

# bcs
    for j,jeff in zip([0,ny],[ny-1,1]):
        for i in range(1,nx):
            data[n] = 4. #penalize
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = -8.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (jeff+1)*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (jeff-1)*(nx+1)+i
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+(i+1)
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = jeff*(nx+1)+(i-1)
            n+=1

    for i,ieff in zip([0,nx],[nx-1,1]):
        for j in range(1,ny):
            data[n] = 4. #penalize
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+i
            n+=1
            data[n] = -8.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+ieff
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j+1)*(nx+1)+ieff
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = (j-1)*(nx+1)+ieff
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(ieff+1)
            n+=1
            data[n] = 1.
            I[n] = j*(nx+1)+i
            J[n] = j*(nx+1)+(ieff-1)
            n+=1

    for i in [0,nx]:
      for j in  [0,ny]:
        data[n] = 4.
        I[n] = j*(nx+1)+i
        J[n] = j*(nx+1)+i
        n += 1

    shape = (ny+1)*(nx+1),(ny+1)*(nx+1)
    At = csc_matrix((data, IJ), shape=shape)

    return At
