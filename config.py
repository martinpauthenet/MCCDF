import math
import numpy as np                 #here we load numpy, calling it 'np' from now on
from mpi4py import MPI
import generate_centers

comm = MPI.COMM_WORLD
mpiRank = comm.Get_rank()
cpu = comm.Get_size()
def get_nameLoad(): return 'fields/load/20000_'
#def get_nameLoad(): return 'fields/save/23000_'
def get_nameSvg():  return 'fields/save/'
def get_save_runtime(): return (time%1000)==0

Reynolds = 100.
def get_nite(): return 200000
theta = 0.#5*math.pi
case = '25e06'
gravi = float(case[0]+'.'+case[1:-2]+'-'+case[-2:])
def get_gravity(): return gravi*math.cos(theta),gravi*math.sin(theta)

def get_load(): return False
def get_load_coarse(): return True
def load_kinematix(): return True
def get_follow(): return True # MECHANICS !!!
def get_force(): return True
def get_save(): return True
number_of_sides = 2
number_of_length = 2
L = 1.
Ly = L*number_of_sides
zoom = 1.
Ncells = int(zoom*50)
variable_time_step = True

LxLy = float(number_of_length)/float(number_of_sides)
Lx = Ly*LxLy
ny = (Ncells)*number_of_sides+1
nx = int(LxLy*(ny-1)+1)
#print Lx,nx,Ly,ny
R = L/(4.*zoom)
Nsphere = number_of_sides*number_of_length
centers0 = np.zeros((Nsphere,2))
generate_centers.array(centers0,R,Lx,Ly,number_of_sides,number_of_length)
#generate_centers.cheval(centers0,R,Lx,Ly,number_of_sides,number_of_length)

dx = Lx/(nx-1)
dy = Ly/(ny-1)
nu = 1e-5
rho = 1.0
time = 0
TS = 0.

#sphere anchor
#K = 2.4039903863e-07 #f=0.25  # link rigidity
K = 6.15421538893e-05#f=4.  # link rigidity
#sqrtKm = 4e-3 #(~U/D)
mstar=1.
m_p = math.pi*R**2*mstar*rho
gamma = 0.   # dissipation displacement

###storage
solu = [None]*Nsphere
solv = [None]*Nsphere
solp = [None]*Nsphere
#forced projectionsF pseudo projectionsP
ibm_u = [None]*Nsphere
ibm_v = [None]*Nsphere
ibm_p = [None]*Nsphere

Ulid = Reynolds*nu/(2.*R)
Usphere = Ulid
frontal_section = (1.-(math.pi*R**2)/(L**2))

CFL_MAX = 1
dt = CFL_MAX*min(dx/Usphere,dx**2/nu)#0.001*550/Reynolds
dt_old = dt
dt_min = dt*1e-10
def set_dt(cfl):
  if variable_time_step:
    global dt
    global dt_old
    dt_old = dt
    if dt > 1.1*CFL_MAX/cfl*dt: dt = CFL_MAX/cfl*dt
    elif dt < 0.9*CFL_MAX/cfl*dt: dt = CFL_MAX/cfl*dt
    dt = min(dt,dx**2/nu)
    return dt < dt_min
  return False

centers = np.zeros((Nsphere,2))
centers[:,:] = centers0
#print centers[2]
dvs = np.zeros((Nsphere,2))
vs = np.zeros((Nsphere,2))
if get_load():
  path_load=get_nameLoad()
  if load_kinematix():
   centers[:,:] = np.loadtxt(path_load+'centers')
   vs[:,:] = np.loadtxt(path_load+'vs')
   dvs[:,:] = np.loadtxt(path_load+'dvs')

flag_slices = [None]*Nsphere

def set_solidPops(solidpops,s_n):
    global solu
    global solv
    global solp
    solu[s_n],solv[s_n],solp[s_n] = solidpops

def get_solidPops(s_n):
    return solu[s_n],solv[s_n],solp[s_n]

def sharedparameters():
    return Ly,Lx,ny,nx,dy,dx,nu,rho,Ulid

def get_sphereParameters():
    return R,m_p,K,gamma,centers0

def get_solidVeloPosit(s_n):
    return dvs[s_n],vs[s_n],centers[s_n]

def get_centers(s_n): 
  return centers[s_n]
def get_centers0(s_n): 
  return centers0[s_n]

def set_solidVeloPosit(val_dvs,val_vs,val_center,s_n):
    global dvs
    global vs
    global centers
    #print val_dvs,val_vs,val_center,s_n
    dvs[s_n],vs[s_n],centers[s_n] = val_dvs,val_vs,val_center

def set_solidPoints(solid_list,s_n):
    global ibm_u
    global ibm_v
    global ibm_p
    ibm_u[s_n],ibm_v[s_n],ibm_p[s_n] = solid_list

def get_ibms(s_n):
    return ibm_u[s_n],ibm_v[s_n],ibm_p[s_n]

if mpiRank == 0:
  print 'number of procs: '+str(cpu)
  print 'cells per diameter: '+str(int(2*R/dx))
  print 'void',frontal_section
  print 'nite: ', get_nite()
  print 'follow: ', get_follow()
  print 'number_of_sides: ', number_of_sides
  print 'number_of_length: ', number_of_length
  print 'Reynolds: ', Reynolds
  print 'Ncells: ', Ncells
  print 'pour JADIM'
  print 'K =', str(K)
  print 'rho_p = ', str(m_p/(math.pi*R**2*1.)) # for the cylinder in Jadim is of lenght 1. in the spanwise direction
  print 'dt = ', str(dt)

def incTime():
    global time
    global TS
    TS+=dt
    time+=1

def getTime():
    return time
    
def numb_sph(): return Nsphere

def get_pseudo(indice_field,s_n):
    ibm = [ibm_u[s_n],ibm_v[s_n],ibm_p[s_n]][indice_field]
    return ibm[2],ibm[3]

def get_forcing(indice_field,s_n):
    ibm = [ibm_u[s_n],ibm_v[s_n],ibm_p[s_n]][indice_field]
    return ibm[0],ibm[1]

def set_flags(local_sn):
  window = get_w()
  for s_n in local_sn:
    #flag_slices[s_n] = np.zeros((window,window+1),dtype=np.int8),np.zeros((window+1,window),dtype=np.int8),np.zeros((window,window),dtype=np.int8)
    flag_slices[s_n] = np.zeros((window+1,window),dtype=np.int8),np.zeros((window,window+1),dtype=np.int8),np.zeros((window,window),dtype=np.int8)
  return
def get_flags_slice(field_numb,s_n): return flag_slices[s_n][field_numb][:,:]
def reset_flags(local_sn):
  for s_n in local_sn:
    for field in range(3):
      flag_slices[s_n][field][:,:] = 0.
  return

def get_Reynolds(U): return 2*R*U/nu

def plotSpheres(fig,plt,fillin):
  for sphere_num in range(numb_sph()):
    center = centers[sphere_num]
    plotSphere(fig,plt,fillin,center)
  return

def plotSphere(fig,plt,fillin,center):    
    circle=plt.Circle((center[0],center[1]),R,color='k',fill=fillin)
    fig.add_artist(circle)
    px,py=center
    while px<0.:px+=Lx
    while px>=Lx:px-=Lx
    circle = plt.Circle((px, py), R, color='k', clip_on=True,fill=fillin)
    fig.add_artist(circle)
    if px+R>Lx:
        circle = plt.Circle((px-Lx, py), R, color='k', clip_on=True,fill=fillin)
        fig.add_artist(circle)
    if px-R<0.:
        circle = plt.Circle((px+Lx, py), R, color='k', clip_on=True,fill=fillin)
        fig.add_artist(circle)
    if py+R>Ly:
        circle = plt.Circle((px, py-Ly), R, color='k', clip_on=True,fill=fillin)
        fig.add_artist(circle)
    if py-R<0.:
        circle = plt.Circle((px, py+Ly), R, color='k', clip_on=True,fill=fillin)
        fig.add_artist(circle)
    return

def get_bornes(s_n):
    center = centers[s_n]
    window = get_w()
    xmin,ymin = int((center[0]-window/2.*dx+0.5*dx)/dx)+1, int((center[1]-window/2.*dx+0.5*dx)/dx)+1
    xmax,ymax = xmin+window,ymin+window
    return np.array([xmin,xmax,ymin,ymax],dtype=np.int)

def get_w(): return int(2.*(R+3.*dx)/dx)

def get_global_sphere_number(CPU):
    to_distrib = np.arange(Nsphere)
    global_sphere_number = np.array_split(to_distrib,CPU)
    return  global_sphere_number

#def get_round():return 10

ipu = [None]*Nsphere
ipv = [None]*Nsphere
ipp = [None]*Nsphere

def get_ip(indice_field,s_n):
  return [ipu,ipv,ipp][indice_field][s_n]

def set_ip(indice_field,s_n,toset):
  global ipu
  global ipv
  global ipp
  [ipu,ipv,ipp][indice_field][s_n] = toset
  return

cutcellspartner=[None]*Nsphere
def get_cutcellspartner(s_n):
  return cutcellspartner[s_n]

def set_cutcellspartner(s_n,toset):
  global cutcellspartner
  cutcellspartner[s_n] = toset
  return

def get_ninterp():
  return 2
