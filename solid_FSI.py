#from matplotlib import cm ##cm = "colormap" for changing the 3d plot color palette
import math
import numpy as np                 #here we load numpy, calling it 'np' from now on
from scipy.sparse import csr_matrix
import config
import time
#import schemes as SC
import py_functions as CF
from scipy.interpolate import interp1d
#NR = config.get_round()

# nodule related
def normal_projection(s_n, Points,xg,yg,projections,center):
    nint = range(len(Points))
    for n in nint: 
      projections[n][:] = CF.get_proj(center[0],center[1],xg.loc(yg.get_i(Points[n],center[0],Lx,xg)),\
                                                          yg.loc(yg.get_j(Points[n],center[1],Ly)),R)
    return
def isinsolid(point,normal,s_n):
    center = config.get_centers(s_n)
    scalar_prod = CF.scalr_prod(normal,center[0],center[1],point[0],point[1])
    #return (scalar_prod-R <= 0.)
    return round(scalar_prod/R,6) <= 1.

def isinnodule(s_n,x,y):
  center = config.get_centers(s_n)
  DX = x-center[0]
  DY = y-center[1]
  r2 = DX**2+DY**2
  return round(r2 / R**2,6)<=1.

def pointsInSphere_first(xg,yg,s_n):
    inSphere = set()
    center = config.get_centers(s_n)
    xmin,xmax = xg.fluids()
    ymin,ymax = yg.fluids()
    for iX in range(xmin,xmax):
      for iY in range(ymin,ymax):
          x = xg.loc_rel(iX,center[0],Lx)
          y = yg.loc_rel(iY,center[1],Ly)
          if isinnodule(s_n,x,y):
              IJ = iX*yg.shape() + iY
              inSphere.add(IJ)
    return inSphere

def pointsInSphere(xg,yg,forced,pseudo,s_n):
    center = config.get_centers(s_n)
    newlyInSphere = set()
    cleared = set()
    for IJ in forced:
      if isinnodule(s_n,xg.loc_rel(yg.get_i(IJ, center[0], Lx,xg), center[0], Lx),\
                        yg.loc_rel(yg.get_j(IJ, center[1], Ly), center[1], Ly)):
        newlyInSphere.add(IJ)
    for IJ in pseudo:
      if not isinnodule(s_n,xg.loc_rel(yg.get_i(IJ, center[0], Lx,xg), center[0], Lx),\
                            yg.loc_rel(yg.get_j(IJ, center[1], Ly), center[1], Ly)):
        cleared.add(IJ)
    return newlyInSphere,cleared

def set_flags(xu,yu,xv,yv,xp,yp,s_n,bornes):
    ibm_u,ibm_v,ibm_p = config.get_ibms(s_n)
    if ibm_u is None: # first time step
      solidpops = pointsInSphere_first(xu,yu,s_n),\
                  pointsInSphere_first(xv,yv,s_n),\
                  pointsInSphere_first(xp,yp,s_n)
      config.set_solidPops(solidpops,s_n)
      potential_pseudo = solidpops
      nonewsfu,nonewsfv,nonewsfp = False,False,False
    else:
      newsolidu,clearedu = pointsInSphere(xu,yu,ibm_u[0],ibm_u[2],s_n)
      newsolidv,clearedv = pointsInSphere(xv,yv,ibm_v[0],ibm_v[2],s_n)
      newsolidp,clearedp = pointsInSphere(xp,yp,ibm_p[0],ibm_p[2],s_n)

      nonewsfu = len(newsolidu)==0 and len(clearedu)==0
      nonewsfv = len(newsolidv)==0 and len(clearedv)==0
      nonewsfp = len(newsolidp)==0 and len(clearedp)==0

      solidpops = config.get_solidPops(s_n)
      solidpops[0].difference_update(clearedu)
      solidpops[1].difference_update(clearedv)
      solidpops[2].difference_update(clearedp)
      solidpops[0].update(newsolidu)
      solidpops[1].update(newsolidv)
      solidpops[2].update(newsolidp)

      potential_pseudo = set(ibm_u[2]),set(ibm_v[2]),set(ibm_p[2])
      potential_pseudo[0].update(newsolidu)
      potential_pseudo[1].update(newsolidv)
      potential_pseudo[2].update(newsolidp)

    ibmu,ibmv,ibmp = ibm(xu,yu,potential_pseudo[0],s_n,0,(bornes[0]-1,bornes[2]),nonewsfu),\
                     ibm(xv,yv,potential_pseudo[1],s_n,1,(bornes[0],bornes[2]-1),nonewsfv),\
                     ibm(xp,yp,potential_pseudo[2],s_n,2,(bornes[0],bornes[2]),nonewsfp)
    #ibmu,ibmv,ibmp = ibm(xu,yu,potential_pseudo[0],s_n,0,[0,-1,1,-1],(bornes[0],bornes[2]-1),nonewsfu),\
    #                 ibm(xv,yv,potential_pseudo[1],s_n,1,[1,-1,0,-1],(bornes[0]-1,bornes[2]),nonewsfv),\
    #                 ibm(xp,yp,potential_pseudo[2],s_n,2,[1,-1,1,-1],(bornes[0],bornes[2]),nonewsfp)

    return ibmu,ibmv,ibmp

def ibm(xg,yg,potent_pseudo,s_n,field_numb,Or,nonewsf):

    center = config.get_centers(s_n)
    
    flags_slice = config.get_flags_slice(field_numb,s_n)
    
    solidpop = config.get_solidPops(s_n)[field_numb]
    #X,Y = np.array(map(yg.get_i,solidpop))-Or[0], np.array(map(yg.get_j,solidpop))-Or[1]
    X,Y = np.array([yg.get_i(x, center[0], Lx,xg) for x in solidpop])-Or[0], np.array([yg.get_j(x, center[1], Ly) for x in solidpop])-Or[1]
    flags_slice[X,Y] = -1

    if nonewsf:
      forcedPoints = config.get_forcing(field_numb,s_n)[0]
      pseudoPoints = config.get_pseudo(field_numb,s_n)[0]

      #X,Y = np.array(map(yg.get_i,forcedPoints))-Or[0], np.array(map(yg.get_j,forcedPoints))-Or[1]
      X,Y = np.array([yg.get_i(x, center[0], Lx,xg) for x in forcedPoints])-Or[0], np.array([yg.get_j(x, center[1], Ly) for x in forcedPoints])-Or[1]
      flags_slice[X,Y] = 1
      #X,Y = np.array(map(yg.get_i,pseudoPoints))-Or[0], np.array(map(yg.get_j,pseudoPoints))-Or[1]
      X,Y = np.array([yg.get_i(x, center[0], Lx,xg) for x in pseudoPoints])-Or[0], np.array([yg.get_j(x, center[1], Ly) for x in pseudoPoints])-Or[1]
      flags_slice[X,Y] = -2

    else:

      forcedPoints = []
      pseudoPoints = []
      # search forced and pseudo
      CF.search_forced_and_pseudo(potent_pseudo,flags_slice,pseudoPoints,forcedPoints,xg,yg,neighbours4,Or[0],Or[1],center,Lx,Ly)

    # PROJECTIONS
    if field_numb == 2:
      projectionsF = np.zeros((len(forcedPoints),4))
      normal_projection(s_n,forcedPoints,xg,yg,projectionsF,center)
    else: 
      projectionsF = None
      pseudoPoints = np.array(pseudoPoints,dtype=np.int)

    projectionsP = np.zeros((len(pseudoPoints),4))
    normal_projection(s_n,pseudoPoints,xg,yg,projectionsP,center)

    return forcedPoints,projectionsF,pseudoPoints,projectionsP

def pressureLinear_set(field,xp,yp,sF_coefsp,interp_points,DuDt,forcedP,proj,Or,center):
    #print DuDt
    nint = range(len(forcedP))
    #CF.set_field_neuman(forcedP,projections,p,nint,Or[0],Or[1],stress_coefp,interp_points,DuDt)
    for n in nint:
        pf = yp.get_i(forcedP[n], center[0], Lx,xp), yp.get_j(forcedP[n], center[1], Ly)
        p1, p2 = interp_points[n][0],interp_points[n][1]        
        #fF[n] = CF.interp_neuman(proj[n][2:],proj[n][:2],p1,p2,xg,yg,[-DuDt[0],-DuDt[1]],field,Or)
        field[pf[1]-Or[1],pf[0]-Or[0]] = (proj[n][2]*(-DuDt[0]) + proj[n][3]*(-DuDt[1]))*sF_coefsp[n][0] +\
                                         field[p1[1]-Or[1],p1[0]-Or[0]]*sF_coefsp[n][1] +\
                                         field[p2[1]-Or[1],p2[0]-Or[0]]*sF_coefsp[n][2] 
    return

def compute_M(od):
    cols = np.arange(od.size)
    return csr_matrix((cols, (od, cols)), shape=(np.amax(od)+1, od.size))

def get_indices_where_sparse(M,numb): return M[numb,:].data

colors=['r','g','b']

def linearizeVelocity_pseudoCNi(indice_field,xg,yg,s_n,Or,ax):
    # mirror point makes BL velocity look bad on fixed cylinder case

    center = config.get_centers(s_n)
    
    flags_slice = config.get_flags_slice(indice_field,s_n)
    flags_slice_P = config.get_flags_slice(2,s_n)

    forcedP,projections = config.get_pseudo(indice_field,s_n)
    nint = range(len(forcedP))

    fF = np.zeros(len(forcedP), dtype='int64,int64,int64,int64,int64,int64,float64,float64,float64')
    data = np.zeros(2*len(forcedP),dtype='float64')
    IJ   = np.zeros((2,2*len(forcedP)),dtype=np.int)

    fF8 = np.zeros(len(forcedP), dtype='float64')
    sF_coefs = np.empty((len(forcedP),3))
    sF_points = np.empty((len(forcedP),4))
    for n in nint:

        proj = projections[n][0], projections[n][1]
        n_x = projections[n][2]
        n_y = projections[n][3]

        pf = yg.get_i(forcedP[n],center[0],Lx,xg), yg.get_j(forcedP[n],center[1],Ly)
        anchor = pf
 
        layer = True
        coefs,goefs,p1,p2 = search_velocity(anchor,xg,yg,proj,flags_slice,Or,projections[n][2:4],layer)
        #FORCE
        sF_coefs[n,:] = goefs
        sF_points[n,:] = p1[0],p1[1],p2[0],p2[1]

        #sF_coefs[n,:] = np.round(sF_coefs[n,:],5)
        #coefs = np.round(coefs,5)

        #layer = True
        #coefs,goefs,p1,p2 = search_velocity(anchor,xg,yg,proj,flags_slice,Or,projections[n][2:4],layer)
        #BOUNDARY CONDITION
        fF[n] = pf[0],pf[1],p1[0],p1[1],p2[0],p2[1],\
                  coefs[1],coefs[2],coefs[0]

        pf_abs = yg.get_i_abs(forcedP[n]), yg.get_j_abs(forcedP[n])
        #pf = xg.ind_perio(pf[0]),yg.ind_perio(pf[1])

        p1[0] = xg.ind_perio(p1[0])#+pf_abs[0]-pf[0])
        p1[1] = yg.ind_perio(p1[1])#+pf_abs[1]-pf[1])
        p2[0] = xg.ind_perio(p2[0])#+pf_abs[0]-pf[0])
        p2[1] = yg.ind_perio(p2[1])#+pf_abs[1]-pf[1])
        #pf    = pf_abs

        IJ_f = pf_abs[1]*(xg.shape())+pf_abs[0]
        IJ_1 = p1[1]*(xg.shape())+p1[0]
        IJ_2 = p2[1]*(xg.shape())+p2[0]

        #if math.fabs(coefs[0])>300:
        if not ax is None:

          col='k'
          DY=0.
    #      if s_n==2: DY,col = +1.,'r'

          #print sF_coefs[n,:],coefs
          #print gn1*gn2
          import matplotlib.pyplot as plt    #here we load matplotlib, calling it 'plt'
          #pf = yg.get_i_abs(forcedP[n]), yg.get_j_abs(forcedP[n])
          plt.plot([xg.loc(pf[0]+2),xg.loc(pf[0]-2),xg.loc(pf[0]),xg.loc(pf[0])],[yg.loc(pf[1]),yg.loc(pf[1]),yg.loc(pf[1]+2),yg.loc(pf[1]-2)],'+',color='b')
          plt.plot([xg.loc(pf[0]+1),xg.loc(pf[0]-1),xg.loc(pf[0]),xg.loc(pf[0])],[yg.loc(pf[1]),yg.loc(pf[1]),yg.loc(pf[1]+1),yg.loc(pf[1]-1)],'+',color='b')
          plt.plot([xg.loc(p1[0]),xg.loc(pf_abs[0]),xg.loc(p2[0])],[yg.loc(p1[1])+DY,yg.loc(pf_abs[1])+DY,yg.loc(p2[1])+DY],color=col)
          plt.plot([xg.loc(p1[0]),xg.loc(p2[0]),CF.loc_perio(proj[0],Lx)],[yg.loc(p1[1])+DY,yg.loc(p2[1])+DY,CF.loc_perio(proj[1],Ly)+DY],'o',color='k')
          plt.plot(xg.loc(pf_abs[0]),yg.loc(pf_abs[1]),'o',color='g')
          plt.plot(CF.loc_perio(proj[0],Lx),CF.loc_perio(proj[1],Ly),'+',color='r')
          plt.plot([CF.loc_perio(proj[0],Lx),CF.loc_perio(proj[0]+dx*n_x,Lx)],[CF.loc_perio(proj[1],Ly),CF.loc_perio(proj[1]+dx*n_y,Ly)],color='k')

        #if np.amax(np.abs(np.array([gns,gn1,gn2])))>100.: print gns,gn1,gn2
        #if True or np.amax(np.abs(coefs))>2.: print coefs
        
        fF8[n] = coefs[0]
            # CNi

        data[2*n],IJ[0,2*n],IJ[1,2*n] = coefs[1],IJ_f,IJ_1
        data[2*n+1],IJ[0,2*n+1],IJ[1,2*n+1] = coefs[2],IJ_f,IJ_2
    #plt.show()
    solid = config.get_solidPops(s_n)[indice_field]
    indptr_penal, indices_penal, data_penal = np.zeros(xg.shape()*yg.shape()+1, dtype = np.int),[],[]
    for IJs in solid:
      #ps = yg.get_i(IJs,center[0],Lx), yg.get_j(IJs,center[1],Ly)
      ps = yg.get_i_abs(IJs), yg.get_j_abs(IJs)
        
      #print ps
      #print yg.get_i_abs(IJs), yg.get_j_abs(IJs)
      #line = CF.IJ_ij_xg(ps[0],ps[1],xg,yg)#ps[1]*xg.shape()+ps[0]
      line = ps[1]*xg.shape()+ps[0]
      indptr_penal[line+1:] += 5
      if ps[0] == 0:# and indice_field==0:
          indices_penal[indptr_penal[line]:indptr_penal[line]] = [line+1,line+xg.shape()-2,line+xg.shape(),line-xg.shape(),line]
      #elif ps[0] == nx-1 and indice_field==0: 
      #    indices_penal[indptr_penal[line]:indptr_penal[line]] = [line-xg.shape()+2,line-1,line+xg.shape(),line-xg.shape(),line]
      elif ps[1] == 0:# and indice_field==1:
          indices_penal[indptr_penal[line]:indptr_penal[line]] = [line+1,line-1,line+xg.shape(),line+(yg.shape()-2)*xg.shape(),line]
      #elif ps[1] == ny-1 and indice_field==1:
      #    indices_penal[indptr_penal[line]:indptr_penal[line]] = [line+1,line-1,line-(yg.shape()-2)*xg.shape(),line-xg.shape(),line]
      else:               
          indices_penal[indptr_penal[line]:indptr_penal[line]] = [line+1,line-1,line+xg.shape(),line-xg.shape(),line]
      data_penal[indptr_penal[line]:indptr_penal[line]] = [1.,1.,1.,1.,-4.]#[-1.,-1.,-1.,-1.,+4.]

    csr_penal = (0.5*nu/dx**2*np.array(data_penal, dtype=np.float),np.array(indices_penal, dtype=np.int),indptr_penal)
    mat_penal = csr_matrix(csr_penal, shape=(xg.shape()*yg.shape(),xg.shape()*yg.shape()))
    mat_ibm = csr_matrix((data, IJ), shape=(xg.shape()*yg.shape(),xg.shape()*yg.shape()))
    return fF,sF_coefs, mat_penal, mat_ibm, fF8, sF_points

def ghostcellCorr(xg,yg,s_n,Or,ax):

    center = config.get_centers(s_n)

    flags_slice = config.get_flags_slice(2,s_n)
    forced,projections = config.get_pseudo(2,s_n)

    indptr, indices, data = np.zeros((nx+1)*(ny+1)+1, dtype = np.int),[],[]
    indptr_penal, indices_penal, data_penal = np.zeros((nx+1)*(ny+1)+1, dtype = np.int),[],[]
    interp_points = np.zeros((len(forced),2,2),dtype=np.int)

    nint = range(len(forced))
    solid = config.get_solidPops(s_n)[2]
    sF_coefs = np.empty((len(forced),3))
    for n in nint:
        
        projection = projections[n]

        proj = projection[0], projection[1]
        n_x = projection[2]
        n_y = projection[3]

        pf = yg.get_i(forced[n],center[0],Lx,xg), yg.get_j(forced[n],center[1],Ly)

        anchor = pf

        layer = False
        coefs,p1,p2 = search_pressure(anchor,xg,yg,proj,flags_slice,Or,projections[n][2:4],layer)
        sF_coefs[n,:] = coefs
        interp_points[n][0][:] = p1
        interp_points[n][1][:] = p2

        #layer = True
        #coefs,p1,p2 = search_pressure(anchor,xg,yg,proj,flags_slice,Or,projections[n][2:4],layer)

        pf_abs = yg.get_i_abs(forced[n]), yg.get_j_abs(forced[n])
        #pf = xg.ind_perio(pf[0]),yg.ind_perio(pf[1])

        p1[0] = xg.ind_perio(p1[0])#+pf_abs[0]-pf[0])
        p1[1] = yg.ind_perio(p1[1])#+pf_abs[1]-pf[1])
        p2[0] = xg.ind_perio(p2[0])#+pf_abs[0]-pf[0])
        p2[1] = yg.ind_perio(p2[1])#+pf_abs[1]-pf[1])
        #pf    = pf_abs

        line = pf_abs[1]*(xg.shape())+pf_abs[0]
        IJ_1 = p1[1]*(xg.shape())+p1[0]
        IJ_2 = p2[1]*(xg.shape())+p2[0]

        if not ax is None:#np.amax(np.abs(coefs)) > 0.95:

          col='k'
          DY=0.
        #  if s_n==1: DY,col = -1.,'r'
 
          import matplotlib.pyplot as plt    #here we load matplotlib, calling it 'plt'
          plt.plot([xg.loc(pf[0]+2),xg.loc(pf[0]-2),xg.loc(pf[0]),xg.loc(pf[0])],[yg.loc(pf[1]),yg.loc(pf[1]),yg.loc(pf[1]+2),yg.loc(pf[1]-2)],'+',color='b')
          plt.plot([xg.loc(pf[0]+1),xg.loc(pf[0]-1),xg.loc(pf[0]),xg.loc(pf[0])],[yg.loc(pf[1]),yg.loc(pf[1]),yg.loc(pf[1]+1),yg.loc(pf[1]-1)],'+',color='b')
          plt.plot([xg.loc(p1[0]),xg.loc(pf[0]),xg.loc(p2[0])],[yg.loc(p1[1])+DY,yg.loc(pf[1])+DY,yg.loc(p2[1])+DY],color='k')
          plt.plot([xg.loc(p1[0]),xg.loc(p2[0]),CF.loc_perio(proj[0],Lx)],[yg.loc(p1[1]),yg.loc(p2[1]),CF.loc_perio(proj[1],Ly)],'o',color='k')
          plt.plot(xg.loc(pf[0]),yg.loc(pf[1]),'o',color='g')
          plt.plot(CF.loc_perio(proj[0],Lx),CF.loc_perio(proj[1],Ly),'+',color='r')
          plt.plot([CF.loc_perio(proj[0],Lx),CF.loc_perio(proj[0]+dx*n_x,Lx)],[CF.loc_perio(proj[1],Ly),CF.loc_perio(proj[1]+dx*n_y,Ly)],color='k')

        #line = CF.IJ_ij_xg(pf[0],pf[1],xg,yg)#pf[1]*(nx+1)+pf[0]
        #IJ_1 = CF.IJ_ij_xg(p1[0],p1[1],xg,yg)#p1[1]*(xg.shape())+p1[0]
        #IJ_2 = CF.IJ_ij_xg(p2[0],p2[1],xg,yg)#p2[1]*(xg.shape())+p2[0]
        #print coefs
        indptr[line+1:] += 2
        indices[indptr[line]:indptr[line]] = [IJ_1, IJ_2]
        data[indptr[line]:indptr[line]] = [-coefs[1],-coefs[2]] #Neumann

    if False: plt.show()
    for IJs in solid:
      #ps = yg.get_i(IJs,center[0],Lx), yg.get_j(IJs,center[1],Ly)
      ps = yg.get_i_abs(IJs), yg.get_j_abs(IJs)
      #line = CF.IJ_ij_xg(ps[0],ps[1],xg,yg)#ps[1]*xg.shape()+ps[0]
      line = ps[1]*xg.shape()+ps[0]
      #ps = yg.get_i(IJs), yg.get_j(IJs)
      #line = ps[1]*(nx+1)+ps[0]
      indptr_penal[line+1:] += 4
      indices_penal[indptr_penal[line]:indptr_penal[line]] = [line+1,line-1,line+(nx+1),line-(nx+1)]
      data_penal[indptr_penal[line]:indptr_penal[line]] = [0.25,0.25,0.25,0.25]#[-1.,-1.,-1.,-1.,+4.]

    return (np.array(data, dtype=np.float),np.array(indices, dtype=np.int),indptr),sF_coefs,\
           (np.array(data_penal, dtype=np.float),np.array(indices_penal, dtype=np.int),indptr_penal),\
           interp_points

def linearizeVelocity_pseudo_set(field,s_n,fFs,u_dirich,orx,ory,mpirank):
    for fF in fFs:
      CF.set_field_dirichlet2(field,fF[1]-ory,fF[0]-orx,fF[6],fF[3]-ory,fF[2]-orx,fF[7],fF[5]-ory,fF[4]-orx,fF[-1],u_dirich,mpirank)      
    return

def bornage(indice_field):
    if indice_field == 0: return [0,-1,1,-1]
    elif indice_field == 1: return [1,-1,0,-1]
    else: return [1,-1,1,-1]

def intersec(ps,pf,pp,normal):
    ambda = lambd(normal[0],normal[1],pp[0],pp[1],ps[0],ps[1],pf[0],pf[1])
    return resu(ambda,ps[0],pf[0]),resu(ambda,ps[1],pf[1])

def linearizeFlux(indice_field,tolin,xg,yg,xo,yo,xp,yp,pf,pline,cline):
    p1 = pline[0]
    p2 = pline[1]
    if indice_field == 0:
      if tolin[1] < yp.loc(pf[1]):
          p1[1] = pf[1]
          p2[1] = pf[1]-1
      else:
          p1[1] = pf[1]
          p2[1] = pf[1]+1
      if tolin[0] < xp.loc(pf[0]):       
          p1[0] = pf[0]-1
          p2[0] = pf[0]-1
      else:
          p1[0] = pf[0]
          p2[0] = pf[0]
      weight = (yg.loc(p2[1])-tolin[1])/(yg.loc(p2[1]) - yg.loc(p1[1]))
    if indice_field == 1:
      if tolin[0] < xp.loc(pf[0]):
          p1[0] = pf[0]
          p2[0] = pf[0]-1
      else:
          p1[0] = pf[0]
          p2[0] = pf[0]+1
      if tolin[1] < yp.loc(pf[1]):
          p1[1] = pf[1]-1
          p2[1] = pf[1]-1
      else:
          p1[1] = pf[1]
          p2[1] = pf[1]
      weight = (xg.loc(p2[0])-tolin[0])/(xg.loc(p2[0]) - xg.loc(p1[0]))
    cline[:] = weight,1.-weight
    return

def search_velocity(anchor,xg,yg,proj,flags_slice,Or,normal,layer):
      return findAp1p2_velocity(proj,xg,yg,anchor,flags_slice,Or,normal,layer)

def search_pressure(anchor,xg,yg,proj,flags_slice,Or,normal,layer):
      return findAp1p2_pressure(proj,xg,yg,anchor,flags_slice,Or,normal,layer)

def findAp1p2_pressure(root,xg,yg,anchor,flags_slice,Or,normal,layer):
        #DX = xg.loc(anchor[0])-root[0]
        #DY = xg.loc(anchor[1])-root[1]
        #r2 = math.sqrt(DX**2+DY**2)/config.dx
        cs,c1,c2,I1,J1,I2,J2 = CF.incr_list_pressure(root[0],root[1],anchor[0],anchor[1],flags_slice,Or[0],Or[1],xg,yg,normal,layer,0.0)
        #I1,J1,(I2,J2, (cs_,c1_,c2_, gn1_,gn2,_gns_)) = CF.incr_list_velocity(root[0],root[1],anchor[0],anchor[1],flags_slice,Or[0],Or[1],xg,yg,normal,0.25)
        #cs,c1,c2 = CF.coefs_P(xg.loc(I1),yg.loc(J1),xg.loc(I2),yg.loc(J2),xg.loc(anchor[0]),yg.loc(anchor[1]),normal[0],normal[1])
        return [cs,c1,c2],[I1,J1],[I2,J2]

def findAp1p2_velocity(root,xg,yg,anchor,flags_slice,Or,normal,layer):
        r2 = CF.dist(root[0],root[1],xg.loc(anchor[0]),yg.loc(anchor[1]))/config.dx
        I1,J1,(I2,J2, (cs,c1,c2,gns,gn1,gn2)) = CF.incr_list_velocity(root[0],root[1],anchor[0],anchor[1],flags_slice,Or[0],Or[1],xg,yg,normal,layer,0.5*r2)
        #cs,c1,c2,I1,J1,I2,J2 = CF.incr_list_pressure(root[0],root[1],anchor[0],anchor[1],flags_slice,Or[0],Or[1],xg,yg,normal)
        #cs,c1,c2,gns,gn1,gn2 = CF.coefs_V(xg.loc(I1),yg.loc(J1),xg.loc(I2),yg.loc(J2),xg.loc(anchor[0]),yg.loc(anchor[1]),root[0],root[1],normal[0],normal[1])
        return [cs,c1,c2],[gns,gn1,gn2],[I1,J1],[I2,J2]

def linearizePressure_surface(interp_points,sF_coefsp,field,DuDt,Or,s_n):
    forced, proj = config.get_pseudo(2,s_n)
    fF = np.zeros(len(forced))
    for n in range(len(forced)):
        forcedIJ = forced[n]
        p1, p2 = interp_points[n][0],interp_points[n][1]
        #fF[n] = CF.interp_neuman(proj[n][2:],proj[n][:2],p1,p2,xg,yg,[-DuDt[0],-DuDt[1]],field,Or)
        fF[n] = (proj[n][2]*(-DuDt[0]) + proj[n][3]*(-DuDt[1]))*sF_coefsp[n][0] + field[p1[1]-Or[1],p1[0]-Or[0]]*sF_coefsp[n][1] + field[p2[1]-Or[1],p2[0]-Or[0]]*sF_coefsp[n][2]
        #fF[n][1] = proj[n][0]
        #fF[n][2] = proj[n][1]
    return fF

def surfaceStress(fF,sF_coefs,v,vs,orx,ory):
    N = sF_coefs.shape[0]
    Stress = np.zeros(N)

    #Stress = np.zeros((N,2))
    #U = np.zeros(3)
    for n in range(N):
        #U[:] = [vs,v[fF[n][1]-ory,fF[n][0]-orx],v[fF[n][3]-ory,fF[n][2]-orx]]
        #Stress[n,:] = np.dot(sF_coefs[n],U)
        Stress[n] = sF_coefs[n][0]*vs+sF_coefs[n][1]*v[fF[n][1]-ory,fF[n][0]-orx]+sF_coefs[n][2]*v[fF[n][3]-ory,fF[n][2]-orx]
        #if np.amax(np.abs(sF_coefs[n]))>500.: Stress[n] = 0.

#pf[0],pf[1],p1[0],p1[1],p2[0],p2[1],coefs[1],coefs[2],coefs[0]
        #print fF.shape#np.amax(np.abs(sF_coefs[n])),Stress[n]#*vs+sF_coefs[n][1]*v[fF[n][1]-ory,fF[n][0]-orx]+sF_coefs[n][2]*v[fF[n][3]-ory,fF[n][2]-orx]
    return Stress
    
def totalForce(u,v,p,xp,yp,s_n,ffsf,ffsp,origine):
    surfaceStressFieldu,curvilinearU,\
    surfaceStressFieldv,curvilinearV,\
    surfaceStressFieldp,curvilinearP,\
    computationPoints = preprocess_pseudo(u,v,p,xp,yp,s_n,ffsf,ffsp,origine)
    forces = integrate_stress(surfaceStressFieldu,curvilinearU,\
                              surfaceStressFieldv,curvilinearV,\
                              surfaceStressFieldp,curvilinearP,\
                              computationPoints)
    return forces
    

def interp_function(surfaceStressField,curvilinear):

    agmi = np.argmin(curvilinear)
    agma = np.argmax(curvilinear)
    x = np.zeros(len(curvilinear)+2)
    y = np.zeros(len(curvilinear)+2)
    x[1:-1] = curvilinear
    y[1:-1] = surfaceStressField
    x[0] = curvilinear[agma]-2.*math.pi
    y[0] = surfaceStressField[agma]
    x[-1] = curvilinear[agmi]+2.*math.pi
    y[-1] = surfaceStressField[agmi]

    return interp1d(x, y, kind='linear')
 
def integrate_stress(surfaceStressFieldu,curvilinearU,\
                     surfaceStressFieldv,curvilinearV,\
                     surfaceStressFieldp,curvilinearP,\
                     computationPoints):

    dAngle = 2.*math.pi/len(computationPoints)
    return dAngle*R*CF.integrate_stress(np.cos(computationPoints),np.sin(computationPoints),\
                        surfaceStressFieldu,curvilinearU,\
                        surfaceStressFieldv,curvilinearV,\
                        surfaceStressFieldp,curvilinearP,\
                        computationPoints,math.pi,nu,rho)

def preprocess_pseudo(u,v,p,xp,yp,s_n,ffsf,ffsp,origine):

    [dusdt,dvsdt],[us,vs] = config.get_solidVeloPosit(s_n)[:-1]

    proju = config.get_pseudo(0,s_n)[1]
    projv = config.get_pseudo(1,s_n)[1]
    projp = config.get_pseudo(2,s_n)[1]
    # arbitrarily:
    N = int(2*math.pi*R/dx)
    dAngle = 2.*math.pi/N
    computationPoints = np.arange(N)*dAngle  

    curvilinearU = curvilinear(proju)
    curvilinearV = curvilinear(projv)
    curvilinearP = curvilinear(projp)

    fFu,sF_coefsu,fFv,sF_coefsv = ffsf
    fFp,sF_coefsp = ffsp

    surfaceStressFieldu = surfaceStress(fFu,sF_coefsu,u,us,origine[0]-1,origine[1])
    surfaceStressFieldv = surfaceStress(fFv,sF_coefsv,v,vs,origine[0],origine[1]-1)
    DuDt = [dusdt,dvsdt]
    surfaceStressFieldp = linearizePressure_surface(fFp,sF_coefsp,p,DuDt,origine,s_n)
    return surfaceStressFieldu,curvilinearU,\
           surfaceStressFieldv,curvilinearV,\
           surfaceStressFieldp,curvilinearP,\
           computationPoints

def curvilinear(proj):
    curvCoords = np.zeros(len(proj),dtype=np.float)
    CF.curvilinear(proj,curvCoords,math.pi)    
    return curvCoords

#parameters
Ly,Lx,ny,nx,dx,dy,nu,rho,Ulid = config.sharedparameters()
R,ms,K,gamma,centers0 = config.get_sphereParameters()

neighbours4 = np.zeros((4,2),dtype = np.int8)
neighbours4[0][0] = 0
neighbours4[0][1] = 1
neighbours4[1][0] = 1
neighbours4[1][1] = 0
neighbours4[2][0] = 0
neighbours4[2][1] = -1
neighbours4[3][0] = -1
neighbours4[3][1] = 0

