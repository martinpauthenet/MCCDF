#from mpl_toolkits.mplot3d import Axes3D    #Library required for projected 3d plots

import math
import numpy as np                 #here we load numpy, calling it 'np' from now on
import time, sys                   #and load some utilities
from scipy.sparse import csc_matrix, linalg as sla, identity
import solid_FSI
import config
import matrix
import random
import py_functions as CF

def intersec(ps,pf,pp,normal):
    ambda = CF.lambd(normal[0],normal[1],pp[0],pp[1],ps[0],ps[1],pf[0],pf[1])
    return CF.resu(ambda,ps[0],pf[0]),CF.resu(ambda,ps[1],pf[1])

def init_tables(line,is_coef_written, coefs):
        is_coef_written[:], coefs[:] = False, 0.
        #line = J*(nx+1)+1
        written_places = [line+1,line-1,line+(nx+1),line-(nx+1),line]
        is_coef_written[written_places] = True,True,True,True,True
        coefs[written_places] = -1.,-1.,-1.,-1.,+4.
        return written_places

def reshape_poisson(xu,yu,xv,yv,xp,yp,rsc,slw,s_n,fFu,fFv,Or,ax):

    center = config.get_centers(s_n)

    is_coef_written = np.zeros(((nx+1)*(ny+1),), dtype = np.bool_)
    coefs = np.zeros((nx+1)*(ny+1))

    pseudou = config.get_pseudo(0,s_n)[0]
    flagsu = config.get_flags_slice(0,s_n)
    pseudov = config.get_pseudo(1,s_n)[0]
    flagsv = config.get_flags_slice(1,s_n)

    indptr, indices, data = np.zeros(((nx+1)*(ny+1)+1,), dtype = np.int), [], []
    forcedp = config.get_forcing(2,s_n)[0]
    projections = config.get_forcing(2,s_n)[1]

    coefs_for_psisource = [None]*len(forcedp)
    # local variables
    normal_cell = np.zeros(2,dtype=np.float)
    tolin = np.zeros(2,dtype=np.float)
    anchor = np.zeros(2,dtype=np.int)
    pline = np.zeros((2,2),dtype=np.int)
    cline = np.zeros(2,dtype=np.float)
    #NR=config.get_round()
    for n in range(len(forcedp)):
        if rsc[n] is None : ' not a cut-cell '# not a cut-cell i
        else: # cut-cell
            IJ = forcedp[n]

    #        print center,yp.get_j_abs(IJ),yp.get_j(IJ,center[1],Ly)

            cell = np.array(rsc[n])
            solidwall = slw[n]

            written_places = init_tables(CF.IJ_ij_xg(yp.get_i_abs(IJ),yp.get_j_abs(IJ),xp,yp),is_coef_written, coefs)
            _vsx, _vsy, _u, _ui, _uj, _v, _vi, _vj = [], [], [], [], [], [], [], []

            barycentre = [np.sum(cell[:,0])/cell.shape[0],np.sum(cell[:,1])/cell.shape[0]]
            if not ax is None:
              import matplotlib.pyplot as plt    #here we load matplotlib, calling it 'plt'
              plt.plot(barycentre[0],barycentre[1],'+',color='m')

            for npoint in range(len(cell)):
                p1 = cell[npoint-1]
                p2 = cell[npoint]
                d = CF.dist(p1[0],p1[1],p2[0],p2[1])

                if not ax is None:
                  plt.plot([p1[0],p2[0]],[p1[1],p2[1]],color='g')
                  plt.plot([p1[0],p2[0]],[p1[1],p2[1]],'o',color='r')

                if not d == 0.:
                 #normal_cell[:] = [round(-(p1[1]-p2[1])/d,10),round((p1[0]-p2[0])/d,10)]
                 normal_cell[:] = [-(p1[1]-p2[1])/d,(p1[0]-p2[0])/d]
                 tolin[:] = [(p1[0]+p2[0])/2.,(p1[1]+p2[1])/2.]
                 #plt.plot(tolin[0],tolin[1],'o',color='b')
                 if bool_normal(normal_cell,tolin,xp,yp,barycentre): normal_cell = -normal_cell

                 if solidwall[npoint-1]==1 and solidwall[npoint]==1:
                   #_vsx.append(d/config.dt*normal_cell[0])
                   #_vsy.append(d/config.dt*normal_cell[1])
                   _vsx.append(d*normal_cell[0])
                   _vsy.append(d*normal_cell[1])
                 else:
                   if not normal_cell[0]==0.:
                     solid_FSI.linearizeFlux(0,tolin,xu,yu,xv,yv,xp,yp,[yp.get_i(IJ,center[0],Lx,xp),yp.get_j(IJ,center[1],Ly)],pline,cline)
                     for fij in range(2):
                       indi,wli = pline[fij],cline[fij]
                       #coef_to_add = round(d/dx*wli*normal_cell[0],NR)
                       coef_to_add = d/dx*wli*normal_cell[0]
                       coef_to_add_p = d/dx*wli*normal_cell[0]
                       if not coef_to_add==0.:
                         if flagsu[indi[0]-(Or[0]-1),indi[1]-(Or[1])]==-2:
                           iw = CF.find_first(CF.IJ_ij_yg(indi[0],indi[1],xu,yu),pseudou)#indi[0]*yu.shape()+indi[1],pseudou)
                           pf0,pf1,p10,p11,p20,p21,coefs1,coefs2,coefs0 = fFu[iw]
                           add_coefs((xu.ind_perio(p10),\
                                      yu.ind_perio(p11)),coef_to_add_p*coefs1,is_coef_written, coefs, written_places,1,0,xp,yp)
                           add_coefs((xu.ind_perio(p20),\
                                      yu.ind_perio(p21)),coef_to_add_p*coefs2,is_coef_written, coefs, written_places,1,0,xp,yp)
                         else:
                           add_coefs((xu.ind_perio(indi[0]),\
                                      yu.ind_perio(indi[1])),coef_to_add_p,is_coef_written, coefs, written_places,1,0,xp,yp)
                         #append_(_u,_ui,_uj,dx/config.dt*coef_to_add,indi[0],indi[1])
                         append_(_u,_ui,_uj,dx*coef_to_add,indi[0],indi[1])

                   if not normal_cell[1]==0.:
                     solid_FSI.linearizeFlux(1,tolin,xv,yv,xu,yu,xp,yp,[yp.get_i(IJ,center[0],Lx,xp),yp.get_j(IJ,center[1],Ly)],pline,cline)
                     for fij in range(2):
                       indi,wli = pline[fij],cline[fij]
                       #coef_to_add = round(d/dx*wli*normal_cell[1],NR)
                       coef_to_add = d/dx*wli*normal_cell[1]
                       coef_to_add_p = d/dx*wli*normal_cell[1]
                       if not coef_to_add==0.:
                         if flagsv[indi[0]-(Or[0]),indi[1]-(Or[1]-1)]==-2:
                           iw = CF.find_first(CF.IJ_ij_yg(indi[0],indi[1],xv,yv),pseudov)#indi[0]*yv.shape()+indi[1],pseudov)
                           pf0,pf1,p10,p11,p20,p21,coefs1,coefs2,coefs0 = fFv[iw]
                           add_coefs((xv.ind_perio(p10),\
                                      yv.ind_perio(p11)),coef_to_add_p*coefs1,is_coef_written, coefs, written_places,0,1,xp,yp)
                           add_coefs((xv.ind_perio(p20),\
                                      yv.ind_perio(p21)),coef_to_add_p*coefs2,is_coef_written, coefs, written_places,0,1,xp,yp)
                         else:
                           add_coefs((xv.ind_perio(indi[0]),\
                                      yv.ind_perio(indi[1])),coef_to_add_p,is_coef_written, coefs, written_places,0,1,xp,yp)
                         #append_(_v,_vi,_vj,dx/config.dt*coef_to_add,indi[0],indi[1])
                         append_(_v,_vi,_vj,dx*coef_to_add,indi[0],indi[1])

            coefs_for_psisource[n] = _vsx, _vsy, _u, _ui, _uj, _v, _vi, _vj, IJ
            update_csr((yp.get_i_abs(IJ),yp.get_j_abs(IJ)), indptr, indices, data, coefs, written_places, xp,yp)

    return (-0.25*np.array(data, dtype=np.float),np.array(indices, dtype=np.int),indptr), coefs_for_psisource
    #return (-0.*np.array(data, dtype=np.float),np.array(indices, dtype=np.int),indptr), coefs_for_psisource

def append_(L,Li,Lj,c,ci,cj):
  L.append(c)
  Li.append(ci)
  Lj.append(cj)
  return

def bool_normal(normal_cell,tolin,xp,yp,barycentre):
  return normal_cell[0]*(tolin[0]-barycentre[0])+normal_cell[1]*(tolin[1]-barycentre[1])<0.

def reshape_psisourceTF(u,v,xp,yp,psisource,coefs_for_psisource,local_bornes,local_sphere_numbers):

    s_n_loc=0
    for s_n in local_sphere_numbers:
      center = config.get_centers(s_n)
      origine = local_bornes[s_n_loc][[0,2]]
      for info in coefs_for_psisource[s_n_loc]:
        if not info == None:
          psisource[s_n_loc][yp.get_j(info[-1],center[1],Ly)-origine[1], yp.get_i(info[-1],center[0],Lx,xp)-origine[0]] = 0.
          psisource[s_n_loc][yp.get_j(info[-1],center[1],Ly)-origine[1], yp.get_i(info[-1],center[0],Lx,xp)-origine[0]] =\
                             compute_massBalanceTF(u[s_n_loc],v[s_n_loc],info[:-1],s_n,origine)
      s_n_loc+=1
    return

def compute_massBalanceTF(u,v,args,s_n,Or):

    vsx, vsy, cu, ui, uj, cv, vi, vj = args
    tempo = 0.
    vsolid = config.get_solidVeloPosit(s_n)[1]
    for c in vsx: tempo += c*vsolid[0]
    for c in vsy: tempo += c*vsolid[1]
    #for c,i,j in zip(cu,ui,uj): tempo += c*u[j-Or[1]+1,i-Or[0]]
    #for c,i,j in zip(cv,vi,vj): tempo += c*v[j-Or[1],i-Or[0]+1]
    for c,i,j in zip(cu,ui,uj): tempo += c*u[j-Or[1],i-Or[0]+1]
    for c,i,j in zip(cv,vi,vj): tempo += c*v[j-Or[1]+1,i-Or[0]]
    tempo /= config.dt
    return tempo

def update_csr(lineIJ, indptr, indices, data, coefs, written_places, xp,yp):
            line = lineIJ[1]*(xp.shape())+lineIJ[0]
            indptr[line+1:] += len(written_places)
            indices[indptr[line]:indptr[line]] = written_places
            data[indptr[line]:indptr[line]] = coefs[written_places]
            return  

def add_coefs(indices,coef_to_add,is_coef_written, coefs, written_places,cu,cv,xp,yp):
 #nega = CF.IJ_ij_xg(indices[0]   ,indices[1]   ,xp,yp)
 #posi = CF.IJ_ij_xg(indices[0]+cu,indices[1]+cv,xp,yp)
 nega = indices[1]*(nx+1)+indices[0]
 posi = (indices[1]+cv)*(nx+1)+(indices[0]+cu)
 coefs[nega] += -coef_to_add
 coefs[posi] += +coef_to_add
 keep_track(is_coef_written, written_places, nega)
 keep_track(is_coef_written, written_places, posi) 
 return

def keep_track(maptrack, written_places, place):
 if not maptrack[place]:
   maptrack[place] = True
   written_places.append(place)
 return

def reshape_cells(xp,yp,s_n,Or):

    center = config.get_centers(s_n)

    flagsp = config.get_flags_slice(2,s_n)
    flagsu = config.get_flags_slice(0,s_n)
    flagsv = config.get_flags_slice(1,s_n)

    forcedp, projectionsFp = config.get_forcing(2,s_n)
    pseudop, projectionsPp = config.get_pseudo(2,s_n)

    flags = np.zeros(4,dtype=np.int8)
    recutcells = [None]*len(forcedp)
    solidwall = [None]*len(forcedp) 
    for n in range(len(forcedp)):        
        IJ = forcedp[n] 
        prjx,prjy = projectionsFp[n][0],projectionsFp[n][1]
        normal = projectionsFp[n][2],projectionsFp[n][3]
        xpxijp, ypyijp = xp.loc(yp.get_i(IJ,center[0],Lx,xp)),yp.loc(yp.get_j(IJ,center[1],Ly))
        coins = [[xpxijp-dx/2., ypyijp-dx/2.],\
                 [xpxijp+dx/2., ypyijp-dx/2.],\
                 [xpxijp+dx/2., ypyijp+dx/2.],\
                 [xpxijp-dx/2., ypyijp+dx/2.]]
        flags[:] = 0
        for c in range(4):
            #if solid_FSI.isinsolid(coins[c],normal,s_n): flags[c] = 1          
            if solid_FSI.isinnodule(s_n,coins[c][0],coins[c][1]): flags[c] = 1
        if np.sum(flags) > 0 : # cut-cell             
            temprc,tempsolid = [],[]
            for c in range(4):
                if flags[c] == 1:
                    coinsinfluid = []
                    if flags[c-1] == 0:
                        coinsinfluid.append(c-1)
                    if c < 3:
                        if flags[c+1] == 0:
                            coinsinfluid.append(c+1)
                    else:
                        if flags[0] == 0:
                            coinsinfluid.append(0)
                    for cif in coinsinfluid:
                        cross = intersec(coins[c],coins[cif],[prjx,prjy],normal)
                        temprc.append([cross[0],cross[1]])
                        tempsolid.append(1)
                else:
                    temprc.append(coins[c]) 
                    tempsolid.append(0)
            recutcells[n] = temprc
            solidwall[n] = tempsolid

    #oldcutcells = config.get_cutcellspartner(s_n)
    #cutcellspartner = np.zeros((len(pseudop),2), dtype='int64')
    for n in range(len(pseudop)):
        IJ = pseudop[n] 
        prjx,prjy = projectionsPp[n][0],projectionsPp[n][1]
        normal = projectionsPp[n][2],projectionsPp[n][3]
        _xijp, _yijp = yp.get_i(IJ,center[0],Lx,xp),yp.get_j(IJ,center[1],Ly)
        xpxijp, ypyijp = xp.loc(yp.get_i(IJ,center[0],Lx,xp)),yp.loc(yp.get_j(IJ,center[1],Ly))
        coins = [[xpxijp-dx/2., ypyijp-dx/2.],\
                 [xpxijp+dx/2., ypyijp-dx/2.],\
                 [xpxijp+dx/2., ypyijp+dx/2.],\
                 [xpxijp-dx/2., ypyijp+dx/2.]]
        flags[:] = 1
        for c in range(4):
          #if not solid_FSI.isinsolid(coins[c],normal,s_n): flags[c] = 0
          if not solid_FSI.isinnodule(s_n,coins[c][0],coins[c][1]): flags[c] = 0
        if np.sum(flags) < 4 : # cut-cell
            maxnormal=0.
            if flagsp[_xijp+1-Or[0],_yijp-Or[1]] == 1:
              if normal[0] > maxnormal:# and (not recutcells[forcedp.index((_xijp+1)*(ny+1) + _yijp)] == None):
                maxnormal = normal[0]
                ijwin = CF.IJ_ij_yg(_xijp+1,_yijp,xp,yp)#(_xijp+1)*(ny+1) + _yijp
                forbid = [1,2]              
            if flagsp[_xijp-1-Or[0],_yijp-Or[1]] == 1: 
              if -normal[0] > maxnormal:# and (not recutcells[forcedp.index((_xijp-1)*(ny+1) + _yijp)] == None):
                maxnormal = -normal[0]
                ijwin = CF.IJ_ij_yg(_xijp-1,_yijp,xp,yp)#(_xijp-1)*(ny+1) + _yijp
                forbid = [0,3,-1]
            if flagsp[_xijp-Or[0],_yijp+1-Or[1]] == 1: 
              if normal[1] > maxnormal:# and (not recutcells[forcedp.index((_xijp)*(ny+1) + _yijp+1)] == None):
                maxnormal = normal[1]
                ijwin = CF.IJ_ij_yg(_xijp,_yijp+1,xp,yp)#_xijp*(ny+1) + (_yijp+1)
                forbid = [2,3,-1]
            if flagsp[_xijp-Or[0],_yijp-1-Or[1]] == 1:
              if -normal[1] > maxnormal:# and (not recutcells[forcedp.index((_xijp)*(ny+1) + _yijp-1)] == None):
                maxnormal = -normal[1]
                ijwin = CF.IJ_ij_yg(_xijp,_yijp-1,xp,yp)#_xijp*(ny+1) + (_yijp-1)
                forbid = [0,1]

     #       cutcellspartner[n] = IJ,ijwin

            try: partner = forcedp.index(ijwin)
            except: print 'partner not found',IJ,_xijp,_yijp
            partneriscut = (not recutcells[partner] == None)
            normal_intersec = normal
            if not partneriscut:
               #prjx,prjy = projectionsPp[n][0],projectionsPp[n][1]
               xp_xijp, yp_yijp = xp.loc(yp.get_i(ijwin,center[0],Lx,xp)),yp.loc(yp.get_j(ijwin,center[1],Ly))
               recutcells[partner] = [[xp_xijp-dx/2., yp_yijp-dx/2.],\
                                      [xp_xijp+dx/2., yp_yijp-dx/2.],\
                                      [xp_xijp+dx/2., yp_yijp+dx/2.],\
                                      [xp_xijp-dx/2., yp_yijp+dx/2.]]
               solidwall[partner] = [0,0,0,0]
               forbid = []
            #else:
            #   prjx,prjy = projectionsFp[partner][0],projectionsFp[partner][1]
            #   normal_intersec = projectionsFp[partner][2],projectionsFp[partner][3]
            prjx,prjy = projectionsPp[n][0],projectionsPp[n][1]
            for c in range(4):
                if flags[c] == 0:
                    coinsinsolid = []
                    if flags[c-1] == 1:
                        coinsinsolid.append(c-1)
                    if c < 3:
                        if flags[c+1] == 1:
                            coinsinsolid.append(c+1)
                    else:
                        if flags[0] == 1:
                            coinsinsolid.append(0)
                    for cis in coinsinsolid:
                        if not (c in forbid and cis in forbid):
                            cross = intersec(coins[c],coins[cis],[prjx,prjy],normal_intersec)
                            recutcells[partner].append([cross[0],cross[1]])
                            solidwall[partner].append(1)
    #config.set_cutcellspartner(s_n,cutcellspartner)
    for n in range(len(forcedp)): 
       if not recutcells[n] is None:#cut cell

         cell = np.array(recutcells[n])
         xanc,yanc = [np.sum(cell[:,0])/cell.shape[0],np.sum(cell[:,1])/cell.shape[0]]
 
         nsol = 0
         sw = []
         for sol in solidwall[n]:
           if sol == 1: sw.append(nsol)
           nsol+=1
         if len(sw) > 2:
           swatt = []
           maximum = 0.
           nfirst = 0
           for SW1 in sw[:-1]:
             for SW2 in sw[nfirst:]:
               measure = CF.na_vec_prod(xanc,yanc,recutcells[n][SW1][0],recutcells[n][SW1][1],recutcells[n][SW2][0],recutcells[n][SW2][1])
               if measure > maximum:
                 maximum = measure
                 swatt = [SW1,SW2]
             nfirst+=1
           recutcells[n] = [recutcells[n][i] for i in range(cell.shape[0]) if (solidwall[n][i] == 0 or i in swatt)]
           solidwall[n] = [solidwall[n][i]   for i in range(cell.shape[0]) if (solidwall[n][i] == 0 or i in swatt)]

         temp = []
         for point in recutcells[n]:
             Xo = point[0]-xanc
             Yo = point[1]-yanc
             temp.append(math.atan2(Yo, Xo))
         #temprc,tempsw,snort,indices_toswap,indices_todelete = [],[],0,[],[]
         temp = np.argsort(temp)
         recutcells[n] = [recutcells[n][i] for i in temp]        
         solidwall[n] = [solidwall[n][i] for i in temp]

    return recutcells,solidwall
####functions definition end
#parameters
Ly,Lx,ny,nx,dx,dy,nu,rho,Ulid = config.sharedparameters()
R = config.get_sphereParameters()[0]
