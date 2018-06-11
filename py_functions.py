from math import sqrt
from math import fabs
from math import atan2
import numpy as np
import config

class Axe:
    def __init__(self, nshape, doffset, dstep):
        self.size = nshape
        self.offset = doffset
        self.step = dstep

    def loc(self, i): return i*self.step - self.offset
    def loc_rel(self, i,rel,L):
      res = self.loc(i)
      while fabs(res-L-rel)<fabs(res-rel): res-=L
      while fabs(res+L-rel)<fabs(res-rel): res+=L
      return res
    def get_rel_val(self,rel,L,res):
        if self.offset==0.:
          while fabs(self.loc(res-(self.size-1))-rel)<fabs(self.loc(res)-rel): res-=(self.size-1)
          while fabs(self.loc(res+(self.size-1))-rel)<fabs(self.loc(res)-rel): res+=(self.size-1)
        else:
          while fabs(self.loc(res-(self.size-2))-rel)<fabs(self.loc(res)-rel): res-=(self.size-2)
          while fabs(self.loc(res+(self.size-2))-rel)<fabs(self.loc(res)-rel): res+=(self.size-2)
        return res

    def get_i(self, IJ, rel,L,xg): return xg.get_rel_val(rel,L,IJ // self.size)
    def get_j(self, IJ, rel,L): return self.get_rel_val(rel,L,IJ % self.size)
    def get_i_abs(self, IJ): return IJ // self.size
    def get_j_abs(self, IJ): return IJ %  self.size

    def shape(self): return self.size

    def fluids(self):
      if self.offset==0.: return 0,self.size-1
      else:               return 1,self.size-1

    def ind_perio(self,i):
      if self.offset==0.:
        while i<0: i+=self.size-1
        while i>=self.size-1: i-=self.size-1
      else:
        while i<1: i+=(self.size-2)
        while i>=(self.size-1): i-=(self.size-2)
      return i

    def ind_perio_buffer(self,i):
      return self.ind_perio(i)

def loc_perio(x,L):
      while x>=L: x-=L
      while x<0: x+=L
      return x

def IJ_ij_xg(itest,jtest,xg,yg):
  itest = xg.ind_perio(itest)
  jtest = yg.ind_perio(jtest)
  return jtest*xg.shape()+itest

def IJ_ij_yg(itest,jtest,xg,yg):
  itest = xg.ind_perio(itest)
  jtest = yg.ind_perio(jtest)
  return itest*yg.shape()+jtest

def no_g(d1, d2): return sqrt(d1**2+d2**2)

def search_forced_and_pseudo(potential_pseudo, flags_slice, pseudoPoints, forcedPoints, xg, yg, neighbours4, Or0, Or1,center,Lx,Ly):
    for IJ in potential_pseudo:
      i,j = yg.get_i(IJ,center[0],Lx,xg),yg.get_j(IJ,center[1],Ly)
      for neighbour in neighbours4:
             # maybe forcing on the ghost points
        itest,jtest = i+neighbour[0], j+neighbour[1]
        test = IJ_ij_yg(itest,jtest,xg,yg)

        if flags_slice[i-Or0,j-Or1] == -1 and flags_slice[itest-Or0,jtest-Or1] >= 0:
                            # pseudo point
          #if   xg.offset==0.:
          #  if yg.get_i_abs(IJ) == 0 :            pseudoPoints.append(IJ + (xg.shape()-1)*yg.shape())
          #  if yg.get_i_abs(IJ) == xg.shape()-1 : pseudoPoints.append(IJ - (xg.shape()-1)*yg.shape())
          #elif yg.offset==0.:
          #  if yg.get_j_abs(IJ) == 0 :            pseudoPoints.append(IJ + (yg.shape()-1))
          #  if yg.get_j_abs(IJ) == yg.shape()-1 : pseudoPoints.append(IJ - (yg.shape()-1))
          pseudoPoints.append(IJ)
          flags_slice[i-Or0,j-Or1] = -2
        if flags_slice[i-Or0,j-Or1] == 0 and flags_slice[itest-Or0,jtest-Or1] < 0:
                            # forcing point
          #if   xg.offset==0.:
          #  if yg.get_i_abs(IJ) == 0 :            forcedPoints.append(IJ + (xg.shape()-1)*yg.shape())
          #  if yg.get_i_abs(IJ) == xg.shape()-1 : forcedPoints.append(IJ - (xg.shape()-1)*yg.shape())
          #elif yg.offset==0.:
          #  if yg.get_j_abs(IJ) == 0 :            forcedPoints.append(IJ + (yg.shape()-1))
          #  if yg.get_j_abs(IJ) == yg.shape()-1 : forcedPoints.append(IJ - (yg.shape()-1))
          forcedPoints.append(IJ)
          flags_slice[i-Or0,j-Or1] = 1
        if flags_slice[itest-Or0,jtest-Or1] == -1 and flags_slice[i-Or0,j-Or1] >= 0:
                            # pseudo point
          #if   xg.offset==0.:
          #  if yg.get_i_abs(test) == 0 :            pseudoPoints.append(test + (xg.shape()-1)*yg.shape())
          #  if yg.get_i_abs(test) == xg.shape()-1 : pseudoPoints.append(test - (xg.shape()-1)*yg.shape())
          #elif yg.offset==0.:
          #  if yg.get_j_abs(test) == 0 :            pseudoPoints.append(test + (yg.shape()-1))
          #  if yg.get_j_abs(test) == yg.shape()-1 : pseudoPoints.append(test - (yg.shape()-1))
          pseudoPoints.append(test)
          flags_slice[itest-Or0,jtest-Or1] = -2
        if flags_slice[itest-Or0,jtest-Or1] == 0 and flags_slice[i-Or0,j-Or1] < 0:
                            # forcing point
          #if   xg.offset==0.:
          #  if yg.get_i_abs(test) == 0 :            forcedPoints.append(test + (xg.shape()-1)*yg.shape())
          #  if yg.get_i_abs(test) == xg.shape()-1 : forcedPoints.append(test - (xg.shape()-1)*yg.shape())
          #elif yg.offset==0.:
          #  if yg.get_j_abs(test) == 0 :            forcedPoints.append(test + (yg.shape()-1))
          #  if yg.get_j_abs(test) == yg.shape()-1 : forcedPoints.append(test - (yg.shape()-1))
          forcedPoints.append(test)
          flags_slice[itest-Or0,jtest-Or1] = 1
    return 0

def scalr_prod(normal,rx,ry,x,y):
  return (x-rx)*normal[0]+(y-ry)*normal[1]

def test_first_P(I,J,Or0,Or1,flags_slice):
 #if not flags_slice_P is None:
 # if flags_slice.shape[0]>flags_slice.shape[1]: return flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0-1,J-Or1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #U
 # if flags_slice.shape[0]<flags_slice.shape[1]: return flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0,J-Or1-1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #V
 return flags_slice[I-Or0,J-Or1] >= 0

def test_second_P(I,J,I1,J1,Or0,Or1,flags_slice):
 #if not flags_slice_P is None:
 # if flags_slice.shape[0]>flags_slice.shape[1]: return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0-1,J-Or1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #U
 # if flags_slice.shape[0]<flags_slice.shape[1]: return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0,J-Or1-1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #V
 return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1] >= 0

def test_first_UV(I,J,Or0,Or1,flags_slice):
 #if not flags_slice_P is None:
 # if flags_slice.shape[0]>flags_slice.shape[1]: return flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0-1,J-Or1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #U
 # if flags_slice.shape[0]<flags_slice.shape[1]: return flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0,J-Or1-1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #V
 return flags_slice[I-Or0,J-Or1] >= 0

def test_second_UV(I,J,I1,J1,Or0,Or1,flags_slice):
 #if not flags_slice_P is None:
 # if flags_slice.shape[0]>flags_slice.shape[1]: return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0-1,J-Or1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #U
 # if flags_slice.shape[0]<flags_slice.shape[1]: return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1]==0# and (flags_slice_P[I-Or0,J-Or1-1]>=0 and flags_slice_P[I-Or0,J-Or1]>=0) #V
 return ((not I==I1) or (not J==J1)) and flags_slice[I-Or0,J-Or1] >= 0

def incr_list_pressure(root0, root1, pf0, pf1, flags_slice, Or0, Or1, xg, yg, normal,layer,K):
        if layer:
          I1,J1,(I2,J2,formur) = incr_list_velocity(root0, root1, pf0, pf1, flags_slice, Or0, Or1, xg, yg, normal,layer,K)
          cs,c1,c2 = coefs_P(xg.loc(I1),xg.loc(J1),xg.loc(I2),xg.loc(J2),xg.loc(pf0),yg.loc(pf1),normal[0],normal[1])
        else:
          I1,J1 = incr_first(root0,root1,pf0, pf1, flags_slice, Or0, Or1, xg, yg, normal)
          (cs,c1,c2),I2,J2 = incr_second(root0, root1, pf0, pf1, flags_slice, Or0, Or1,xg, yg, normal,I1,J1)
        return cs,c1,c2,I1,J1,I2,J2

def incr_first(root0,root1,pf0, pf1, flags_slice, Or0, Or1, xg, yg, normal):
        N=1
        max_scal = -1e36
        for i in [-N,+N]:
            I,J = pf0+i,pf1
            if test_first_P(I,J,Or0,Or1,flags_slice): 
              xx = dist(root0,root1,xg.loc(I),yg.loc(J))
              if xx>1e-15:#1e-2*dx:
               scalr_ = scalr_prod(normal,root0,root1,xg.loc(I),yg.loc(J))/xx
               if scalr_>max_scal:
                max_scal=scalr_
                Ir,Jr = I,J
            I,J = pf0,pf1+i
            if test_first_P(I,J,Or0,Or1,flags_slice):
              xx = dist(root0,root1,xg.loc(I),yg.loc(J))
              if xx>1e-15:#1e-2*dx:
               scalr_ = scalr_prod(normal,root0,root1,xg.loc(I),yg.loc(J))/xx
               if scalr_>max_scal:
                max_scal=scalr_
                Ir,Jr = I,J
        return Ir,Jr

def incr_second(root0, root1, pf0, pf1, flags_slice, Or0, Or1,xg, yg, normal,I1,J1):
        N=1
      #while True:
        #mi=+1e36
        coefsr=None
        for i in [-N,+N]:
          for j in range(-(N-1),N):
            I,J = pf0+i,pf1+j
            if test_second_P(I,J,I1,J1,Or0,Or1,flags_slice):
              coefs = coefs_P(xg.loc(I1),yg.loc(J1),xg.loc(I),yg.loc(J),xg.loc(pf0),yg.loc(pf1),normal[0],normal[1])
              if not coefs is None:# and fabs(coefs[0])<mi:
                #mi = coefs[0]
                coefsr,Ir,Jr = coefs,I,J
            I,J = pf0+j,pf1+i
            if test_second_P(I,J,I1,J1,Or0,Or1,flags_slice):
              coefs = coefs_P(xg.loc(I1),yg.loc(J1),xg.loc(I),yg.loc(J),xg.loc(pf0),yg.loc(pf1),normal[0],normal[1])
              if not coefs is None:# and fabs(coefs[0])<mi:
                #mi = coefs[0]
                coefsr,Ir,Jr = coefs,I,J
        if not coefsr is None: return coefsr,Ir,Jr
        for i in [-N,+N]:
          for j in [-N,+N]:
            I,J = pf0+i,pf1+j
            if test_second_P(I,J,I1,J1,Or0,Or1,flags_slice):
              coefs = coefs_P(xg.loc(I1),yg.loc(J1),xg.loc(I),yg.loc(J),xg.loc(pf0),yg.loc(pf1),normal[0],normal[1])
              if not coefs is None:# and fabs(coefs[0])<mi:
                #mi = coefs[0]
                coefsr,Ir,Jr = coefs,I,J
        return coefsr,Ir,Jr

def incr_list_velocity(root0, root1, pf0, pf1, flags_slice, Or0, Or1, xg, yg, normal,layer,K):
      dx = config.dx
      N=1
      I1r = None
      while I1r is None:
        max_scal = -1e36
        for i in [-N,+N]:
          for j in range(-(N-1),N):
            I1,J1 = pf0+i,pf1+j
            if test_first_UV(I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))>=K*dx):
              xx = dist(root0,root1,xg.loc(I1),yg.loc(J1))
              if xx>1e-15:#1e-2*dx:
               scalr_ = scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))/xx
               if scalr_>max_scal:
                max_scal=scalr_
                I1r,J1r = I1,J1
            I1,J1 = pf0+j,pf1+i
            if test_first_UV(I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))>=K*dx):
              xx = dist(root0,root1,xg.loc(I1),yg.loc(J1))
              if xx>1e-15:#1e-2*dx:
               scalr_ = scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))/xx
               if scalr_>max_scal:
                max_scal=scalr_
                I1r,J1r = I1,J1
        if True:# I1r is None:
         for i in [-N,+N]:
          for j in [-N,+N]:
            I1,J1 = pf0+i,pf1+j
            if test_first_UV(I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))>=K*dx):
              xx = dist(root0,root1,xg.loc(I1),yg.loc(J1))
              if xx>1e-15:#1e-2*dx:
               scalr_ = scalr_prod(normal,root0,root1,xg.loc(I1),yg.loc(J1))/xx
               if scalr_>max_scal:
                max_scal=scalr_
                I1r,J1r = I1,J1
        N+=1
      res_second = loop_second(I1r,J1r,pf0,pf1,flags_slice,Or0,Or1,xg,yg,normal,root0, root1,layer,K)
      return I1r,J1r,res_second

def coefs_P(x1,y1, x2,y2, xg,yg, nx,ny):
  c = - (nx*(y2-y1)-ny*(x2-x1))
  if fabs(c)>0.:
    cs = (x1-xg)*(y2-yg)-(x2-xg)*(y1-yg)
    c1 = -nx*(y2-yg)+ny*(x2-xg)
    c2 = +nx*(y1-yg)-ny*(x1-xg)
    cs   /= c
    c1   /= c
    c2   /= c
    #print c1*c2
    if c1*c2 >= -1e-15: return cs,c1,c2
    #return cs,c1,c2
  return

def coefs_V(x1,y1, x2,y2, xg,yg, xs,ys, nx,ny):
  c = (x1-xs)*(y2-ys)-(x2-xs)*(y1-ys)
  gn1 = +nx*(y2-ys)-ny*(x2-xs)
  gn2 = -nx*(y1-ys)+ny*(x1-xs)
  if (fabs(c)>0. and gn1*gn2>=-1e-15):
    cs = (x2-x1)*(yg-ys)+(xg-xs)*(y1-y2)    
   #if True:#1.+cs < 100.:
    c1 = (xg-xs)*(y2-ys)-(x2-xs)*(yg-ys)
    c2 = (x1-xs)*(yg-ys)-(xg-xs)*(y1-ys)
    cs   /= c
    c1   /= c
    c2   /= c
    gns = +nx*(y1-y2)+ny*(x2-x1)
    gn1/= c
    gn2/= c
    gns/= c
    return 1.+cs,c1,c2,gns,gn1,gn2
  return

def loop_second(I1,J1,pf0,pf1,flags_slice, Or0, Or1, xg, yg, normal,root0, root1,layer,K):
     dx=config.dx
     froma = pf0,pf1
     N=1
     I2r = None
     while I2r is None:
       max_scal = -1e36
       for i in [-N,+N]:
          for j in range(-(N-1),N):
            I2,J2 = froma[0]+i,froma[1]+j
            if test_second_UV(I2,J2,I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))>=K*dx):
             formu = coefs_V(xg.loc(I1),yg.loc(J1),xg.loc(I2),yg.loc(J2),xg.loc(pf0),yg.loc(pf1),root0, root1,normal[0],normal[1])
             if not formu is None:# and formu[0] < mi:
              xx=dist(root0,root1,xg.loc(I2),yg.loc(J2))
              if xx>1e-15:#1e-2*dx:
               _scal = scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))/xx
               if _scal>max_scal:
                 max_scal=_scal
                 I2r,J2r,formur = I2,J2,formu
            I2,J2 = froma[0]+j,froma[1]+i
            if test_second_UV(I2,J2,I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))>=K*dx):
             formu = coefs_V(xg.loc(I1),yg.loc(J1),xg.loc(I2),yg.loc(J2),xg.loc(pf0),yg.loc(pf1),root0, root1,normal[0],normal[1])
             if not formu is None:# and formu[0] < mi:
              xx=dist(root0,root1,xg.loc(I2),yg.loc(J2))
              if xx>1e-15:#1e-2*dx:
               _scal = scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))/xx
               if _scal>max_scal:
                 max_scal=_scal
                 I2r,J2r,formur = I2,J2,formu
       if True:# I2r is None:
        for i in [-N,+N]:
          for j in [-N,+N]:
            I2,J2 = froma[0]+i,froma[1]+j
            if test_second_UV(I2,J2,I1,J1,Or0,Or1,flags_slice) and ((not layer) or scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))>=K*dx):
             formu = coefs_V(xg.loc(I1),yg.loc(J1),xg.loc(I2),yg.loc(J2),xg.loc(pf0),yg.loc(pf1),root0, root1,normal[0],normal[1])
             if not formu is None:# and formu[0] < mi:
              xx=dist(root0,root1,xg.loc(I2),yg.loc(J2))
              if xx>1e-15:#1e-2*dx:
               _scal = scalr_prod(normal,root0,root1,xg.loc(I2),yg.loc(J2))/xx
               if _scal>max_scal:
                 max_scal=_scal
                 I2r,J2r,formur = I2,J2,formu
       N+=1
     return I2r,J2r,formur       

def mu(normalx, normaly, p1x, p1y, p2x, p2y, ppx, ppy):
  if (normalx*(p2y-p1y)-normaly*(p2x-p1x)) == 0.: mu_ = 100. # lines are parallel
  else: mu_ = (normalx*(ppy-p1y)-normaly*(ppx-p1x))/(normalx*(p2y-p1y)-normaly*(p2x-p1x))
  return mu_

def set_field_dirichlet2(field, fF1, fF0, fF6, fF3, fF2, fF7, fF5, fF4, fFs, u_dirich, mpirank):
      field[fF1,fF0] = fF6*field[fF3,fF2]+fF7*field[fF5,fF4]+fFs*u_dirich

def resu(lambd, ps, pf):  
  result = ps + lambd*(pf-ps)
  return result

def lambd(normal0, normal1, pp0, pp1, ps0, ps1, pf0, pf1):
  res = (normal0*(pp0-ps0)+normal1*(pp1-ps1))/(normal0*(pf0-ps0)+normal1*(pf1-ps1))
  return res

def get_proj(center0, center1, tolin0, tolin1, R):
  ntl0 = tolin0-center0
  ntl1 = tolin1-center1
  normtl = no_g(ntl0,ntl1)
  ntl0 = ntl0/normtl
  ntl1 = ntl1/normtl
  xp,yp = center0+R*ntl0,center1+R*ntl1
  return [xp,yp,ntl0,ntl1]

def ecart(v1,v2):
  return v1-v2

def na_vec_prod(xr, yr, x1, y1, x2, y2):
  return fabs((x1-xr)*(y2-yr)-(y1-yr)*(x2-xr))

def alg_vec_prod(xr, yr, x1, y1, x2, y2):
  return (x1-xr)*(y2-yr)-(y1-yr)*(x2-xr)

def dist(xr, yr, x1, y1):
  return sqrt((x1-xr)**2+(y1-yr)**2)

def sq_dist(xr, yr, x1, y1):
  return (x1-xr)**2+(y1-yr)**2
  #return x1*x1-2.*x1*xr+xr*xr+y1*y1-2.*y1*yr+yr*yr

def find_first(item, vec):
  indice = 0
  for test in vec:
    if test == item: return indice
    indice+=1
  print item, '!!!!!!!!!!!!!!!!!not found!!!!!!!!!!!!!!!!!!!!!!!'
  return

def curvilinear(projections, curvCoords, pi):
    n = 0
    for proj in projections:
        angle = atan2(proj[3], proj[2])
        if angle < 0 :
            angle = 2*pi + angle
        curvCoords[n] = angle
        n+=1
    return 0

def totalDerivative(projections, dusdt, dvsdt, DuDt):
    n = 0
    for proj in projections:
        DuDt[n] = dusdt*proj[2] + dvsdt*proj[3]
        n+=1
    return 0

def interpolate_(surfaceStressField, curvilinear, angle, pi):    

    argmin = np.argmin(curvilinear)
    argmax = np.argmax(curvilinear)
    curvmin = curvilinear[argmin]
    curvmax = curvilinear[argmax]

    if angle < curvmin or angle > curvmax:

        if angle > pi: # just above the biggest
            dtot = 2*pi + curvmin - curvmax  
            dloc = angle - curvmax
        else: # just under the smallest
            dtot = curvmin - (curvmax - 2*pi)  
            dloc = angle - (curvmax - 2*pi)
        #return np.array([(surfaceStressField[argmin,0]-surfaceStressField[argmax,0])/dtot*dloc + surfaceStressField[argmax,0], \
        #       (surfaceStressField[argmin,1]-surfaceStressField[argmax,1])/dtot*dloc + surfaceStressField[argmax,1]])
        return (surfaceStressField[argmin]-surfaceStressField[argmax])/dtot*dloc + surfaceStressField[argmax]
    else:
        ninf = argmin
        nsup = argmax
        nmax = len(curvilinear) - 1
        n = 0
        while n <= nmax:
            if angle >= curvilinear[n] and curvilinear[n] > curvilinear[ninf]: ninf = n
            if angle <= curvilinear[n] and curvilinear[n] < curvilinear[nsup] : nsup = n
            n += 1
    
        if nsup == ninf: return surfaceStressField[nsup]
        else:
            dtot = curvilinear[nsup] - curvilinear[ninf]  
            dloc = angle - curvilinear[ninf]
            #return np.array([(surfaceStressField[nsup,0]-surfaceStressField[ninf,0])/dtot*dloc + surfaceStressField[ninf,0], \
            #       (surfaceStressField[nsup,1]-surfaceStressField[ninf,1])/dtot*dloc + surfaceStressField[ninf,1]])
            return (surfaceStressField[nsup]-surfaceStressField[ninf])/dtot*dloc + surfaceStressField[ninf]

def integrate_stress(n0, n1, surfaceStressFieldu, curvilinearU, surfaceStressFieldv, curvilinearV, surfaceStressFieldp, curvilinearP, computationPoints, pi, nu, rho):
    fvx, fvy, fpx, fpy = 0.,0.,0.,0.
    i = 0
    for angle in computationPoints:

           dudn  = interpolate_(surfaceStressFieldu,curvilinearU,angle,pi)
           dvdn  = interpolate_(surfaceStressFieldv,curvilinearV,angle,pi)
           press = interpolate_(surfaceStressFieldp,curvilinearP,angle,pi)
        
           #fvx += rho*nu*(dudn*n1[i] - dvdn*n0[i])*(+n1[i])
           #fvy += rho*nu*(dudn*n1[i] - dvdn*n0[i])*(-n0[i])
           fvx += rho*nu*(dudn*n1[i] - dvdn*n0[i])*(+n1[i])
           fvy += rho*nu*(dudn*n1[i] - dvdn*n0[i])*(-n0[i])
           
           fpx += ( -press*n0[i] )
           fpy += ( -press*n1[i] )
           i+=1
    return np.array([fvx, fvy, fpx, fpy])
