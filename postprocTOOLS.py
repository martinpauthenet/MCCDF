import matplotlib.pyplot as plt
import matplotlib.colors as colors
import config
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, linalg as sla, identity
#parameters
Ly,Lx,ny,nx,dy,dx,nu,rho,Ulid = config.sharedparameters()
R,ms,K,gamma,centers0 = config.get_sphereParameters()
_Nit = config.get_nite()

def plot_cutcell(cell,center,sn,toplot):
  #center = config.get_centers(sn)
  size = 6
  fig, ax = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
  #circle=plt.Circle((center[0],center[1]),R,color='k',fill=False)
  #fig.gca().add_artist(circle)
  #ax.add_artist(circle)

  config.plotSphere(ax,plt,False,config.get_centers(sn))

  for i in range(len(cell)):
      ax.plot([cell[i][0],cell[i-1][0]],[cell[i][1],cell[i-1][1]],'+',color='red')
  ax.plot([center[0]],[center[1]],'o',color='g')

  for i in range(1,3):
    ax.plot([center[0]],[center[1]+i*dx/2.],'o',color='k')
    ax.plot([center[0]],[center[1]-i*dx/2.],'o',color='k')
    ax.plot([center[0]+i*dx/2.],[center[1]],'o',color='k')
    ax.plot([center[0]-i*dx/2.],[center[1]],'o',color='k')

  ax.plot([toplot[0]],[toplot[1]],'o',color='b')

#  ax.set_xlim([center[0]-2*dx,center[0]+2*dx])
#  ax.set_ylim([center[1]-2*dx,center[1]+2*dx])

  plt.show()
  return

def plot_stencils(p1,p2,pf,proj,s_n,xg,yg):#,xp,yp):
  center = config.get_centers(s_n)
  #print center
  #circle=plt.Circle((center[0],center[1]),R,color='k',fill=False)
  size = 6
  fig, ax = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
  #fig.gca().add_artist(circle)
  #ax.add_artist(circle)
  config.plotSphere(ax,plt,False,center)
  #for i in range(5):
  #    ax.plot([xp.loc(pf[0])+i*dx+dx/2.,xp.loc(pf[0])+i*dx+dx/2.],[yp.loc(pf[1])-20*dx,yp.loc(pf[1])+20*dx],color='grey')
  #    ax.plot([xp.loc(pf[0])-i*dx+dx/2.,xp.loc(pf[0])-i*dx+dx/2.],[yp.loc(pf[1])-20*dx,yp.loc(pf[1])+20*dx],color='grey')
  #    ax.plot([xp.loc(pf[0])-20*dx,xp.loc(pf[0])+20*dx],[yp.loc(pf[1])-i*dx+dx/2.,yp.loc(pf[1])-i*dx+dx/2.],color='grey')
  #    ax.plot([xp.loc(pf[0])-20*dx,xp.loc(pf[0])+20*dx],[yp.loc(pf[1])+i*dx+dx/2.,yp.loc(pf[1])+i*dx+dx/2.],color='grey')

  for p in [p1,p2]:
    ax.plot([xg.loc(p[0])],[yg.loc(p[1])],'+',color='k')

  ax.plot([proj[0]],[proj[1]],'x',color='r')
  cadre = 2
  ax.set_xlim([proj[0]-cadre*dx,proj[0]+cadre*dx])
  ax.set_ylim([proj[1]-cadre*dx,proj[1]+cadre*dx])  

  ax.plot([xg.loc(pf[0])],[yg.loc(pf[1])],'+',color='g')

  plt.show()
  return

def seeMatrix(mat):
    #mindata=np.amin(np.abs(mat.data))
    #print mindata
    #mat.data*=np.abs(mat.data)>0.1*nu*dt/dx**2
    #mindata=np.amin(np.abs(mat.data))
    #print mindata

    size = 6
    fig, ax = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
    #mat.data = np.array(np.abs(mat.data)==1.,dtype=np.int)+(1-np.array(np.abs(mat.data)==1.,dtype=np.int))*1.
    nubl = np.amax(mat.indptr[1:]-mat.indptr[:-1])
    print nubl
    nubl = np.amin(mat.indptr[1:]-mat.indptr[:-1])
    print set(mat.indptr[1:]-mat.indptr[:-1])
    im = plt.imshow(mat.todense(), interpolation='nearest', origin='lower',cmap=plt.cm.autumn) 
    CBI = plt.colorbar(im, orientation='vertical', shrink=0.6)
    plt.gca().invert_yaxis()

def plotIlus(u,v,p,nsp):
         uu = np.zeros((ny+1,nx+1))
         vv = np.zeros((ny+1,nx+1))
         uu[1:-1,1:-1] = (u[1:-1,:-1]+u[1:-1,1:])/2
         vv[1:-1,1:-1] = (v[:-1,1:-1]+v[1:,1:-1])/2
         V =np.sqrt(uu**2 + vv**2)
         #uu = uu/V
         #vv = vv/V
         #uu[nsp[1],nsp[0]]=0.
         #vv[nsp[1],nsp[0]]=0.
         V[nsp[1],nsp[0]]=0.
         

         if False: 
          forcedp, projectionsp = config.get_forcing(2)
          vs = config.get_solidVeloPosit()[1]
          fFu = solid_FSI.linearizeVelocity_visu(uu,xp,yp,vs[0,0])
          fFv = solid_FSI.linearizeVelocity_visu(vv,xp,yp,vs[0,1])

          for n in range(len(forcedp)):
           IJ = forcedp[n]
           uu[yp.get_j(IJ),yp.get_i(IJ)] = fFu[n]
           vv[yp.get_j(IJ),yp.get_i(IJ)] = fFv[n]
          V =np.sqrt(uu**2 + vv**2)

         uu = uu[1:-1,1:-1]
         vv = vv[1:-1,1:-1]
         V = V[1:-1,1:-1]
         #V[:,nx/2:]=0.
         dpi = 100
         fig, ax = plt.subplots(dpi=dpi) 
         im1 = ax.imshow(V, interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         config.plotSpheres(ax,plt,True)

def plotFields(u,v,p,nsp,centers):#,p1,p2):
         
    #   uu = np.zeros((ny+1,nx+1))
    #   vv = np.zeros((ny+1,nx+1))
       uu = (u[1:-1,:-1]+u[1:-1,1:])/2
       vv = (v[:-1,1:-1]+v[1:,1:-1])/2
       V =np.sqrt(uu**2 + vv**2)
         #uu = uu/V
         #vv = vv/V
       #uu[nsp[1]-1,nsp[0]-1]=0.
       #vv[nsp[1]-1,nsp[0]-1]=0.
    #   V[nsp[1],nsp[0]]=10.

       #uu = uu[1:-1,1:-1]
       #vv = vv[1:-1,1:-1]
       #V = V[1:-1,1:-1]
       
       dpi = 100
       #fig, ax = plt.subplots(dpi=dpi)
       #uu[0.7*vv.shape[1]:,:] = 0.
       #uu[:0.3*vv.shape[1],:] = 0.
       #V0 = config.Reynolds*config.nu/(2.*config.R)
       #ax.plot((V0-uu[0.5*uu.shape[0],:])/V0)
       #for i in range(1): ax.plot((V0-vv[:,0.5*vv.shape[1]+i])/V0)

       #dpi = 100
       #figsl, axsl = plt.subplots(dpi=dpi)
       #x,y = np.linspace(0.,Lx-dx,nx-1)+dx/2.,np.linspace(0.,Ly-dx,ny-1)+dx/2.
       #axsl.streamplot(x, y, uu, vv, density=1)#, density=1, INTEGRATOR='RK4', color='b')

       if True:
         #V[:,nx/2:]=0.
         dpi = 100
         fig, (ax1,ax2,ax3) = plt.subplots(1,3, dpi=dpi)

     #    uu[0.7*vv.shape[0]:,:] = 0.
     #    uu[:0.3*vv.shape[0],:] = 0.
         im1 = ax1.imshow(uu, origin='lower',interpolation='nearest',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         im3 = ax3.imshow(vv, origin='lower',interpolation='nearest',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         im2 = ax2.imshow(p[1:-1,1:-1], origin='lower',interpolation='nearest',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))

         if False:
           step = 2
           Xg, Yg = np.meshgrid(np.linspace(0.,Lx-dx,nx-1)+dx/2.,np.linspace(0.,Ly-dx,ny-1)+dx/2.)
           plt.quiver(Xg[::step,::step],Yg[::step,::step],uu[::step,::step],vv[::step,::step],headaxislength=0.) ##plotting velocity
         if True:
           for center in centers:
             config.plotSphere(ax1,plt,False,center)
             config.plotSphere(ax2,plt,False,center)
             config.plotSphere(ax3,plt,False,center)
           #config.plotSpheres(ax1,plt,False,centers)
           #config.plotSpheres(ax2,plt,False,centers)

         fig.colorbar(im1,ax=ax1)
         fig.colorbar(im2,ax=ax2)
         fig.colorbar(im3,ax=ax3)
         return ax2

def save_shot(image,u,v,p,nsp,step,centers):
         print 'shot image ' +str(image)+ ' ...'
         uu = np.zeros((ny+1,nx+1))
         vv = np.zeros((ny+1,nx+1))
         uu[1:-1,1:-1] = (u[1:-1,:-1]+u[1:-1,1:])/2
         vv[1:-1,1:-1] = (v[:-1,1:-1]+v[1:,1:-1])/2
         V =np.sqrt(uu**2 + vv**2)
         V[nsp[1],nsp[0]]=0.

         uu = uu[1:-1,1:-1]
         vv = vv[1:-1,1:-1]
         V = V[1:-1,1:-1]
         #V[:,nx/2:]=0.
         dpi = 100
         fig, (ax1,ax2) = plt.subplots(1,2, dpi=dpi) 
         im1 = ax1.imshow(V, interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))

         im2 = ax2.imshow(p[1:-1,1:-1], interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))         
         Xg, Yg = np.meshgrid(np.linspace(0.,Lx-dx,nx-1)+dx/2.,np.linspace(0.,Ly-dx,ny)+dx/2.)
         plt.quiver(Xg[::step,::step],Yg[::step,::step],uu[::step,::step],vv[::step,::step]) ##plotting velocity

         #fig.colorbar(im1,ax=ax1)
         #fig.colorbar(im2,ax=ax2)
         for center in centers:
           config.plotSphere(ax1,plt,True,center)
           config.plotSphere(ax2,plt,True,center)

         plt.savefig('im'+'%05d.png'%image, dpi = 170)
         plt.close(fig)
         print 'done.'
         return


import math
def plotConvective(uc,vc,u,v,nsp):

         uu = np.zeros((ny-1,nx-1))
         vv = np.zeros((ny-1,nx-1))
         uu[:,:] = (u[1:-1,:-1]+u[1:-1,1:])/2.
         vv[:,:] = (v[:-1,1:-1]+v[1:,1:-1])/2.
         uu[nsp[1]-1,nsp[0]-1]=0.
         vv[nsp[1]-1,nsp[0]-1]=0.
         V =uu**2 + vv**2

         uuc = np.zeros((ny-1,nx-1))
         vvc = np.zeros((ny-1,nx-1))
         uuc[:,:] = (uc[:,:-1]+uc[:,1:])/2.
         vvc[:,:] = (vc[:-1,:]+vc[1:,:])/2.
         uuc[nsp[1]-1,nsp[0]-1]=0.
         vvc[nsp[1]-1,nsp[0]-1]=0.
         Vc =uuc**2 + vvc**2

         print math.sqrt((np.sum(Vc)/((ny-1)*(nx-1)))*1.5e-9/(np.sum(V)/((ny-1)*(nx-1))))*(5e-4/1e-5/1.2233e-02)

         fill = True 
         dpi = 100
         #fig, ax1 = plt.subplots(dpi=dpi)
         #im1 = ax1.imshow(uu, interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         #config.plotSpheres(ax1,plt,fill)
         #fig.colorbar(im1)
         
         #fig, ax2 = plt.subplots(dpi=dpi)
         #im2 = ax2.imshow(vv, interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         #config.plotSpheres(ax2,plt,fill)
         #fig.colorbar(im2)

         fig, ax3 = plt.subplots(dpi=dpi)
         im3 = ax3.imshow(V/np.amax(V), interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly))
         config.plotSpheres(ax3,plt,fill)
         fig.colorbar(im3)

         return

def compare(hc,hnc,Nsphere):
  couleurs = colors.cnames.items()
  size = 6
  fig1, ax1 = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)

  vmvx,vmvy,vmpx,vmpy,YS,XS = hc
  #delt = 3.3e-8
  #print delt
  #vmvx-=delt
  #vmpx-=delt
  #vmvx*=1./2.39e-6
  #vmpx*=1./2.39e-6
  for sn in [3]:
        #if sn==0:
         couleur = couleurs[sn][0]
         x,y=config.get_centers(sn)
         #ax1.plot(vmvy[:,sn]+vmpy[:,sn],'-',label='FY '+str(x)+','+str(y),color='g',lw=2.)
         ax1.plot(vmvx[:,sn]+vmpx[:,sn],'-',label='FX '+str(x)+','+str(y),color='g',lw=2.)
  vmvx,vmvy,vmpx,vmpy,YS,XS = hnc
  #vmvx*=1./2.39e-6
  #vmpx*=1./2.39e-6
  for sn in [3]:
        #if sn==0:
         couleur = couleurs[sn][0]
         x,y=round(config.get_centers(sn)[0],2),round(config.get_centers(sn)[1],2)
         #ax1.plot(vmvy[:,sn]+vmpy[:,sn],'-',label='FY '+str(x)+','+str(y),color='k',lw=1.)
         ax1.plot(vmvx[:,sn]+vmpx[:,sn],'-',label='FY '+str(x)+','+str(y),color='k',lw=2.)

  #fig1, ax1 = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)

  #vmvx,vmvy,vmpx,vmpy,YS,XS = hc
  #print YS.shape
  #for sn in [3]:
  #       couleur = couleurs[sn][0]
  #       x,y=config.get_centers(sn)
         #ax1.plot(vmvy[:,sn]+vmpy[:,sn],'-',label='FY '+str(x)+','+str(y),color='g',lw=2.)
  #       ax1.plot(YS[:,sn],vmvx[:,sn]+vmpx[:,sn],'-',label='FX '+str(x)+','+str(y),color='g',lw=2.)
  #vmvx,vmvy,vmpx,vmpy,YS,XS = hnc
  #for sn in [3]:
  #       couleur = couleurs[sn][0]
  #       x,y=round(config.get_centers(sn)[0],2),round(config.get_centers(sn)[1],2)
         #ax1.plot(vmvy[:,sn]+vmpy[:,sn],'-',label='FY '+str(x)+','+str(y),color='k',lw=1.)
  #       ax1.plot(YS[:,sn],vmvx[:,sn]+vmpx[:,sn],'-',label='FY '+str(x)+','+str(y),color='k',lw=2.)
  
  return

def plot_details(frate,history,Nsphere,u,v,p,nsp,centers,time):
    if True:
      size = 6
      fig1, ax1 = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
      fig2, ax2 = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
      couleurs = colors.cnames.items()
      vmvx,vmvy,vmpx,vmpy,YS,XS = history
     
    if True:
       for sn in range(Nsphere):
         couleur = couleurs[sn][0]
         ax1.plot(time, vmpx[:,sn],'-',label='Fpx '+str(sn),color=couleur)
         #ax1.plot(time, vmpx[:,sn],'-',label='Fpx '+str(sn),color=couleur)
         #ax1.plot(time, vmpx[:,sn],'-',label='Fp '+str(sn),color=couleur)
         #ax1.plot(time, vmvx[:,sn],'-.',label='Fvx '+str(sn),color=couleur)
         #ax1.plot(time, vmpy[:,sn],'.-',label='Fp '+str(sn),color=couleur)
         #ax2.plot((YS[:,sn]-centers[sn][1])/(2*R),'-.',color=couleur)
         #ax2.plot(time, XS[:,sn],'-.',color=couleur)
         ax2.plot(time,XS[:,sn],'-',color=couleur)

    ax1.legend(loc=0)
    #ax2.set_ylim([0.,Ly])
    #ax2.set_xlim([0.,Lx])

    plotFields(u,v,p,nsp,centers)

    size = 6
    fig, ax = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110)
     #plt.plot(np.arange(200000)+200000,frate,'.',color='k')

     #frate = np.loadtxt('../deformable_ref/fields/save/2x2seq1/200000_flow_rate')
    plt.plot(time, frate/0.804,'-')
     #frate = np.loadtxt('../deformable_softer10/fields/save/2x2seq1/200000_flow_rate')
     #plt.plot(np.arange(200000)*dt*100*1e-5,frate,'-',label='fs*=0.1')
     #frate = np.loadtxt('../deformable_harder10/fields/save/2x2/200000_flow_rate')
     #plt.plot(np.arange(200000)*dt*100*1e-5,frate,'-',label='fs*=10')
     #plt.xlabel('t*')
     #plt.ylabel('Reynolds number')
     #plt.legend(loc=2)

    return

def show(): plt.show()

def plotXY(xf,yf):
    xplot = np.zeros(0)
    yplot = np.zeros(0)
    for i in xf:
        for j in yf:
            xplot = np.append(xplot,i)
            yplot = np.append(yplot,j)
    plt.plot(xplot,y,marker='.',linestyle='None')

def plotFlag(flags,Nx,Ny):
      V = np.zeros((ny+1,nx+1))
      size = 6
      fig, ax = plt.subplots(figsize=((size+3, (1.-0)/(1.-0)*(size+3))), dpi=110) 
      for i in range(Nx):
        for j in range(Ny):
          V[j,i] = flags[i*(Ny)+j]
      im = plt.imshow(V, interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly)) 
      CBI = plt.colorbar(im, orientation='vertical', shrink=0.6)
      return

def savepicture(i,fig):
    uu = (u[1:,:]+u[:-1,:])/2
    vv = (v[:,1:]+v[:,:-1])/2
    V =np.sqrt(uu**2 + vv**2)
    #im = plt.imshow(V, interpolation='nearest', origin='lower', extent=(0.0,Lx,0.0,Ly))
    im = plt.imshow(p[1:-1,1:-1], interpolation='nearest', origin='lower',cmap=plt.cm.autumn, extent=(0.0,Lx,0.0,Ly)) 
    #CBI = plt.colorbar(im, orientation='vertical', shrink=0.6)
    step = 1
    Xg, Yg = np.meshgrid(x,y)
    plt.quiver(Xg[::step,::step],Yg[::step,::step],uu[::step,::step],vv[::step,::step]) ##plotting velocity
    centers = config.get_centers()
    for center in centers:
        circle=plt.Circle((center[0],center[1]),R,color='k',fill=False)
        #fig.gca().add_artist(circle)    
    plt.xlabel('')
    plt.ylabel('')   
    plt.savefig('osc'+'%04d.png'%i, dpi = 150) 
    fig.clear()

def saveflowaxis(u):
        
        #uu = (u[1:-1,:-1]+u[1:-1,1:])/2.
        uu = (u[:-1,1:-1]+u[1:,1:-1])/2.
        
        V0 = config.Reynolds*config.nu/(2.*config.R)
        center = np.loadtxt(path_load+'centers')

        xaxis = Ly - (np.linspace(0.,Ly-dx,ny-1)+dx/2.)-(Ly-center[1])
        xaxis /= 2.*config.R
        uu = (V0-uu[:,0.5*uu.shape[1]])/V0
        print uu.shape,xaxis.shape
        plt.plot(xaxis,uu)
        #plt.show()
        np.savetxt('fields/save/uu',uu)
        np.savetxt('fields/save/xx',xaxis)

if __name__ == '__main__' :
 
  if True:
      import sys
      args = sys.argv[1:]
      if len(args)==1:
        path_load = config.get_nameSvg()+args[0]+'_'
        #path_load = config.get_nameLoad()+''
        #plt.plot(np.loadtxt(path_load+'viscX')+np.loadtxt(path_load+'presX'),color = 'k')
        #path_load = 'fields/save/quadratic/50_'
        #plt.plot(np.loadtxt(path_load+'viscX')+np.loadtxt(path_load+'presX'),color = 'r')
        #plt.show()

        u = np.loadtxt(path_load+'u')
        v = np.loadtxt(path_load+'v')
        p = np.loadtxt(path_load+'p')

        #saveflowaxis(v)
        frate = np.loadtxt(path_load+'flow_rate')
        time_ = np.loadtxt(path_load+'time')
        history = np.loadtxt(path_load+'viscX'),np.loadtxt(path_load+'viscY'),np.loadtxt(path_load+'presX'),\
                  np.loadtxt(path_load+'presY'),np.loadtxt(path_load+'YS'),np.loadtxt(path_load+'XS')
        centers = np.loadtxt(path_load+'centers')

        Nsphere = (history[0].shape)[1]
        if True:plot_details(frate,history,Nsphere,u,v,p,[0,0],centers,time_)
        else: plotFields(u,v,p,[0,0],centers)
         
        #  Nsphere = 1
        #  plot_details(frate,history,Nsphere,u,v,p,[0,0],[centers])
        #plot_details(frate,history,1,u,v,p,[0,0])
        show()

      else:  
        path_load = config.get_nameSvg()+args[0]+'_'
        #saveflowaxis(v)
        frate = np.loadtxt(path_load+'flow_rate')
        plt.plot(frate)
        plt.show()
