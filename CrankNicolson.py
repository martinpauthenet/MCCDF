def gradv(bcs,temp_):

    uInp = temp_
    uInp[:,1:-1] = (u[:,:-1]+u[:,1:])/2
    if bcs == 'periodic':
     uInp[:,0] = uInp[:,nx-1]
     uInp[:,-1] = uInp[:,1]
    elif bcs == 'wall':
     uInp[:,0] = 2*u[:,0]  - uInp[:,1]
     uInp[:,-1] = 2*u[:,-1]  - uInp[:,-2]
    grad(uInp,gxu,gyu)

    vInp = temp_
    vInp[1:-1,:] = (v[:-1,:]+v[1:,:])/2
    if bcs == 'periodic':
     vInp[0,:] = vInp[ny-1,:]
     vInp[-1,:] = vInp[1,:]
    if bcs == 'wall':
     vInp[0,:] = 2*v[0,:] - vInp[1,:]
     vInp[-1,:] = 2*v[-1,:] - vInp[-2,:]
    grad(vInp,gxv,gyv)

    return

def convective(bcs):

    gradv(bcs,temp_)

    temp_v = temp_[:-1,:]#np.zeros((ny,nx+1))
    temp_v[:,1:-1] = v[:,1:-1]*gyu[:,:]
    temp_v[:,[0,-1]] = temp_v[:,[-2,1]]
    conv_u1[:,:] = u[1:-1,:]*gxu[:,:] + (temp_v[:-1,:-1]+temp_v[1:,1:]+temp_v[1:,:-1]+temp_v[:-1,1:])/4.

    temp_[:-1,:] = 0.

    temp_u = temp_[:,:-1]#np.zeros((ny+1,nx))
    temp_u[1:-1,:] = u[1:-1,:]*gxv[:,:]
    temp_u[[0,-1],:] = temp_u[[-2,1],:]
    conv_v1[:,:] = v[:,1:-1]*gyv[:,:] + (temp_u[:-1,:-1]+temp_u[1:,1:]+temp_u[:-1,1:]+temp_u[1:,:-1])/4.

    return     

def grad(phi,gx,gy):
    # grad for a pressure point field
    gx[:,:] = (phi[1:-1,1:] - phi[1:-1,:-1])/dx 
    gy[:,:] = (phi[1:,1:-1] - phi[:-1,1:-1])/dy
    return gx,gy
    
def set_dirichlet(pseudo,fFs,source,vels,yg):
  n=0
  for fF in fFs:
    source[yg.get_j_abs(pseudo[n]),yg.get_i_abs(pseudo[n])] = fF*vels
    n+=1
  return

def set_dirichlet_at_pseudo(fFu,fFv):

  for s_n in local_sphere_numbers:
    us,vs = config.get_solidVeloPosit(s_n)[1]
    if mpiRank == 0:
      pseudo = config.get_pseudo(0,s_n)[0]
      set_dirichlet(pseudo,fFu[s_n],sourceu,us,yu)
      pseudo = config.get_pseudo(1,s_n)[0]
      set_dirichlet(pseudo,fFv[s_n],sourcev,vs,yv)
    else:
      pseudo = config.get_pseudo(0,s_n)[0]
      comm.send([pseudo,fFu[s_n],us],dest=0,tag=10*s_n+0)
      pseudo = config.get_pseudo(1,s_n)[0]
      comm.send([pseudo,fFv[s_n],vs],dest=0,tag=10*s_n+1)
  if mpiRank == 0:
    for proc in range(1,cpu):
      for s_n in global_sphere_number[proc]:
        pseudou,fFsu,velsu = comm.recv(source=proc,tag=10*s_n+0)
        set_dirichlet(pseudou,fFsu,sourceu,velsu,yu)
        pseudov,fFsv,velsv = comm.recv(source=proc,tag=10*s_n+1)
        set_dirichlet(pseudov,fFsv,sourcev,velsv,yv)

  return

def velocity_BCs(fFu,fFv,lb):
    # velocity fields extension
    s_n_loc=0
    for s_n in local_sphere_numbers:
      us,vs = config.get_solidVeloPosit(s_n)[1]
      orx,ory = lb[s_n_loc][[0,2]]
      solid_FSI.linearizeVelocity_pseudo_set(buffer_u[s_n_loc],s_n,fFu[s_n],us,orx-1,ory,mpiRank)
      solid_FSI.linearizeVelocity_pseudo_set(buffer_v[s_n_loc],s_n,fFv[s_n],vs,orx,ory-1,mpiRank)
      s_n_loc+=1
    return

def pressure_BCs(stress_coefp,interp_pseudo,xp,yp,local_bornes):
    # pressure field extension
    s_n_loc=0
    for s_n in local_sphere_numbers:
      center = config.get_centers(s_n)
      origine = local_bornes[s_n_loc][[0,2]]
      pseudop,proj_pseudop = config.get_pseudo(2,s_n)
      dvsdt=config.get_solidVeloPosit(s_n)[0]
      solid_FSI.pressureLinear_set(buffer_p[s_n_loc],xp,yp,stress_coefp[s_n],interp_pseudo[s_n],dvsdt,pseudop,proj_pseudop,origine,center)
      s_n_loc+=1
    return

def _save(time_time,frate,history,global_sphere_number):
  vmv_x,vmv_y,vmp_x,vmp_y,YS_,XS_ = np.zeros((_Nit,Nsphere),dtype=np.float64),np.zeros((_Nit,Nsphere),dtype=np.float64),\
                                    np.zeros((_Nit,Nsphere),dtype=np.float64),np.zeros((_Nit,Nsphere),dtype=np.float64),\
                                    np.zeros((_Nit,Nsphere),dtype=np.float64),np.zeros((_Nit,Nsphere),dtype=np.float64)
  for cp in range(cpu):
    vmvx,vmvy,vmpx,vmpy,YS,XS = history[cp]
    for sn_index in range(len(global_sphere_number[cp])):
      sn = global_sphere_number[cp][sn_index]
      vmv_x[:,sn],vmv_y[:,sn],vmp_x[:,sn],vmp_y[:,sn],YS_[:,sn],XS_[:,sn] = vmvx[:,sn_index],vmvy[:,sn_index],\
                                                                            vmpx[:,sn_index],vmpy[:,sn_index],\
                                                                            YS[:,sn_index],XS[:,sn_index]
  path_save = config.get_nameSvg()+str(config.getTime())+'_'
  np.savetxt(path_save+'time', time_time)
  np.savetxt(path_save+'flow_rate', frate)
  np.savetxt(path_save+'u', u)
  np.savetxt(path_save+'v', v)
  np.savetxt(path_save+'p', p)
  np.savetxt(path_save+'viscX', vmv_x)
  np.savetxt(path_save+'viscY', vmv_y)
  np.savetxt(path_save+'presX', vmp_x)
  np.savetxt(path_save+'presY', vmp_y)
  np.savetxt(path_save+'XS', XS_)
  np.savetxt(path_save+'YS', YS_)  

  np.savetxt(path_save+'centers', config.centers)
  np.savetxt(path_save+'dvs', config.dvs)
  np.savetxt(path_save+'vs', config.vs)

  return

def integrate_stress(forces,s_n_global,s_n_local,XS,YS,vmvx,vmvy,vmpx,vmpy):
         fvx,fvy,fpx,fpy = forces
         vmvx[config.getTime(),s_n_local],vmvy[config.getTime(),s_n_local] = fvx,fvy
         vmpx[config.getTime(),s_n_local],vmpy[config.getTime(),s_n_local] = fpx,fpy
         dvss,vss = config.get_solidVeloPosit(s_n_global)[:-1]
         center = config.get_centers(s_n_global)
         center0 = config.get_centers0(s_n_global)
         XS[config.getTime(),s_n_local] = center[0]
         YS[config.getTime(),s_n_local] = center[1]

         vssx_new = 0.
         vssy_new = 0.
         if config.get_follow():
           err=1e6
           #while err>1e-10:
           for n in range(1):
             stretchY = center0[1]-center[1]
             stretchX = center0[0]-center[0]
             if config.getTime() > 0:
               vssy_new = vss[1] + 0.5*((vmvy[config.getTime()-1,s_n_local]+vmpy[config.getTime()-1,s_n_local]+config.K*(center0[1]-YS[config.getTime()-1,s_n_local])) +\
                                         fvy+fpy+config.K*(stretchY))*config.dt/config.m_p
               vssx_new = vss[0] + 0.5*((vmvx[config.getTime()-1,s_n_local]+vmpx[config.getTime()-1,s_n_local]+config.K*(center0[0]-XS[config.getTime()-1,s_n_local])) +\
                                         fvx+fpx+config.K*(stretchX))*config.dt/config.m_p
             else:
               vssy_new = vss[1] + (fvy+fpy+config.K*(stretchY))*config.dt/config.m_p
               vssx_new = vss[0] + (fvx+fpx+config.K*(stretchX))*config.dt/config.m_p

             cx = XS[config.getTime(),s_n_local] + (vss[0] + vssx_new)/2.*config.dt
             cy = YS[config.getTime(),s_n_local] + (vss[1] + vssy_new)/2.*config.dt
             #err = max(math.fabs(center[0]-cx),math.fabs(center[1]-cy))/config.R
             #print err, center[0]-cx,center[1]-cy
             center[0] = cx
             center[1] = cy

           dvss[0] = (vssx_new - vss[0])/config.dt
           dvss[1] = (vssy_new - vss[1])/config.dt
            #center[0] = center[0] + (vss[0] + vssx_new)/2.*config.dt
            #center[1] = center[1] + (vss[1] + vssy_new)/2.*config.dt
         #print dvss,vss,fvx,fpx,s_n_global
         vss[0] = vssx_new
         vss[1] = vssy_new
         return dvss,vss,center

def interm(nsp,coefs_for_psisource,local_bornes):
    if mpiRank==0: 
      sourcePsi()
      psisource[nsp[1],nsp[0]] = 0.

    scatter_psisource(local_bornes)
    cut_cells.reshape_psisourceTF(buffer_u,buffer_v,xp,yp,buffer_pisou,coefs_for_psisource,local_bornes,local_sphere_numbers)
    gather_psisource(local_bornes)

    return

def nspp(local_bornes):
  nsp_s = [None]*3
  if mpiRank == 0: yg = yu,yv,yp
  for fnum in [0,1,2]:
    if mpiRank == 0: nspi,nspj=np.array([],dtype=np.int),np.array([],dtype=np.int)
    else: solidpop = np.array([],dtype=np.int)
    s_n_loc=0
    for s_n in local_sphere_numbers:
      #Flags = config.get_flags_slice(fnum,s_n)
      if mpiRank == 0:
        solidpop = config.get_solidPops(s_n)[fnum]
        #nspi,nspj = np.append(nspi,np.array(map(yg[fnum].get_i,solidpop))),np.append(nspj,np.array(map(yg[fnum].get_j,solidpop)))
        nspi = np.append(nspi,np.array([yg[fnum].get_i_abs(x) for x in solidpop]))
        nspj = np.append(nspj,np.array([yg[fnum].get_j_abs(x) for x in solidpop]))
      else:
        solidpop = np.append(solidpop,list(config.get_solidPops(s_n)[fnum]))
      s_n_loc+=1
    if mpiRank == 0:
      for n in range(1,cpu):
        nsolpop = comm.recv(source=n,tag=77)
        if nsolpop > 0:
          s_buffer = np.zeros(nsolpop,dtype=np.int)
          comm.Recv([s_buffer,MPI.INT],source=n,tag=78)
          #nspi = np.append(nspi,np.array(map(yg[fnum].get_i,s_buffer)))
          #nspj = np.append(nspj,np.array(map(yg[fnum].get_j,s_buffer)))          
          nspi = np.append(nspi,np.array([yg[fnum].get_i_abs(x) for x in s_buffer]))
          nspj = np.append(nspj,np.array([yg[fnum].get_j_abs(x) for x in s_buffer]))
      nsp_s[fnum] = (nspi,nspj)
    else:
      comm.send(len(solidpop),dest=0,tag=77)
      if len(solidpop) > 0:
        comm.Send([solidpop,MPI.INT],dest=0,tag = 78)
  return nsp_s

def initialize(Nit,Nsphere_local):
    if mpiRank == 0: 
      frate = np.zeros(Nit)
      time_time = np.zeros(Nit)
      if config.get_load():
       path_load = config.get_nameLoad()
       uL = np.loadtxt(path_load+'u')
       vL = np.loadtxt(path_load+'v')
       pL = np.loadtxt(path_load+'p')
       if not config.get_load_coarse():
        nxL,nyL = uL.shape[1]-1,uL.shape[0]-2
        for j in range((nx-1)/nxL):
          for i in range((ny-1)/nyL):
            u[1+i*nyL:1+(i+1)*nyL,j*nxL:(j+1)*nxL] = uL[1:-1,:-1]
            v[i*nyL:(i+1)*nyL,1+j*nxL:1+(j+1)*nxL] = vL[:-1,1:-1]
            p[1+i*nyL:1+(i+1)*nyL,1+j*nxL:1+(j+1)*nxL] = pL[1:-1,1:-1]
       else:
        from scipy.interpolate import griddata
        nxL,nyL = uL.shape[1],uL.shape[0]-2
        xL, yL = np.linspace(0.,Lx,nxL), np.linspace(0.+0.5*dx,Ly-0.5*dx,nyL)
        xL,yL = np.meshgrid(xL,yL)
        nxu,nyu = u.shape[1],u.shape[0]-2
        xu, yu = np.linspace(0.,Lx,nxu), np.linspace(0.+0.5*dx,Ly-0.5*dx,nyu)
        xu,yu = np.meshgrid(xu,yu)
        urav = u[1:-1,:].flat
        urav[:] = griddata((xL.ravel(),yL.ravel()),uL[1:-1,:].ravel(),(xu.ravel(),yu.ravel()),method='linear')
        
        nxL,nyL = vL.shape[1]-2,vL.shape[0]
        xL, yL = np.linspace(0.+0.5*dx,Lx-0.5*dx,nxL), np.linspace(0.,Ly,nyL)
        xL,yL = np.meshgrid(xL,yL)
        nxv,nyv = v.shape[1]-2,v.shape[0]
        xv, yv = np.linspace(0.+0.5*dx,Lx-0.5*dx,nxv), np.linspace(0.,Ly,nyv)
        xv,yv = np.meshgrid(xv,yv)
        vrav = v[:,1:-1].flat
        vrav[:] = griddata((xL.ravel(),yL.ravel()),vL[:,1:-1].ravel(),(xv.ravel(),yv.ravel()),method='linear')

        nxL,nyL = pL.shape[1]-2,pL.shape[0]-2
        xL, yL = np.linspace(0.+0.5*dx,Lx-0.5*dx,nxL), np.linspace(0.+0.5*dx,Ly-0.5*dx,nyL)
        xL,yL = np.meshgrid(xL,yL)
        nxp,nyp = p.shape[1]-2,p.shape[0]-2
        xp, yp =  np.linspace(0.+0.5*dx,Lx-0.5*dx,nxp), np.linspace(0.+0.5*dx,Ly-0.5*dx,nyp)
        xp,yp = np.meshgrid(xp,yp)
        prav = p[1:-1,1:-1].flat
        prav[:] = griddata((xL.ravel(),yL.ravel()),pL[1:-1,1:-1].ravel(),(xp.ravel(),yp.ravel()),method='linear')
 
      else: u[:,:],v[:,:],p[:,:] = 0.,0.,0.

      u[[0,-1],:] = u[[-2,1],:]
      u[:,-1] = u[:,0]
      v[:,[0,-1]] = v[:,[-2,1]]
      v[-1,:] = v[0,:]
      p[[0,-1],:] = p[[-2,1],:]
      p[:,[0,-1]] = p[:,[-2,1]]

    else: 
      frate = None
      time_time = None

    vmvx = np.zeros((Nit,Nsphere_local),dtype=np.float64)
    vmvy = np.zeros((Nit,Nsphere_local),dtype=np.float64)
    vmpx = np.zeros((Nit,Nsphere_local),dtype=np.float64)
    vmpy = np.zeros((Nit,Nsphere_local),dtype=np.float64)
    XS = np.zeros((Nit,Nsphere_local),dtype=np.float64)
    YS = np.zeros((Nit,Nsphere_local),dtype=np.float64)

    return vmvx,vmvy,vmpx,vmpy,XS,YS,frate,time_time

def post_process(): return config.dt/dx*max(np.amax(np.abs(u[:,:])), np.amax(np.abs(v[:,:])))

def writes_solid_into_config(local_bornes):
      config.reset_flags(local_sphere_numbers)
      s_n_loc=0
      for s_n in local_sphere_numbers:
        bornes = local_bornes[s_n_loc]
        result = solid_FSI.set_flags(xu,yu,xv,yv,xp,yp,s_n,bornes)
        config.set_solidPoints(result,s_n)
        s_n_loc+=1
      return

def send_csr(buff,field,s_n,proc):
          comm.send([len(buff.data),len(buff.indices),len(buff.indptr)],dest=proc,tag=100*s_n+10*field+3)
          comm.Send([np.array(buff.data,dtype=np.float),MPI.DOUBLE],dest = proc,tag = 100*s_n+10*field+0)
          comm.Send([np.array(buff.indices,dtype=np.int),MPI.INT],dest = proc,tag = 100*s_n+10*field+1)
          comm.Send([np.array(buff.indptr,dtype=np.int),MPI.INT],dest = proc,tag = 100*s_n+10*field+2)
          return
def recv_csr(field,s_n,proc):
          ldata,lindices,lindptr = comm.recv(source = proc,tag = 100*s_n+10*field+3)
          data,indices,indptr = np.zeros(ldata,dtype=np.float),np.zeros(lindices,dtype=np.int),np.zeros(lindptr,dtype=np.int)
          comm.Recv([data,MPI.DOUBLE],source = proc,tag = 100*s_n+10*field+0)
          comm.Recv([indices,MPI.INT],source = proc,tag = 100*s_n+10*field+1)
          comm.Recv([indptr,MPI.INT],source = proc,tag = 100*s_n+10*field+2)
          return data,indices,indptr

def send_csr_bp(buff):
          comm.send([len(buff.data),len(buff.indices),len(buff.indptr)],dest=0,tag=3)
          comm.Send([np.array(buff.data,dtype=np.float),MPI.DOUBLE],dest = 0,tag=0)
          comm.Send([np.array(buff.indices,dtype=np.int),MPI.INT],dest = 0,tag=1)
          comm.Send([np.array(buff.indptr,dtype=np.int),MPI.INT],dest = 0,tag=2)
          return
def recv_csr_bp(proc):
          ldata,lindices,lindptr = comm.recv(source = proc,tag=3)
          data,indices,indptr = np.zeros(ldata,dtype=np.float),np.zeros(lindices,dtype=np.int),np.zeros(lindptr,dtype=np.int)
          comm.Recv([data,MPI.DOUBLE],source = proc,tag=0)
          comm.Recv([indices,MPI.INT],source = proc,tag=1)
          comm.Recv([indptr,MPI.INT],source = proc,tag=2)
          return data,indices,indptr

def gather_local_fields(local_bornes,fields):
      if mpiRank == 0:
        for proc in range(1,cpu):
          for s_n in global_sphere_number[proc]:
            borne = comm.recv(source=proc,tag=77)
            if 0 in fields: assign_slice_periodic(u,borne[2],borne[3],borne[0]-1,borne[1],xu,yu,csr_matrix(recv_csr(0,s_n,proc),shape = (window,window+1)).todense(),True)
            if 1 in fields: assign_slice_periodic(v,borne[2]-1,borne[3],borne[0],borne[1],xv,yv,csr_matrix(recv_csr(1,s_n,proc),shape = (window+1,window)).todense(),True)
            #if 0 in fields: assign_slice_periodic(u,borne[2]-1,borne[3],borne[0],borne[1],xu,yu,csr_matrix(recv_csr(0,s_n,proc),shape = (window+1,window)).todense())
            #if 1 in fields: assign_slice_periodic(v,borne[2],borne[3],borne[0]-1,borne[1],xv,yv,csr_matrix(recv_csr(1,s_n,proc),shape = (window,window+1)).todense())
            if 2 in fields: assign_slice_periodic(p,borne[2],borne[3],borne[0],borne[1],xp,yp,csr_matrix(recv_csr(2,s_n,proc),shape = (window,window)).todense(),True)
        for s_n_loc in range(Nsphere_local):
          borne = local_bornes[s_n_loc]
          if 0 in fields: assign_slice_periodic(u,borne[2],borne[3],borne[0]-1,borne[1],xu,yu,buffer_u[s_n_loc],True)
          #print np.sum(buffer_u[s_n_loc]),config.get_centers0(s_n_loc),config.get_centers(s_n_loc),s_n_loc
          if 1 in fields: assign_slice_periodic(v,borne[2]-1,borne[3],borne[0],borne[1],xv,yv,buffer_v[s_n_loc],True)
          #if 0 in fields: assign_slice_periodic(u,borne[2]-1,borne[3],borne[0],borne[1],xu,yu,buffer_u[s_n_loc])
          #if 1 in fields: assign_slice_periodic(v,borne[2],borne[3],borne[0]-1,borne[1],xv,yv,buffer_v[s_n_loc])
          if 2 in fields: assign_slice_periodic(p,borne[2],borne[3],borne[0],borne[1],xp,yp,buffer_p[s_n_loc],True)
        u[[0,-1],:] = u[[-2,1],:]
        u[:,-1] = u[:,0]
        v[:,[0,-1]] = v[:,[-2,1]]
        v[-1,:] = v[0,:]
        p[[0,-1],:] = p[[-2,1],:]
        p[:,[0,-1]] = p[:,[-2,1]]
      else:
        s_n_loc=0
        for s_n in local_sphere_numbers:
          comm.send(local_bornes[s_n_loc],dest=0,tag=77)
          if 0 in fields: send_csr(csr_matrix(buffer_u[s_n_loc]),0,s_n,0)
          if 1 in fields: send_csr(csr_matrix(buffer_v[s_n_loc]),1,s_n,0)
          if 2 in fields: send_csr(csr_matrix(buffer_p[s_n_loc]),2,s_n,0)
          s_n_loc+=1
      #print np.amax(np.abs(buffer_pisou[0]-buffer_pisou[1]))
      #print np.amax(np.abs(buffer_pisou[3]-buffer_pisou[2]))
      #print np.amax(np.abs(buffer_u[0]-buffer_u[1]))
      #print np.amax(np.abs(buffer_u[0]-buffer_u[1]))
      #print np.amax(np.abs(buffer_v[3]-buffer_v[2]))
      #print np.amax(np.abs(buffer_v[3]-buffer_v[2]))

      return

def scatter_local_fields(local_bornes,fields):
      if mpiRank == 0:
        for proc in range(1,cpu):
          for s_n in global_sphere_number[proc]:
            borne = comm.recv(source=proc,tag=77)
            if 0 in fields: 
              slice_periodic(buffer_fl[:-1,:],u,borne[2],borne[3],borne[0]-1,borne[1],xu,yu) #u[borne[2]-1:borne[3],borne[0]:borne[1]]
              #slice_periodic(buffer_fl[:,:-1],u,borne[2]-1,borne[3],borne[0],borne[1],xu,yu) #u[borne[2]-1:borne[3],borne[0]:borne[1]]
              send_csr(csr_matrix(buffer_fl[:-1,:]),0,s_n,proc)
            if 1 in fields: 
              slice_periodic(buffer_fl[:,:-1],v,borne[2]-1,borne[3],borne[0],borne[1],xv,yv) #v[borne[2]:borne[3],borne[0]-1:borne[1]]
              #slice_periodic(buffer_fl[:-1,:],v,borne[2],borne[3],borne[0]-1,borne[1],xv,yv) #v[borne[2]:borne[3],borne[0]-1:borne[1]]
              send_csr(csr_matrix(buffer_fl[:,:-1]),1,s_n,proc)
            if 2 in fields:
              slice_periodic(buffer_fl[:-1,:-1],p,borne[2],borne[3],borne[0],borne[1],xp,yp) #p[borne[2]:borne[3],borne[0]:borne[1]]
              send_csr(csr_matrix(buffer_fl[:-1,:-1]),2,s_n,proc)
        for s_n_loc in range(Nsphere_local):
          borne = local_bornes[s_n_loc]
          slice_periodic(buffer_u[s_n_loc],u,borne[2],borne[3],borne[0]-1,borne[1],xu,yu) #u[borne[2]-1:borne[3],borne[0]:borne[1]]
          slice_periodic(buffer_v[s_n_loc],v,borne[2]-1,borne[3],borne[0],borne[1],xv,yv) #v[borne[2]:borne[3],borne[0]-1:borne[1]]
          #slice_periodic(buffer_u[s_n_loc],u,borne[2]-1,borne[3],borne[0],borne[1],xu,yu) #u[borne[2]-1:borne[3],borne[0]:borne[1]]
          #slice_periodic(buffer_v[s_n_loc],v,borne[2],borne[3],borne[0]-1,borne[1],xv,yv) #v[borne[2]:borne[3],borne[0]-1:borne[1]]
          slice_periodic(buffer_p[s_n_loc],p,borne[2],borne[3],borne[0],borne[1],xp,yp) #p[borne[2]:borne[3],borne[0]:borne[1]]
      else:
        s_n_loc=0
        for s_n in local_sphere_numbers:
          comm.send(local_bornes[s_n_loc],dest=0,tag=77)          
          if 0 in fields:
            A = csr_matrix(recv_csr(0,s_n,0),shape = (window,window+1)).todense()
            buffer_u[s_n_loc][:,:] = A          
          if 1 in fields:
            A = csr_matrix(recv_csr(1,s_n,0),shape = (window+1,window)).todense()
            buffer_v[s_n_loc][:,:] = A          
          if 2 in fields:
            A = csr_matrix(recv_csr(2,s_n,0),shape = (window,window)).todense()
            buffer_p[s_n_loc][:,:] = A
          s_n_loc+=1
      return

def gather_psisource(local_bornes):
      if mpiRank == 0:
        for proc in range(1,cpu):
          for s_n in global_sphere_number[proc]:
            borne = comm.recv(source=proc,tag=77)
            assign_slice_periodic(psisource,borne[2],borne[3],borne[0],borne[1],xp,yp,csr_matrix(recv_csr(2,s_n,proc),shape = (window,window)).todense(),False)
        for s_n_loc in range(Nsphere_local):
          borne = local_bornes[s_n_loc]
          assign_slice_periodic(psisource,borne[2],borne[3],borne[0],borne[1],xp,yp,buffer_pisou[s_n_loc],False)
          #print np.amax(np.abs(buffer_pisou[s_n_loc]))
      else:
        s_n_loc=0
        for s_n in local_sphere_numbers:
          comm.send(local_bornes[s_n_loc],dest=0,tag=77)
          send_csr(csr_matrix(buffer_pisou[s_n_loc]),2,s_n,0)
          s_n_loc+=1

      return

def scatter_psisource(local_bornes):
      if mpiRank == 0:
        for proc in range(1,cpu):
          for s_n in global_sphere_number[proc]:
            borne = comm.recv(source=proc,tag=77)
            slice_periodic(buffer_fl[:-1,:-1],psisource,borne[2],borne[3],borne[0],borne[1],xp,yp) #psisource[borne[2]:borne[3],borne[0]:borne[1]]
            send_csr(csr_matrix(buffer_fl[:-1,:-1]),2,s_n,proc)
        for s_n_loc in range(Nsphere_local):
          borne = local_bornes[s_n_loc]
          slice_periodic(buffer_pisou[s_n_loc],psisource,borne[2],borne[3],borne[0],borne[1],xp,yp) #psisource[borne[2]:borne[3],borne[0]:borne[1]]
      else:
        s_n_loc=0
        for s_n in local_sphere_numbers:
          comm.send(local_bornes[s_n_loc],dest=0,tag=77)
          buffer_pisou[s_n_loc][:,:] = csr_matrix(recv_csr(2,s_n,0),shape = (window,window)).todense()
          s_n_loc+=1
      return

def lmb(psisource,velomoy): 
  return np.amax(np.abs(psisource[1:-1,1:-1]))/(dx/config.dt*velomoy)

def append_connexions(ligne,indices,indptr,boundary):
  for ptr in range(indptr[ligne],indptr[ligne+1]):
    if not indices[ptr] in boundary: 
      boundary.append(indices[ptr])
      append_connexions(indices[ptr],indices,indptr,boundary)
  return

def add_csrs(solver_,matcsr_,_AT_,a_permanent,Nx,Ny,nullspace):
  N = Nx*Ny
  csrs_to_add = [None]*(cpu)
  if mpiRank == 0:
    csrs_to_add[0] = (matcsr_.data,matcsr_.indices,matcsr_.indptr)
    for proc in range(1,cpu): 
      csrs_to_add[proc] = recv_csr_bp(proc)
    matcsr_add = matrix.add_csr(csrs_to_add,N)
    if not _AT_ is None: matcsr_add = matcsr_add + _AT_
  else: 
    send_csr_bp(matcsr_)
  if not a_permanent is None:
    if mpiRank == 0:
      #if nullspace: matcsr_add = a_permanent
      matcsr_add = matcsr_add + a_permanent
      matcsr_add.data *= np.abs(matcsr_add.data) > 1e-15
      matcsr_add.eliminate_zeros()
      #print np.amin(np.abs(matcsr_add.data))
      csr_ = matcsr_add.indptr,matcsr_add.indices,matcsr_add.data
    else: 
      csr_ = None
      matcsr_add = 0.
    solver_.set_Mat_directMethod(csr_,nullspace)

  else: print 'SHOULD BE CORRECTIVE METHOD'
  return matcsr_add

def sourcePsi():
    psisource[1:-1,1:-1] = (u[1:-1,1:] - u[1:-1,:-1])*dy/config.dt + (v[1:,1:-1] - v[:-1,1:-1])*dx/config.dt #!!
    return

def laplaci_n(Ut,Vt):
  Su[:] = 0.5*nu/dx**2*Ut.dot(U) #!!
  Sv[:] = 0.5*nu/dx**2*Vt.dot(V) #!!
  return

def navierstokes_forcingCNi(nsp_p,nsp_u,nsp_v,prev_nsp_u,prev_nsp_v,ffu8,ffv8, matcsr_penal_u,matcsr_IBM_u,atu,\
                                                                         matcsr_penal_v,matcsr_IBM_v,atv,cfl):
#Implicit treatment of the viscous terms eliminates the numerical viscous stability restriction. bof in fact
#This restriction is particularly severe for low-Reynolds-number flows and near boundaries.    
  if mpiRank == 0:
    conv_u2[:,:], conv_v2[:,:] = conv_u1, conv_v1
    convective('periodic')
    conv_u2[prev_nsp_u[1]-1,prev_nsp_u[0]], conv_v2[prev_nsp_v[1],prev_nsp_v[0]-1] = conv_u1[prev_nsp_u[1]-1,prev_nsp_u[0]], conv_v1[prev_nsp_v[1],prev_nsp_v[0]-1]
    grad(p,gxp,gyp)

    mpcu = np.abs(u[1:ny,:] + config.dt*( - gxp - conv_u1 - 1./2.*(config.dt/config.dt_old)*(conv_u1-conv_u2) + 2.*sourceu[1:ny,:] + fu ))
    mpcv = np.abs(v[:,1:nx] + config.dt*( - gyp - conv_v1 - 1./2.*(config.dt/config.dt_old)*(conv_v1-conv_v2) + 2.*sourcev[:,1:nx] + fv ))
    max_predicted_cfl = max(cfl,max(np.amax(mpcu),np.amax(mpcv))/dx*config.dt)

  global forwards
  if mpiRank == 0: forwards = [max_predicted_cfl]*cpu
  max_predicted_cfl = comm.scatter(forwards, root=0)
  global reboot
  if max_predicted_cfl > 0.:
    if mpiRank == 0: reboot = reboot or config.set_dt(max_predicted_cfl)
    else: config.set_dt(max_predicted_cfl)

  # ASSEMBLE !! #
  if True or config.getTime()==0 or config.variable_time_step or config.get_follow():
    add_csrs(solver_u, config.dt*matcsr_penal_u-matcsr_IBM_u, config.dt*atu, aperu, nx, ny+1, False) # identite + 0.5*nu*dt/dx**2*M(penal) - coefs_interp_triangle - 0.5*nu/dx**2*dt*M !!
    add_csrs(solver_v, config.dt*matcsr_penal_v-matcsr_IBM_v, config.dt*atv, aperv, nx+1, ny,False)

  if mpiRank == 0:
   sourceu[1:ny,:-1] = u[1:ny,:-1] + config.dt*( - gxp - conv_u1 - 1./2.*(config.dt/config.dt_old)*(conv_u1-conv_u2) + sourceu[1:ny,:] + fu )[:,:-1]
   sourceu[nsp_u[1],nsp_u[0]] = 0.
   #sourceu[:,-1] = 0.
   sourcev[:-1,1:nx] = v[:-1,1:nx] + config.dt*( - gyp - conv_v1 - 1./2.*(config.dt/config.dt_old)*(conv_v1-conv_v2) + sourcev[:,1:nx] + fv )[:-1,:]
   sourcev[nsp_v[1],nsp_v[0]] = 0.
   #sourcev[-1,:] = 0.

  if config.get_follow():
  ## set dirichlet time n+1 at pseudo
    set_dirichlet_at_pseudo(ffu8,ffv8)

  solver_u.set_Vectors(Su)
  solver_u.solveLinear()
  if not mpiRank == 0: solver_u.get_result()
  else: U[:] = solver_u.get_result()
  #else: solver_u.get_result()

  if False and config.getTime()>200:
        import matplotlib.pyplot as plt
        #A=matcsr.todense()
        #x = np.ones((ny+1,nx))
        #xf = x.flat
        #xf[:] = A.dot(xf)
        #plt.imshow(A,interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        #A=matcsr
        A=Au
        X=U
        b=sourceu
        x = np.ones((ny+1,nx))
        xf = x.flat
        xf[:] = A.dot(X)
        plt.imshow(np.abs(x-b),interpolation='nearest')
        plt.colorbar()
        plt.show()

  solver_v.set_Vectors(Sv)
  solver_v.solveLinear()
  if not mpiRank == 0: solver_v.get_result()
  else: V[:] = solver_v.get_result()
  #else: solver_v.get_result()

  if False and config.getTime()>200:
        import matplotlib.pyplot as plt
        #A=matcsr.todense()
        #x = np.ones((ny+1,nx))
        #xf = x.flat
        #xf[:] = A.dot(xf)
        #plt.imshow(A,interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        #A=matcsr
        A=Av
        X=V
        b=sourcev
        x = np.ones((ny,nx+1))
        xf = x.flat
        xf[:] = A.dot(X)
        plt.imshow(np.abs(x-b),interpolation='nearest')
        plt.colorbar()
        plt.show()

  return

def slice_periodic(buff,field,ys,ye,xs,xe,xg,yg):
  x,y=range(xs,xe),range(ys,ye)
  for i in range(xe-xs):
    x[i] = xg.ind_perio_buffer(x[i])
  for i in range(ye-ys):
    y[i] = yg.ind_perio_buffer(y[i])
  ymin,ymax,xmin,xmax = [],[],[],[]
  i=0
  while i < len(x)-1:
    xmin.append(i)
    while i < len(x)-1 and x[i] < x[i+1]: i+=1
    xmax.append(i)
    i+=1
  i=0
  while i < len(y)-1:
    ymin.append(i)
    while i < len(y)-1 and y[i] < y[i+1]: i+=1
    ymax.append(i)
    i+=1
  for i in range(len(xmin)):
    for j in range(len(ymin)):
      buff[ymin[j]:ymax[j]+1,xmin[i]:xmax[i]+1] = field[y[ymin[j]]:y[ymax[j]]+1,x[xmin[i]]:x[xmax[i]]+1]
  #buff[:,:] = field[ys:ye,xs:xe]
  return

def assign_slice_periodic(field,ys,ye,xs,xe,xg,yg,data,bord):
  x,y=range(xs,xe),range(ys,ye)
  for i in range(xe-xs):
    x[i] = xg.ind_perio_buffer(x[i])
  for i in range(ye-ys):
    y[i] = yg.ind_perio_buffer(y[i])
  ymin,ymax,xmin,xmax = [],[],[],[]
  i=0
  while i < len(x)-1:
    xmin.append(i)
    while i < len(x)-1 and x[i] < x[i+1]: i+=1
    xmax.append(i)
    i+=1
  i=0
  while i < len(y)-1:
    ymin.append(i)
    while i < len(y)-1 and y[i] < y[i+1]: i+=1
    ymax.append(i)
    i+=1
  for i in range(len(xmin)):
    for j in range(len(ymin)):
      field[y[ymin[j]]:y[ymax[j]]+1,x[xmin[i]]:x[xmax[i]]+1] = data[ymin[j]:ymax[j]+1,xmin[i]:xmax[i]+1]
  #field[ys:ye,xs:xe] = data[:,:]
  return

def solveP_C():
    
    global forwards
    vmvx,vmvy,vmpx,vmpy,XS,YS,frate,time_time = initialize(_Nit,Nsphere_local)

    coefu_f,coefv_f,ffu8,ffv8,stress_coefu,stress_coefv,stress_coefp,sF_pointsu,sF_pointsv =\
    [None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere
    csr_ghost,csr_ghost_penal,interp_pseudo,interp_forcing,csr_reshape =\
    [None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere,[None]*Nsphere
    cfps = [None]*Nsphere_local

    if mpiRank == 0:
      Ut, Vt = matrix.matrixUt(), matrix.matrixVt()
      image = 0

    # load initial conditions to procs
    local_bornes = map(config.get_bornes, local_sphere_numbers)
    scatter_local_fields(local_bornes,[0,1,2])

    if True:
        if mpiRank == 0: print 'save...'
        history = comm.gather([vmvx,vmvy,vmpx,vmpy,YS,XS], root=0)
        gather_local_fields(local_bornes,[0,1,2])
        if mpiRank == 0:
          _save(time_time,frate,history,global_sphere_number)
          print 'saved'

    if mpiRank == 0: bigst = temps.clock()
    global reboot
    cfl=0.
    while config.getTime() < _Nit and not reboot : #solving NavierStokes equations until ...

      if config.getTime() == 0 or config.get_follow():
        local_bornes = map(config.get_bornes, local_sphere_numbers) #write bornes
        #print local_bornes
        writes_solid_into_config(local_bornes)
        s_n_loc=0
        matcsr_IBM_u,matcsr_IBM_v = csr_matrix(((ny+1)*(nx),(ny+1)*(nx)), dtype = np.float),csr_matrix(((nx+1)*(ny),(nx+1)*(ny)), dtype = np.float)
        matcsr_penal_u,matcsr_penal_v = csr_matrix(((ny+1)*(nx),(ny+1)*(nx)), dtype = np.float),csr_matrix(((nx+1)*(ny),(nx+1)*(ny)), dtype = np.float)
        for s_n in local_sphere_numbers:

          #plotu,plotv,plotp = True,True,True
          #plotu,plotv,plotp = True,False,False
          plotu,plotv,plotp = False,False,False
          axc=None
          if (plotu+plotv+plotp) and s_n==0:
           import postprocTOOLS as ppt
           import matplotlib.pyplot as plt
           (nsp_u,nsp_v,nsp_p) = nspp(local_bornes)
           ax = ppt.plotFields(u,v,p,nsp_p,config.centers)
          if (plotu+plotv+plotp): axc=ax

          if plotu: axu=ax
          else: axu=None

          origine = local_bornes[s_n_loc][0]-1,local_bornes[s_n_loc][2]
          (coefu_f[s_n], stress_coefu[s_n], matpenalu, matIBMu, ffu8[s_n],sF_pointsu[s_n]) = solid_FSI.linearizeVelocity_pseudoCNi(0,xu,yu,s_n,origine,axu)
          matcsr_IBM_u = matcsr_IBM_u + matIBMu #0 + ( 0.5*nu*dt/dx**2*M(penal) - coefs_interp_triangle )!!
          matcsr_penal_u = matcsr_penal_u + matpenalu

          if plotv: axv=ax
          else: axv=None

          origine = local_bornes[s_n_loc][0],local_bornes[s_n_loc][2]-1
          (coefv_f[s_n], stress_coefv[s_n], matpenalv, matIBMv, ffv8[s_n],sF_pointsv[s_n]) = solid_FSI.linearizeVelocity_pseudoCNi(1,xv,yv,s_n,origine,axv)
         # print np.amax(np.abs(ffv8[s_n]))
          matcsr_IBM_v = matcsr_IBM_v + matIBMv
          matcsr_penal_v = matcsr_penal_v + matpenalv

          if plotp: axp=ax
          else: axp=None

          origine = local_bornes[s_n_loc][0],local_bornes[s_n_loc][2]
          (csr_ghost[s_n], stress_coefp[s_n], csr_ghost_penal[s_n], interp_pseudo[s_n]) = solid_FSI.ghostcellCorr(xp,yp,s_n,origine,axp) #!!
         # print np.amin(np.abs(stress_coefp[s_n]))
         # print (np.abs(interp_pseudo[s_n]-interp_pseudo[0]))
         # print local_bornes[s_n_loc] - local_bornes[0]

          rsc,slw = cut_cells.reshape_cells(xp,yp,s_n,origine)
          result = cut_cells.reshape_poisson(xu,yu,xv,yv,xp,yp,rsc,slw,s_n,coefu_f[s_n],coefv_f[s_n],origine,axc) #!!
          if (plotu+plotv+plotp) and s_n==len(local_sphere_numbers)-1: 
           plt.show()
           #plt.savefig('%4d'%config.getTime()+'.png')
           plt.close()

          csr_reshape[s_n] = result[0]
          cfps[s_n_loc] =    result[1] #!!
          s_n_loc+=1        
        matcsr_c = matrix.poisson_c(csr_reshape, csr_ghost, csr_ghost_penal,local_sphere_numbers) # -0.25*d/dx* + (1 - coefs_interp_triangle) + 0.25*M(penal) !!
        # ASSEMBLE only pressure
        #Ap = add_csrs(solver_p,matcsr_c,None,aperp,nx+1,ny+1,True)
        laplPsi = bcslapl - 4.*add_csrs(solver_p,matcsr_c,None,aperp,nx+1,ny+1,True) # -0.25*M - 0.25*d/dx* - coefs_interp_triangle + 0.25*M(penal) !!
        #print laplPsi

        #for arrx in stress_coefu[2]:
        # for x in arrx:
        #  err=1e36
        #  for arry in stress_coefu[3]:
        #   for y in arry:
        #    if abs(x-y)<err: err=abs(x-y)
        #  print err
        #import matplotlib.pyplot as plt
        #matcsr = add_csrs(solver_p,matcsr_c,None,aperp,nx+1,ny+1,True)
        #A=matcsr.todense()
        #x = np.ones((ny,nx+1))
        #xf = x.flat
        #xf[:] = A.dot(xf)
        #plt.imshow(A,interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        #A=matcsr
        #x = np.ones((ny+1,nx+1))
        #xf = x.flat
        #xf[:] = A.dot(xf)
        #plt.imshow(x,interpolation='nearest')
        #plt.colorbar()
        #plt.show()

        (nsp_u,nsp_v,nsp_p) = nspp(local_bornes)
        if config.getTime() == 0:
          previous_local_bornes = local_bornes
        if mpiRank==0 and config.getTime()==0:
          convective('periodic')
          conv_u2[:,:], conv_v2[:,:] = conv_u1, conv_v1
          prev_nsp_u,prev_nsp_v = nsp_u,nsp_v
        elif config.getTime()==0: prev_nsp_u,prev_nsp_v = None,None

######################################################################################################################
      if config.get_follow():
        ##################### update ghosts with new geometry, for source terms
        velocity_BCs(coefu_f,coefv_f,previous_local_bornes)
        pressure_BCs(stress_coefp,interp_pseudo,xp,yp,previous_local_bornes)
      #if mpiRank == 0: gather_local_fields(None,[0,1,2])
      gather_local_fields(previous_local_bornes,[0,1,2])

      if config.get_follow(): 
        previous_local_bornes = local_bornes #store previous bornes

      ## prepare source for CNi
      ## compute Laplacian time n
      if mpiRank == 0: laplaci_n(Ut,Vt)

      ##################### time advance
      navierstokes_forcingCNi(nsp_p,nsp_u,nsp_v,prev_nsp_u,prev_nsp_v,ffu8,ffv8, matcsr_penal_u,matcsr_IBM_u,atu,\
                                                                           matcsr_penal_v,matcsr_IBM_v,atv,cfl)
      
      if mpiRank==0 and config.get_follow(): prev_nsp_u,prev_nsp_v = nsp_u,nsp_v # store previous pseudo

      # predicted velocity
      scatter_local_fields(local_bornes,[0,1])

      if mpiRank == 0:
        st1,st2 = '',''
        velomoy = np.sum(np.sqrt(u[:-1,:]**2 + v[:,:-1]**2))/(nx*ny)  
        local_mass_balance = 1.
        
      if mpiRank == 0: interm(nsp_p,cfps,local_bornes)
      else: interm(None,cfps,local_bornes)
      nite = 1
      if mpiRank == 0:
          if velomoy > 0.:
            local_mass_balance = lmb(psisource,velomoy)
            st1 = '%.4e'%local_mass_balance

      for it in range(nite):
      #r = 3.
      #while r > 1.1:
         # distribute
         if mpiRank == 0:
           #print "projection. makes the rhs consistent. stabilizes."
           #psisource[:,:] -= np.sum(psisource)/(psisource.shape[0]*psisource.shape[1])

           #Psisource = -0.25 * psisource.ravel()
           psisource[:,:] *= -0.25
           solver_p.set_Vectors(Psisource) #!!
         else: solver_p.set_Vectors(None)
         # solve distributed
         solver_p.solveLinear()
         # gather result
         if not mpiRank == 0: solver_p.get_result()
         else:

           Psi[:] = solver_p.get_result()

           if False and config.getTime()>100:
             import matplotlib.pyplot as plt
        #A=matcsr.todense()
        #x = np.ones((ny+1,nx))
        #xf = x.flat
        #xf[:] = A.dot(xf)
        #plt.imshow(A,interpolation='nearest')
        #plt.colorbar()
        #plt.show()
        #A=matcsr
             #A=Ap
             #X=Psi
             #b=psisource
             #x = np.ones((ny+1,nx+1))
             #xf = x.flat
             #xf[:] = A.dot(X)
             plt.imshow(np.abs(sourceu),interpolation='nearest')
             plt.colorbar()
             plt.show()

           #u[:,:] = u - (psi[:,1:] - psi[:,:-1])/dx*config.dt
           #{v[:,:] = v - (psi[1:,:] - psi[:-1,:])/dy*config.dt
           u[1:-1,:] += - (psi[1:-1,1:] - psi[1:-1,:-1])/dx*config.dt
           v[:,1:-1] += - (psi[1:,1:-1] - psi[:-1,1:-1])/dy*config.dt

           # advance pressure CNi
           P[:]   -= 0.5*nu/dx**2*config.dt*(laplPsi.dot(Psi))
           p[:,:] += psi
           p[nsp_p[1],nsp_p[0]] = 0.
           # projects pressure
           p[:,:] -= np.sum(p[1:-1,1:-1]/((nx-1)*(ny-1)))
           p[nsp_p[1],nsp_p[0]] = 0.

         scatter_local_fields(local_bornes,[0,1,2])
         #print '##################### update ghosts after correction'
         velocity_BCs(coefu_f,coefv_f,local_bornes)
         pressure_BCs(stress_coefp,interp_pseudo,xp,yp,local_bornes)         

         #print config.get_centers(0)
         #print np.sum(buffer_u[0][:,-1])
         #print config.get_centers(1)
         #print np.sum(buffer_u[1][:,-1])

         if mpiRank==0: interm(nsp_p,cfps,local_bornes)
         else: interm(None,cfps,local_bornes)

      if mpiRank == 0:
         if velomoy > 0.:                          
           local_mass_balance = lmb(psisource,velomoy)
           st2 = '%.4e'%local_mass_balance
      
            # compute force
      if config.get_follow() or config.get_force():
        #print '# compute force'
        s_n_loc=0
        for s_n in local_sphere_numbers:
          ffsf = sF_pointsu[s_n],stress_coefu[s_n],sF_pointsv[s_n],stress_coefv[s_n]
          ffsp = interp_pseudo[s_n],stress_coefp[s_n]
          borne = local_bornes[s_n_loc]
          local = solid_FSI.totalForce(buffer_u[s_n_loc],buffer_v[s_n_loc],buffer_p[s_n_loc],xp,yp,s_n,ffsf,ffsp,borne[[0,2]])
          dvss_toset,vss_toset,center_toset = integrate_stress(local,s_n,s_n_loc,XS,YS,vmvx,vmvy,vmpx,vmpy)
          config.set_solidVeloPosit(dvss_toset,vss_toset,center_toset,s_n)
          s_n_loc+=1
        if mpiRank == 0:
          for n in range(1,cpu):
            lsn = comm.recv(source=n,tag=77)
            for s_n in lsn:
              dvs,vs,position = comm.recv(source=n,tag=s_n)
              config.set_solidVeloPosit(dvs,vs,position,s_n)
        else:
          comm.send(local_sphere_numbers,dest=0,tag=77)
          for s_n in local_sphere_numbers:
            dvs,vs,position = config.get_solidVeloPosit(s_n)
            comm.send([dvs,vs,position],dest=0,tag=s_n)

      if mpiRank == 0:
        UU = (u[1:-1,:-1]+u[1:-1,1:])/2.
        UU[nsp_p[1]-1,nsp_p[0]-1] = 0.
        Re = config.get_Reynolds(np.sum(UU/(ny-1)/(nx-1)))
        frate[config.getTime()] = Re
      config.incTime()

      if mpiRank == 0: 
        cfl = post_process()
      # forward reboot
        forwards = [reboot]*cpu
        if reboot: print ' cfl: '+str(cfl),' dt: '+str(config.dt),' << '+str(config.dt_min)
        else:
          time_time[config.getTime()-1] = config.TS
          print str(config.getTime())+'/'+str(_Nit)+' dt: '+'%.2e'%(config.dt)+' time: '+'%.2e'%(config.TS)+' CFL: '+'%.2e'%(cfl),'g: '+'%.6e'%(fu),\
                'Reynolds', '%.4e'%(Re),'leaks:', st1, '->', st2
      reboot = comm.scatter(forwards, root=0)

      if config.get_save_runtime():
        if mpiRank == 0: print 'save...'
        history = comm.gather([vmvx,vmvy,vmpx,vmpy,YS,XS], root=0)
        gather_local_fields(local_bornes,[0,1,2])
        if mpiRank == 0: 
          _save(time_time,frate,history,global_sphere_number)
          print 'saved'

    #plt.show()
    #plt.savefig('localFR.png',dpi = 200)
    if mpiRank == 0: print 'save...'
    history = comm.gather([vmvx,vmvy,vmpx,vmpy,YS,XS], root=0)
    gather_local_fields(local_bornes,[0,1,2])
    if mpiRank == 0:
      _save(time_time,frate,history,global_sphere_number)
      print 'saved'

    if mpiRank == 0: 
      print 'total execution time: ', temps.clock() - bigst
      return (frate,history,global_sphere_number),nsp_p    
    else: return

def create_solver(matcsr_,N,comm, cpu, mpiRank,maxite,relative, typeS):
  solver_ = petsc_solve.Solver(comm, cpu, mpiRank, N,maxite,relative, typeS)
  csr_ = matcsr_.indptr, matcsr_.indices, matcsr_.data #!!
  solver_.set_Mat_p(csr_)
  return solver_

#!/bin/env python
if __name__ == '__main__' :
#def run():
  from petsc4py import PETSc
  from mpi4py import MPI
  import petsc_solve
  import matrix
  import time as temps
  import py_functions as grid
  #import matplotlib.pyplot as plt
  
#parallel info
  comm = MPI.COMM_WORLD
  mpiRank = comm.Get_rank()
  cpu = comm.Get_size()

  from scipy.sparse import csr_matrix, identity
  import numpy as np
  import solid_FSI
  import cut_cells
  import math
  import config
  Uinf_n,rate_n=0.,0.
  fu,fv=config.get_gravity()

#parameters
  Ly,Lx,ny,nx,dy,dx,nu,rho,Ulid = config.sharedparameters()
  _Nit = config.get_nite()
  forwards = None
  #distribute local sphere numbers
  Nsphere = config.numb_sph()
  if mpiRank == 0:
    global_sphere_number = config.get_global_sphere_number(cpu)
    forwards = global_sphere_number
  local_sphere_numbers = comm.scatter(forwards, root=0)
  Nsphere_local = len(local_sphere_numbers)
  window = config.get_w()
  if mpiRank==0: 
    buffer_fl = np.zeros((window+1,window+1))
    u = np.zeros((ny+1,nx),dtype=np.float64)
    v = np.zeros((ny,nx+1),dtype=np.float64)
    p = np.zeros((ny+1,nx+1),dtype=np.float64)
    U = u.flat
    V = v.flat
    P = p.flat
    sourceu = np.zeros((ny+1,nx),dtype=np.float64)
    Su = sourceu.flat
    sourcev = np.zeros((ny,nx+1),dtype=np.float64)
    Sv = sourcev.flat
    conv_u1,conv_u2 = np.zeros((ny-1,nx),dtype=np.float64),np.zeros((ny-1,nx),dtype=np.float64)
    conv_v1,conv_v2 = np.zeros((ny,nx-1),dtype=np.float64),np.zeros((ny,nx-1),dtype=np.float64)
    gxu,gyu = np.zeros(((ny+1)-2,(nx+1)-1)), np.zeros(((ny+1)-1,(nx+1)-2))
    gxv,gyv = np.zeros(((ny+1)-2,(nx+1)-1)), np.zeros(((ny+1)-1,(nx+1)-2))
    gxp,gyp = np.zeros(((ny+1)-2,(nx+1)-1)), np.zeros(((ny+1)-1,(nx+1)-2))
    temp_ = np.zeros((ny+1,nx+1),dtype=np.float64)
  else:
    Su,Sv=None,None
  buffer_u = [None]*Nsphere_local
  buffer_v = [None]*Nsphere_local
  buffer_p = [None]*Nsphere_local
  buffer_pisou = [None]*Nsphere_local
  for n in range(Nsphere_local):
    buffer_u[n] = np.zeros((window,window+1))
    buffer_v[n] = np.zeros((window+1,window))
    #buffer_u[n] = np.zeros((window+1,window))
    #buffer_v[n] = np.zeros((window,window+1))
    buffer_p[n] = np.zeros((window,window))
    buffer_pisou[n] = np.zeros((window,window))
  if mpiRank == 0:
    Usphere = Ulid
    psi = np.zeros((ny+1,nx+1))
    Psi = psi.flat
    psisource = np.zeros((ny+1,nx+1))
    Psisource = psisource.flat
# u points
  xu = grid.Axe(nx,0.,dx)
  yu = grid.Axe(ny+1,dx/2.,dx)
# v points
  xv = grid.Axe(nx+1,dx/2.,dx)
  yv = grid.Axe(ny,0.,dx)
# p points
  xp = grid.Axe(nx+1,dx/2.,dx)
  yp = grid.Axe(ny+1,dx/2.,dx)
  config.set_flags(local_sphere_numbers)

 #SOLVER
  aperp = matrix.poisson_permanent() #!!
  bcslapl = matrix.BCSlaplacianPsi() #!!
  solver_p = create_solver(aperp,(ny+1)*(nx+1),comm, cpu, mpiRank,1e8,1e-6,"pressure")
  atu,aperu = matrix.Fbcu_permanent() #!!
  solver_u = create_solver(aperu,(ny+1)*(nx),comm, cpu, mpiRank,50000,1e-8,"velocity")
  atv,aperv = matrix.Fbcv_permanent() #!!
  solver_v = create_solver(aperv,(ny)*(nx+1),comm, cpu, mpiRank,50000,1e-8,"velocity")

  reboot = False
  if mpiRank == 0: 
    res,nsp = solveP_C()
    frate,history,global_sphere_number = res
  else: solveP_C()

