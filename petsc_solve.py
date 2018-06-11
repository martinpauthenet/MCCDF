from mpi4py import MPI as _MPI
from petsc4py import PETSc as _PETSc
import numpy as _np
from scipy.sparse import csr_matrix
import time as temps
class Solver:

    def __init__(self, comm, cpu, rank, dim, maxit,relative, solver):
        
        self.A_PETSc = _PETSc.Mat()
        self.A_PETSc_c = _PETSc.Mat()
        self.A_PETSc_p = _PETSc.Mat()
        self.method_PETSc = _PETSc.KSP()

        self.null_space = None
        self.comm = comm
        self.dim = dim
        self.cpu = cpu
        self.rank = rank

        csr = (_np.zeros(self.dim+1, dtype=_np.int32), _np.zeros(0, dtype=_np.int32), _np.zeros(0))
        self.A_PETSc.create(self.comm)
        self.A_PETSc.setSizes([self.dim,self.dim])
        self.A_PETSc.setType('mpiaij')
        self.A_PETSc.setPreallocationCSR(csr)

        self.A_PETSc_p.create(self.comm)
        self.A_PETSc_p.setSizes([self.dim,self.dim])
        self.A_PETSc_p.setType('mpiaij')
        self.A_PETSc_p.setPreallocationCSR(csr)

        self.A_PETSc_c.create(self.comm)
        self.A_PETSc_c.setSizes([self.dim,self.dim])
        self.A_PETSc_c.setType('mpiaij')
        self.A_PETSc_c.setPreallocationCSR(csr)

        self.x_PETSc, self.b_PETSc = self.A_PETSc_p.getVecs()
        length_x = self.b_PETSc.getOwnershipRange()[1] - self.b_PETSc.getOwnershipRange()[0]

        self.method_PETSc.create(self.comm)

        if solver == "pressure" :
          self.method_PETSc.setType('gmres')
          #self.method_PETSc.setType('bicg')
          #self.method_PETSc.setType('bcgs')
          #self.method_PETSc.setType('cg') #JADIM
          #self.method_PETSc.setType('preonly')
          pc = _PETSc.PC()
          pc.create(self.comm)
          pc.setType('asm')#additive schwarz
          #pc.setType('pbjacobi') #JADIM
          #pc.setType('sor')
          #pc.setASMType(2)

        #  pc.setType('ilu')
          self.method_PETSc.setPC(pc)
          #print self.method_PETSc.getPCSide()
        if solver == "velocity" :
          self.method_PETSc.setType('bcgs')
          #self.method_PETSc.setType('gmres')
          #self.method_PETSc.setType('cg') #JADIM
          pc = _PETSc.PC()
          pc.create(self.comm)
          pc.setType('asm')#additive schwarz
          #pc.setType('pbjacobi') #JADIM
          #pc.setType('sor')
          self.method_PETSc.setPC(pc)

        pc = self.method_PETSc.getPC()
        if self.rank==0: print solver, self.method_PETSc.getType(), pc.getType(), relative, maxit

        #self.method_PETSc.setInitialGuessNonzero(1)
        self.method_PETSc.setInitialGuessNonzero(0)
        self.method_PETSc.setTolerances(rtol=relative, atol=1e-50, divtol=1e5, max_it=maxit)

    def set_Mat_p(self,csrMat_p):

        if self.rank == 0:  
         forwards = [None] *self.cpu
         Iminmax = self.A_PETSc_p.getOwnershipRanges()
         nf = 0
         for nf in range(self.cpu):
          Imin, Imax = Iminmax[nf], Iminmax[nf+1]
          forwards[nf] =  (csrMat_p[0][Imin:Imax+1]-csrMat_p[0][Imin],\
                           csrMat_p[1][csrMat_p[0][Imin] : csrMat_p[0][Imax]],\
                           csrMat_p[2][csrMat_p[0][Imin] : csrMat_p[0][Imax]])
        else: forwards = None
        #forward matrix csr data
        rows, cols, values = self.comm.scatter(forwards, root=0)
        self.A_PETSc_p.createAIJ(size=(self.dim,self.dim), csr=(rows, cols, values), comm=self.comm)
        
    def set_Mat_correctiveMethod(self, csrMat_c, nullspace):
############## matrix
        if self.rank == 0:  
         forwards = [None] *self.cpu
         Iminmax = self.A_PETSc_c.getOwnershipRanges()
         nf = 0
         for nf in range(self.cpu):
          Imin, Imax = Iminmax[nf], Iminmax[nf+1]
          forwards[nf] =  (csrMat_c[0][Imin:Imax+1]-csrMat_c[0][Imin],\
                           csrMat_c[1][csrMat_c[0][Imin] : csrMat_c[0][Imax]],\
                           csrMat_c[2][csrMat_c[0][Imin] : csrMat_c[0][Imax]])
        else: forwards = None
        #forward matrix csr data
        rows, cols, values = self.comm.scatter(forwards, root=0)
        self.A_PETSc_c.createAIJ(size=(self.dim,self.dim), csr=(rows, cols, values), comm=self.comm)   
        self.A_PETSc = (self.A_PETSc_p+self.A_PETSc_c)

        if nullspace:
          nsp = _PETSc.NullSpace()	
          nsp.create(constant=True, vectors=(), comm=self.comm)
          self.A_PETSc.setNullSpace(nsp)
        self.method_PETSc.setOperators(self.A_PETSc)
        #self.A_PETSc.assemble()        
        #CSR = self.A_PETSc.getValuesCSR()
        #print CSR[2].shape

    def set_Mat_directMethod(self, csrMat_d, nullspace):
      if self.rank == 0:
        forwards = [None] *self.cpu
        Iminmax = self.A_PETSc_c.getOwnershipRanges()
        for nf in range(self.cpu):
          Imin, Imax = Iminmax[nf], Iminmax[nf+1]
          forwards[nf] =  (csrMat_d[0][Imin:Imax+1]-csrMat_d[0][Imin],\
                           csrMat_d[1][csrMat_d[0][Imin] : csrMat_d[0][Imax]],\
                           csrMat_d[2][csrMat_d[0][Imin] : csrMat_d[0][Imax]])
      else: forwards = None
        #forward matrix csr data
      rows, cols, values = self.comm.scatter(forwards, root=0)
      self.A_PETSc_c.createAIJ(size=(self.dim,self.dim), csr=(rows, cols, values), comm=self.comm)
      self.A_PETSc = self.A_PETSc_c

      if nullspace:
        self.null_space = _PETSc.NullSpace()
        self.null_space.create(constant=True, vectors=(), comm=self.comm)
        self.A_PETSc.setNullSpace(self.null_space)
      self.method_PETSc.setOperators(self.A_PETSc)

      #if self.rank == 0:
      #  csr = self.A_PETSc.getValuesCSR()
        #print csr[0].shape,csr[1].shape,csr[2].shape
      #  array = csr_matrix((csr[2], csr[1], csr[0]), (self.dim, self.dim)).todense()
        #_np.savetxt('a',array)
      #  import matplotlib.pyplot as plt
      #  im = plt.imshow(array,interpolation='nearest')
      #  plt.colorbar(im)
      #  plt.show()
        #for i in range(array.shape[0]):
        #  print array[i,:]
      #if self.rank == 0: print data

    def set_Vectors(self, numpRHS):
############## RHS

        if self.rank == 0:
         #split rhs
         #forwards = [None] *self.cpu
         coupes = self.b_PETSc.getOwnershipRanges()
         for nf in range(self.cpu): #forwards[nf] = numpRHS[coupes[nf]:coupes[nf+1]]
          if nf == 0: self.b_PETSc.array[:] = numpRHS[coupes[nf]:coupes[nf+1]]
          else: self.comm.Send([numpRHS[coupes[nf]:coupes[nf+1]], _MPI.DOUBLE], dest=nf, tag=77)
        #forward RHS data
        else: self.comm.Recv([self.b_PETSc.array, _MPI.DOUBLE], source=0, tag=77)
         #self.b_PETSc.array[:] = self.comm.scatter(forwards, root=0)

    def solveLinear(self):

        self.method_PETSc.solve(self.b_PETSc,self.x_PETSc)
        convergedReason = self.method_PETSc.getConvergedReason()

        if convergedReason<0:
          if self.rank==0: print "linear solver has not converged for reason", convergedReason

          opt_init = self.method_PETSc.getPC().getType()

          self.method_PETSc.setInitialGuessNonzero(0)
          if convergedReason<0:
#            if self.rank==0: print "linear solver has not converged for reason", convergedReason
            pc = _PETSc.PC()
            pc.create(self.comm)
            pc.setType('sor')#successive over relaxation
            self.method_PETSc.setPC(pc)
            self.method_PETSc.setOperators(self.A_PETSc)
            if self.rank==0: print "change preconditionner to", self.method_PETSc.getPC().getType()

            self.method_PETSc.solve(self.b_PETSc,self.x_PETSc)
            convergedReason = self.method_PETSc.getConvergedReason()

          self.method_PETSc.setInitialGuessNonzero(1)
          if convergedReason<0:
            if self.rank==0: print "linear solver has not converged for reason", convergedReason
            pc.setType('asm')#successive over relax
            self.method_PETSc.setPC(pc)
            self.method_PETSc.setOperators(self.A_PETSc)
            if self.rank==0: print "change preconditionner to", self.method_PETSc.getPC().getType()

            self.method_PETSc.solve(self.b_PETSc,self.x_PETSc)
            convergedReason = self.method_PETSc.getConvergedReason()

          if convergedReason<0:
            if self.rank==0: print "linear solver has not converged for reason", convergedReason
          else:            
            if self.rank==0: print "linear solver has finally converged"

          self.method_PETSc.setInitialGuessNonzero(0)
          #if self.rank==0: print "setInitialGuess back to non zero"
          pc.setType(opt_init)
          self.method_PETSc.setPC(pc)
          self.method_PETSc.setOperators(self.A_PETSc)
          if self.rank==0: print "change preconditionner back to", self.method_PETSc.getPC().getType()

        return

    def get_result(self):
        result = self.comm.gather(self.x_PETSc.array, root=0)
        if self.rank == 0: return _np.concatenate(result)
        return

