# Importing FEniCS and useful libraries

from fenics import * # I used 2018 version
from mshr import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Creating the mesh

l, h = 5., 1. # lenght, height
r = 0.25 # radious
N = 200 # mesh density

S0 = Rectangle(Point(0, 0), Point(l, h))
C0 = Circle(Point(4, 0.5), r)
domain = S0 - C0 

mesh = generate_mesh(domain, N)

plt.figure(figsize = (16,9), dpi=300)
plot(mesh, linewidth = 0.2)

# Defining the finite element function space

V = VectorFunctionSpace(mesh, 'P', degree = 1)

# Defining the boundary conditions

def left(x, on_boundary):
    return near(x[0], 0) and on_boundary

def right(x, on_boundary):
    return near(x[0], 5) and on_boundary

u_L = Constant((0,0)) # Left
u_R = Constant((0.001, 0)) # Right 

bc_L = DirichletBC(V, u_L, left)
bc_R = DirichletBC(V, u_R, right)

bcs = [bc_L, bc_R]

# Defining the variational problem

E = Constant(206180) # Young's Modulus [MPa]
nu = Constant(0.3) # Poisson coefficient
model = 'plane_stress'
d = 2 # Dimension

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

def eps(v):
    return sym(grad(v))

if model == 'plane_stress':
    lmbda = 2*mu*lmbda/(lmbda+2*mu)
    
def sigma(v):
    return lmbda*tr(eps(v))*Identity(d) + 2.0*mu*eps(v)
    
f = Constant((0, 0)) # Source term
u = TrialFunction(V)
v = TestFunction(V)

# We define the bilinear and linear form
a = inner(sigma(u), eps(v))*dx
L = inner(f, v)*dx

# Solving the variational problem

u = Function(V, name = 'Displacement')
solve(a == L, u, bcs)

# Computing the Von Mises Stress

s = dev(sigma(u)) # Deviatoric stress
sv = sqrt(1.5*inner(s, s)) # Von Mises Stress

# Computing the Average VMS
sv_avg = assemble(sv*dx) / assemble(1*dx(mesh))
print('Average Von Mises Stress = %.2f' % (sv_avg), 'MPa')

# Computing the Max. VMS
W = FunctionSpace(mesh, 'DG', 0)
sv = project(sv, W)
sv_max = sv.vector().norm('linf')
print('Max. Von Mises Stress = %.2f' % (sv_max), 'MPa')

# Compute the maximum strain
max_strain = (np.abs(u.vector().get_local()).max() / 5)*100
print('Max. Strain = %.2f' % (max_strain), '%')

# Saving files to do post-prosessing in ParaView

File('traction/displacement.pvd') << u
File('traction/von_mises.pvd') << sv

# Making some basics plots

plt.figure(figsize = (16,9), dpi=300)

plt.subplot(211)
p0 = plot((u/l)*100, cmap = 'turbo', mode='displacement') # Sin " mode = 'displacement' " se ven los vectores.
plt.colorbar(p0, label = 'Engineering Strain [%]', orientation='horizontal')
plot(mesh, linewidth = 0.2)
plt.axis('off')

plt.subplot(212)
p1 = plot(sv, cmap = 'turbo')
plt.colorbar(p1, label = 'Von Mises Stress [MPa]', orientation='horizontal')
plt.axis('off')

filename = 'traction/fig1.png'
plt.savefig(filename, dpi=300)

plt.figure(figsize = (16,9), dpi=300)

plt.subplot(321)
p2 = plot((eps(u)[0,0]), cmap = 'turbo', mode = 'color')
plt.colorbar(p2, label = r'$\varepsilon_{xx}$', orientation='horizontal',  format='%.0e')
plt.axis('off')

plt.subplot(322)
p3 = plot(sigma(u)[0,0], cmap = 'turbo', mode = 'color')
plt.colorbar(p3, label = r'$\sigma_{xx}$ [MPa]', orientation='horizontal')
plt.axis('off')

plt.subplot(323)
p4 = plot((eps(u)[1,1]), cmap = 'turbo', mode = 'color')
plt.colorbar(p4, label = r'$\varepsilon_{yy}$', orientation='horizontal', format='%.0e')
plt.axis('off')

plt.subplot(324)
p5 = plot(sigma(u)[1,1], cmap = 'turbo', mode = 'color')
plt.colorbar(p5, label = r'$\sigma_{yy}$ [MPa]', orientation='horizontal')
plt.axis('off')

plt.subplot(325)
p6 = plot((eps(u)[0,1]), cmap = 'turbo', mode = 'color')
plt.colorbar(p6, label = r'$\varepsilon_{xy}$', orientation='horizontal', format='%.0e')
plt.axis('off')

plt.subplot(326)
p7 = plot(sigma(u)[0,1], cmap = 'turbo', mode = 'color')
plt.colorbar(p7, label = r'$\sigma_{xy}$ [MPa]', orientation='horizontal')
plt.axis('off')

filename = 'traction/fig2.png'
plt.savefig(filename, dpi=300)

