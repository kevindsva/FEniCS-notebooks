# Importing FEniCS and useful libraries

from fenics import *
from mshr import *
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Defining the time steps

T = 3.0              # final time
num_steps = 300      # number of time steps
dt = T / num_steps   # time step size

# Defining the mesh

S0 = Rectangle(Point(0, 0), Point(1, 1))
C0 = Circle(Point(0.5, 0.5), 0.25)
domain = S0  - C0 

mesh = generate_mesh(domain, 32)
plot(mesh)

# Defining the finite element function space

V = FunctionSpace(mesh, 'P', 1)

# Defining the boundary conditions

# Circle
u_D = 100.0

def boundary(x, on_boundary):
    d0 = sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2)
    return(on_boundary and d0 < 0.3)

bc = DirichletBC(V, u_D, boundary)

# Initial value

u_0 = Expression('1000', degree=1)
u_n = interpolate(u_0, V)

# Defining Trial and Test functions

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0) # Source term

# Bilinear and linear forms

#Constant thermal diffusivity
k = Constant(0.1)

# We define the variational problem with the form: F(u,v) = 0
F = u*v*dx + dt*k*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx

# We let to FEniCS to choose the terms for the bilinear and linear form
# by using: "lhs=left hand side" and "rhs=right hand side"
a, L = lhs(F), rhs(F) 

# Post-processing

vtkfile = File('Trascient heat equation in plate with a hole/solution.pvd')

# Solving the variational problem

# Time-stepping
u = Function(V)
t = 0

for n in range(num_steps):
    # Update current time
    t += dt
    
    # Compute solution
    solve(a == L, u, bc)
    
    # Save to file and plot solution
    vtkfile << (u, t)

    # Compute the maximum temperature at t = i
    max_temp = np.abs(u.vector().get_local()).max()
    print('t = %.2f: Max. temperature = %.3g' % (t, max_temp))
    
    # Update previous solution
    u_n.assign(u)
    
# Making a basic plot of the solution at t=3

plt.figure(figsize = (16, 9), dpi=75)
p = plot(u, cmap = 'inferno')
plt.title('Temperature distribution at t = %g'%t, fontsize=16)
plt.colorbar(p, label = 'Temperature (°C)')
plot(mesh, linewidth = 0.4)

filename = 'Trascient heat equation in plate with a hole/trascient-heat-equation.png'
plt.savefig(filename, dpi=300)
