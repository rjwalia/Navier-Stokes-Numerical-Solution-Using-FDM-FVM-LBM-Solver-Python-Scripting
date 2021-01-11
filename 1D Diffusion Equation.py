import numpy                    #here we load numpy library that provides a bunch of useful matrix operations akin to MATLAB
from matplotlib import pyplot   #2D plotting library that we will use to plot our results

lineSingle = '------------------------------------------------'

print("Solving 1D Diffusion Equation using Finite Difference Method\n")

#setting the grid

nx = 41                     #grid points
dx  = 2 / (nx - 1)          #grid spacing   
nt = 20                     #number of timesteps
nu = 0.3                    #viscosity
cfl = 0.4                   
dt = cfl*dx**2/nu           #based on von neumaan stability analysis

#innitial condition

print(lineSingle)
print("Computing Innitial Solution...")


u = numpy.ones(nx)                  
u[int(0.5/dx):int(1/dx+1)] = 2      #Square Wave Profile

print("Printing Innitial Solution...")
print(lineSingle)

print(u)

pyplot.plot(numpy.linspace(0,2,nx), u, label='Initial Solution')

#discritization

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

un = numpy.ones(nx)  

for n in range(nt+1):               #time marching
    un = u.copy()       
    for i in range(1, nx-1):        #Space marching
        
        u[i] = un[i] + nu * dt/dx**2 *(un[i+1] - 2*un[i] + un[i-1]) #Central Differnece Scheme

print(lineSingle)
print("Printing Numerical Solution......")
print(lineSingle)
        
print(u)

print(lineSingle)
print("Plotting Innitial & Numerical Solution")
print(lineSingle)

pyplot.plot(numpy.linspace(0,2,nx), u, label='Numerical Solution')
pyplot.title('1D Diffusion Convecction')
pyplot.xlabel('Grid Space')
pyplot.ylabel('Velocity')

pyplot.legend()
pyplot.show()
