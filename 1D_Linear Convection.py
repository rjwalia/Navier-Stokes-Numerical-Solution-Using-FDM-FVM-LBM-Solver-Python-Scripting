import numpy                                                                #here we load numpy library that provides a bunch of useful matrix operations akin to MATLAB
from matplotlib import pyplot                                               #2D plotting library that we will use to plot our results

lineSingle = '------------------------------------------------'

print("1D Linear Equation\n")

nx = 60                  #grip points
dx = 4 / (nx -1)         #grid spacing 
nt = 90                  #number of timesteps
dt = 0.025               #timestep size
c = 1                    #wavespeed 

#innitial condition

#innitial condition

print(lineSingle)
print("Computing Innitial Solution...")

u = numpy.ones(nx)                #u = 1 across all grid points
u[int(0.5/dx):int(1/dx+1)] = 2    #u = 2 from x = 0.5 to 1, Square Profile

print("Printing Innitial Solution...")
print(lineSingle)

print(u)

pyplot.plot(numpy.linspace(0, 4, nx), u, label='Initial Solution')

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

#discritization

un = numpy.ones(nx)                                       #initialize a temporary array, un is solution for next time step

for n in range(nt):                                       #time marching
    un = u.copy()                                         
    for i in range(1, nx):                                #Space marching
        u[i] = un[i] - c*dt/dx*(un[i] - un[i - 1])        #Backward Differnece Scheme

print(lineSingle)
print("Printing Numerical Solution......")
print(lineSingle)
        
print(u)

print(lineSingle)
print("Plotting Innitial & Numerical Solution")
print(lineSingle)

pyplot.plot(numpy.linspace(0, 4, nx), u, label='Convected Solution')
pyplot.title('1D Linear Convecction')
pyplot.xlabel('Grid Space')
pyplot.ylabel('Velocity')

pyplot.legend()
pyplot.show()

