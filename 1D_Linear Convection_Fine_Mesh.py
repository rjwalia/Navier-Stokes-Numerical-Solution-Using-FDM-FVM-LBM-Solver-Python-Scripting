import numpy                            #here we load numpy library that provides a bunch of useful matrix operations akin to MATLAB
from matplotlib import pyplot           #2D plotting library that we will use to plot our results

nx = 800                    #grip point
dx = 2 / (nx -1)            #distance between 2 nodes
nt = 220                    #number of timesteps
dt = 0.0025                 #amount of timestep each time the wave covers
c = 1                       #wavespeed

#innitial condition

x = numpy.linspace(0, 2, nx)
u = numpy.ones(nx)                        #u = 1 across all grid points
for j in range(nx):
    if 0.5 <= x[j] and x[j] <= 1:
        u[j] = 2                          #u = 2 from x = 0.5 to 1, Square Profile
    else:
        u[j] = 1
print(u)

pyplot.plot(numpy.linspace(0, 2, nx), u);

#calculation discritization of the convection equation

un = numpy.ones(nx)                                       #initialize a temporary array, un is solution for next time step

for n in range(nt):                                       #time marching
    un = u.copy()                                         
    for i in range(1, nx):                                #Space marching
        u[i] = un[i] - c*dt/dx*(un[i] - un[i - 1])        #Backward Differnece Scheme


pyplot.plot(numpy.linspace(0, 2, nx), u);
print(u)
pyplot.show()

