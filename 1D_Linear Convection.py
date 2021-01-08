import numpy                                                                #here we load numpy library that provides a bunch of useful matrix operations akin to MATLAB
from matplotlib import pyplot                                               #2D plotting library that we will use to plot our results



nx = 60                  #grip point
dx = 4 / (nx -1)         #distance between 2 nodes
nt = 90                  #number of timesteps
dt = 0.025               #amount of timestep each time the wave covers
c = 1                    #wavespeed 

#innitial condition

u = numpy.ones(nx)                #u = 1 across all grid points
u[int(0.5/dx):int(1/dx+1)] = 2    #u = 2 from x = 0.5 to 1, Square Profile
print(u)

pyplot.plot(numpy.linspace(0, 4, nx), u)

#calculation discritization of the convection equation

un = numpy.ones(nx)                                       #initialize a temporary array, un is solution for next time step

for n in range(nt):                                       #time marching
    un = u.copy()                                         
    for i in range(1, nx):                                #Space marching
        u[i] = un[i] - c*dt/dx*(un[i] - un[i - 1])        #Backward Differnece Scheme



print(u)
pyplot.plot(numpy.linspace(0, 4, nx), u)
pyplot.show()

