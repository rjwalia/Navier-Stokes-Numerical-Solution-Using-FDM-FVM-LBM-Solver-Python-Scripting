import numpy                            #here we load numpy library that provides a bunch of useful matrix operations akin to MATLAB
from matplotlib import pyplot           #2D plotting library that we will use to plot our results

nx = 800                    #grip points
dx = 2 / (nx -1)            #grid spacing
nt = 220                    #number of timesteps
dt = 0.0025                 #timestep size
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

pyplot.plot(numpy.linspace(0, 2, nx), u, label='Initial Solution');

#discritization

un = numpy.ones(nx)                                       

for n in range(nt):                                       #time marching
    un = u.copy()                                         
    for i in range(1, nx):                                #Space marching
        u[i] = un[i] - c*dt/dx*(un[i] - un[i - 1])        #Backward Differnece Scheme


pyplot.plot(numpy.linspace(0, 2, nx), u, label='Convected Solution');
pyplot.title('1D Linear Convecction')
pyplot.xlabel('Grid Space')
pyplot.ylabel('Velocity')

print(u)
pyplot.legend()
pyplot.show()

