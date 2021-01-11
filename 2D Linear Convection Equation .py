from mpl_toolkits.mplot3d import Axes3D        #To plot a projected 3D result, make sure that you have added the Axes3D library

import numpy
from matplotlib import pyplot, cm
import time

lineSingle = '------------------------------------------------'

print("Solving 2D Linear Convection Equation using Finite Difference Method\n")

print(lineSingle)
print("METHOD - I")
print("Using Nested FOR Loop")
print(lineSingle)


#meshing
    
nx    = 81                  #grid points in x-Direction
ny    = 81                  #grid points in y-Direction

nt    = 80                  #number of time step
c     = 1                   #wave speed kept constant

#grid spacing

dx    = 2 / (nx - 1)
dy    = 2 / (ny - 1)

CFL = 0.2
dt  = CFL*dx              #timestep size

x = numpy.linspace(0, 2, nx)         #array along x
y = numpy.linspace(0, 2, ny)         #array along y

u  = numpy.ones((ny, nx))        
un = numpy.ones((ny, nx))            #2d temporaray array where we copy our velocity field

#innitial condition

u[int(0.5/dy):int(1/dy+1),int(0.5/dx):int(1/dx+1)] = 2       #Cuboidic Wave Profile

#plotting innitial condition

print(lineSingle)
print("Plotting Innitial Solution: Cuboidic Wave Profile")
print(lineSingle)

fig = pyplot.figure(figsize=(11, 7), dpi=100)       #innitilize plot window
ax  = fig.gca(projection = '3d')                    #defining axis is 3d
X,Y = numpy.meshgrid(x, y)                          #Generating 2D Mesh

#assign plot window the axes label ax, specifies its 3d projection plot
#plot_surface is regular plot command but it takes a grid of x and y for data point position

surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_title('Innitial Velocity Field')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()


start1 = time.time()                                     #calculating solving time
start1 = numpy.around(start1, decimals = 2)

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

for n in range(nt + 1):                     #time marching
    un = u.copy()    
    row, col = u.shape

    #space marching
    
    for j in range(1,row):  
        for i in range(1,col):

            #Backward Difference Scheme
            
            u[j,i] = (un[j,i] - (c * dt/dx * (un[j,i] - un[j,i-1])) - (c * dt/dx * (un[j,i] - un[j-1,i])))

            #Boundary Conditions, U = 1 for x = 0,2 & Y = 0,2

            u[0, :]  = 1
            u[-1, :] = 1
            u[:, 0]  = 1
            u[:, -1] = 1
            
print("Solving time with Nested FOR loop: %.4s seconds"% (time.time() - start1))

print(lineSingle)
print("Plotting Numerical Solution")
print(lineSingle)

fig   = pyplot.figure(figsize=(11,7), dpi=100)
ax    = fig.gca(projection = '3d')
surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_title('Method - I:Using Nested FOR Loop')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()

 

#using array operation

print(lineSingle)
print("METHOD - II")
print("Using ARRAYS Operation")
print(lineSingle)

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

start2 = time.time()
for n in range(nt + 1):
    un = u.copy()        #itterative process
    u[1:, 1:] = (un[1:,1:] - (c * dt / dx * (un[1:,1:] - un[1:,:-1])) - (c * dt / dy * (un[1:,1:] - un[:-1,1:])))     #Backward Difference Scheme

    u[0, :]  = 1
    u[-1, :] = 1
    u[:, 0]  = 1
    u[:, -1] = 1
    
print("Solving time with Array Operation: %.4s seconds"% (time.time() - start2))

print(lineSingle)
print("Plotting Numerical Solution")
print(lineSingle)

fig = pyplot.figure(figsize=(11,7), dpi = 100)
ax = fig.gca(projection = '3d')
surf3 = ax.plot_surface(X, Y, u[:], cmap = cm.viridis)
ax.set_title('Method - II: Using ARRAYS Operation')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()
