import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

lineSingle = '------------------------------------------------'

print("Solving 2D Diffusion Convection Equation using Finite Difference Method\n")

nx = 31                  #grid points in x-Direction
ny = 31                  #grid points in y-Direction
nu = 0.05                #viscosity

#grid spacing

dx = 2/(nx-1)
dy = 2/(ny-1)

cfl = 0.25
dt =  cfl*dx*dy/nu       #time step based on von neumann stability analysis   
nt = 10                  #number of time step

x = numpy.linspace(0, 2, nx)       #array along x
y = numpy.linspace(0, 2, ny)       #array along y

#2d temporaray array where we copy our velocity field

u = numpy.ones((ny,nx))
un = numpy.ones((ny,nx))

#innitial condition
#Cuboidic Wave Profile

u[int(0.5/dy):int(1/dy + 1),int(0.5/dx):int(1/dx+1)] = 2

#plotting innitial condition

print(lineSingle)
print("Plotting Innitial Solution: Cuboidic Wave Profile")
print(lineSingle)


fig = pyplot.figure()
ax  = fig.add_subplot(projection = '3d')
X,Y = numpy.meshgrid(x, y)
surf = ax.plot_surface(X,Y,u, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_title('Initial Velocity Field')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')


ax.set_xlim(0,2)
ax.set_ylim(0,2)
ax.set_zlim(1,2.5)

pyplot.show()

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

for n in range(nt+1):
        un = u.copy()

        #Central Difference Scheme
        
        u[1:-1,1:-1] = (un[1:-1,1:-1] + (nu*dt/dx**2)*(un[2:,1:-1] -2*un[1:-1,1:-1] + un[0:-2,1:-1]) + (nu*dt/dy**2)*(un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]))

        #Boundary Condition
        
        u[0,:]  = 1
        u[-1,:] = 1
        u[:,0]  = 1
        u[:,-1] = 1

print(lineSingle)
print("Plotting Numerical Solution")
print(lineSingle)

fig = pyplot.figure()
ax  = fig.add_subplot(projection = '3d')
surf = ax.plot_surface(X,Y,u[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
ax.set_zlim(1, 2.5)

ax.set_title('Final Velocity Field')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()
