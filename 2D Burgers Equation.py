import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

lineSingle = '------------------------------------------------'

print("Solving 2D Burgers Equation using Finite Difference Method")
print("Convection Term: Backward Difference Scheme")
print("Diffusion Term: Central Difference Scheme\n")

nx = 41          #Grid Points Along X direction
ny = 41          #Grid Points Along Y direction

nt = 100          #Number of Time Steps

#Grid Spacing

dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

cfl = 0.009

nu = 0.01                #viscosity

dt = cfl*dx*dy/nu         #timestep size



x = numpy.linspace(0,2,nx)     #Coordinate Along X direction
y = numpy.linspace(0,2,ny)     #Coordinate Along Y direction

#2d temporaray array where we copy our velocity field

u = numpy.ones((ny,nx))
v = numpy.ones((ny, nx))
un = numpy.ones((ny,nx))
vn = numpy.ones((ny,nx))
comb = numpy.ones((ny,nx))

#innitial condition
#Cuboidic Wave Profile

u[int(0.5/dy):int(1/dy+1),int(0.5/dx):int(1/dx+1)] = 2
v[int(0.5/dy):int(1/dy+1),int(0.5/dx):int(1/dx+1)] = 2

print(lineSingle)
print("Plotting Innitial Solution: Cuboidic Wave Profile")
print(lineSingle)

fig = pyplot.figure(figsize=(11,7),dpi=100)          #Initializing the figure
ax  = fig.gca(projection='3d')
X,Y = numpy.meshgrid(x,y)

ax.plot_surface(X,Y,u[:],cmap=cm.viridis,rstride=1,cstride=1)
ax.plot_surface(X,Y,v[:],cmap=cm.viridis,rstride=1,cstride=1)

ax.set_title('Initial Velocity Field')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

for n in range(nt+1):          #time marching
    un = u.copy()
    vn = v.copy()

    #Backward Difference Scheme for Convection Term
    #Central Difference Scheme for Diffusion Term

    u[1:-1,1:-1] = (un[1:-1,1:-1] - dt/dx*un[1:-1,1:-1]*(un[1:-1,1:-1]-un[1:-1,0:-2])-
                    dt/dy*vn[1:-1,1:-1]*(un[1:-1,1:-1] - un[0:-2,1:-1])+
                    nu*dt/dx**2 * (un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])+
                    nu*dt/dy**2 * (un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1]))

    v[1:-1,1:-1] =(vn[1:-1,1:-1] - dt/dx*un[1:-1,1:-1]*(vn[1:-1,1:-1]-vn[1:-1,0:-2])-
                    dt/dy*vn[1:-1,1:-1]*(vn[1:-1,1:-1] - vn[0:-2,1:-1])+
                    nu*dt/dx**2 * (vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])+
                    nu*dt/dy**2 * (vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1]))
    
    #boundary condition
    
    u[0,:]  = 1
    u[-1,:] = 1
    u[:,0]  = 1
    u[:,-1] = 1

    v[0,:]  = 1
    v[-1,:] = 1
    v[:,0]  = 1
    v[:,-1] = 1
    
print(lineSingle)
print("Plotting Numerical Solution")
print(lineSingle)

fig = pyplot.figure(figsize=(11,7),dpi=100)
ax  = fig.gca(projection='3d')

X,Y = numpy.meshgrid(x,y)
ax.plot_surface(X,Y,u[:],cmap=cm.viridis,rstride=1,cstride=1)
ax.plot_surface(X,Y,v[:],cmap=cm.viridis,rstride=1,cstride=1)
ax.set_title('Final Velocity Field')
ax.set_xlabel('X Spacing')
ax.set_ylabel('Y Spacing')
ax.set_zlabel('Velocity')
pyplot.show()
