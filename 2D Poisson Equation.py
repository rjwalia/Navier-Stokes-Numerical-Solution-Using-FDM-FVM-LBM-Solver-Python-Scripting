import numpy                                #give mathamatical or matrix expressions like array
from matplotlib import pyplot, cm           #plotting library that we will use to plot our results
from mpl_toolkits.mplot3d import Axes3D     #To plot a projected 3D result, make sure that you have added the Axes3D library

lineSingle = '------------------------------------------------'

print("Solving Poisson Equation for Pressure using Finite Difference Method\n")

#meshing

nx = 60            #Grid Points along X direction
ny = 60            #Grid Points along Y direction

iteration = input('Enter the number of Iterations to Solve: ')          

if iteration.isdigit() == False:
    print("Please provide an integer\n")
else:
    iteration = int(iteration)

#Grid Spacing

xmin = 0
xmax = 2
ymin = 0
ymax = 1

dx = (xmax - xmin) / (nx - 1)         
dy = (ymax - ymin) / (ny - 1)

#initilization

p = numpy.zeros((ny,nx))
pd = numpy.zeros((ny,nx))
b = numpy.zeros((ny,nx))              
x = numpy.linspace(xmin,xmax,nx)
y = numpy.linspace(ymin,ymax,ny)

#sourceterm on RHS of Poisson Equation

b[int(ny/4),int(nx/4)] = 100
b[int(3*ny/4),int(3*nx/4)] = -100

#Defining a Function for plotting initial & steady state solution

def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection = '3d')

    #Generating 2D Mesh    
    X, Y = numpy.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, b, rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Initial Solution')
    ax.set_xlabel('X Spacing')
    ax.set_ylabel('Y Spacing')
    ax.set_zlabel('Velocity')
    
print(lineSingle)
print("Plotting Innitial Solution")
print(lineSingle)

plot2D(x,y,p)
pyplot.show()

#Solving the Poisson Equation

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

for it in range(iteration):
    
    pd = p.copy()

    #Central Difference Scheme

    p[1:-1,1:-1] = (((pd[1:-1,2:] + pd[1:-1,:-2])*dy**2 + (pd[2:,1:-1] + pd[:-2,1:-1])*dx**2
                     - b[1:-1,1:-1]*dx**2 * dy**2) / (2*(dx**2 + dy**2)))

    #Boundary Condition
    
    p[0,:] = 0
    p[ny-1,:] = 0
    p[:,0] = 0
    p[:,nx-1] = 0
    
print(lineSingle)
print("Iterations Completed!")
print(lineSingle)

print(lineSingle)
print("Plotting Solution")
print(lineSingle)

def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection = '3d')
    
    X, Y = numpy.meshgrid(x, y)           #Generating 2D Mesh
    
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_title('Steady State Solution')
    ax.set_xlabel('X Spacing')
    ax.set_ylabel('Y Spacing')
    ax.set_zlabel('Velocity')

plot2D(x,y,p)
pyplot.show()
