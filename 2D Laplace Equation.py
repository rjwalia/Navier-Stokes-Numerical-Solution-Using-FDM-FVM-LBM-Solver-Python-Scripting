import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

lineSingle = '------------------------------------------------'

print("Solving Laplace Equation using Finite Difference Method\n")

#Function for plotting initial & steady state solution

def plot2D(x, y, p):
    
    fig = pyplot.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(projection = '3d')
    
    X,Y=numpy.meshgrid(x,y)      #Generating 2D Mesh
    
    surf = ax.plot_surface(X,Y,p[:],rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xlabel('X Spacing')
    ax.set_ylabel('Y Spacing')
    ax.set_zlabel('Velocity')
    ax.view_init(30, 225)
    

#Function for solving Laplace Equation
    
def laplace2d(p, y, dx, dy, residual_target):
    
    residual = 1                           #innitial error
    pm = numpy.empty_like(p)
    
    iteration = 0
    while residual > residual_target:         #Convergence Criteria
        pn = p.copy()
        p[1:-1,1:-1] = ((dy**2*(pn[2:,1:-1]+pn[0:-2,1:-1]) +
                         dx**2*(pn[1:-1,2:]+pn[1:-1,0:-2]))/
                        (2*(dx**2 + dy**2)))

        #Boundary Condition

        p[:,0]  = 0         # p = 0 @ x = 0
        p[:,-1] = y         # p = y @ x = 2
        p[0,:]  = p[1,:]    # dp/dy = 0  @ y = 0
        p[-1,:] = p[-2,:]   # dp/dy = 0  @ y = 1
        
        residual = (numpy.sum(numpy.abs(p[:]) - numpy.abs(pn[:]))/
                  numpy.sum(numpy.abs(pn[:])))
        
        iteration += 1
        
    print('number of iteration :', iteration)
    return p
    

nx = 31           #Grid Point Along X
ny = 31           #Grid Point Along Y

#Grid Spacing

dx = 2 / (nx-1)
dy = 2 / (ny-1)

#initial condition
p = numpy.zeros((ny,nx))   # innitial guess, pressure everywhere is 0



#array
x = numpy.linspace(0,2,nx)
y = numpy.linspace(0,1,ny)

#Innitial Condition

p[:,0]  = 0         # p = 0 @ x = 0
p[:,-1] = y         # p = y @ x = 2
p[0,:]  = p[1,:]    # dp/dy = 0  @ y = 0
p[-1,:] = p[-2,:]   # dp/dy = 0  @ y = 1

#Calling function to plot initial state solution

plot2D(x,y,p)

print(lineSingle)
print("Plotting Innitial Solution")
print(lineSingle)


pyplot.show()

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)
print(lineSingle)
print("Solution Converged!")
print(lineSingle)

p = laplace2d(p, y, dx, dy, 1e-5)          #Calling function to start the simulation

print(lineSingle)
print("Plotting Numerical Solution")
print(lineSingle)

#Calling function to plot stead state solution

plot2D(x,y,p)
pyplot.show()

        
                          
    
