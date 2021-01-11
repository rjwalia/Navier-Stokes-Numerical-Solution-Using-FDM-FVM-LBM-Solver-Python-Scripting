import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

lineSingle = '------------------------------------------------'

print("Solving Channel Flow in 2D using Finite Difference Method\n")

print("Unsteady Term  : Forward Euler Scheme")
print("Convection Term: Backward Difference Scheme")
print("Diffusion Term : Central Difference Scheme\n")

print("Channel Flow Velocity: 1 m/s")

#Solving RHS of poisson eqution in a separate function

def build_up_b(rho, dt, dx, dy, u, v):
    b = numpy.zeros_like(u)

    b[1:-1,1:-1] = (rho*(1/dt*((u[2:,1:-1]-u[:-2,1:-1])/(2*dx) + (v[1:-1,2:]-v[1:-1,:-2])/(2*dy))
                          - ((u[2:,1:-1]-u[:-2,1:-1])/(2*dx))**2 -
                          2*((u[1:-1,2:]-u[1:-1,:-2])/(2*dy) * (v[2:,1:-1]-v[:-2,1:-1])/(2*dx)) -
                           ((v[1:-1,2:] - v[1:-1,:-2])/(2*dy))**2))

    #periodic BC Pressure @ x = 2

    b[-1,1:-1] = (rho*(1/dt*((u[0,1:-1]-u[-2,1:-1])/(2*dx) + (v[-1,2:]-v[-1,:-2])/(2*dy))
                          - ((u[0,1:-1]-u[-2,1:-1])/(2*dx))**2 -
                          2*((u[-1,2:]-u[-1,:-2])/(2*dy) * (v[0,1:-1]-v[-2,1:-1])/(2*dx)) -
                           ((v[-1,2:] - v[-1,:-2])/(2*dy))**2))
      


    #periodic BC Pressure @ x = 0

    b[0,1:-1] = (rho*(1/dt*((u[1,1:-1]-u[-1,1:-1])/(2*dx) + (v[0,2:]-v[0,:-2])/(2*dy))
                          - ((u[1,1:-1]-u[-1,1:-1])/(2*dx))**2 -
                          2*((u[0,2:]-u[0,:-2])/(2*dy) * (v[1,1:-1]-v[-1,1:-1])/(2*dx)) -
                           ((v[0,2:] - v[0,:-2])/(2*dy))**2))

    return b

#Solving Poisson Equation for Pressure 

def pressure_poisson_periodic(p, dx, dy):
    pn = numpy.empty_like(p)

    for q in range(iteration):
        pn = p.copy()

        p[1:-1,1:-1] = (((pn[2:,1:-1] + pn[:-2,1:-1])*(dy**2) 
                         + (pn[1:-1,2:]+p[1:-1,:-2])*(dx**2)) /(2*(dx**2 + dy**2)) 
                         - dx**2 * dy**2/(2 * (dx**2 + dy**2)) * b[1:-1,1:-1])
                
        

     #periodic BC Pressure @ x = 2

        p[-1,1:-1] = (((pn[0,1:-1] + pn[-2,1:-1])*(dy**2) + (pn[-1,2:]+p[-1,:-2])*(dx**2)) /
                      (2*(dx**2 + dy**2)) - dx**2 * dy**2/(2 * (dx**2 + dy**2)) * b[-1,1:-1])

    #periodic BC Pressure @ x = 0

        p[0,1:-1] = (((pn[1,1:-1] + pn[-1,1:-1])*(dy**2) + (pn[0,2:]+p[0,:-2])*(dx**2)) /
                        (2*(dx**2 + dy**2)) - dx**2 * dy**2/(2 * (dx**2 + dy**2)) * b[0,1:-1])

    #wall bc, pressure

        p[:,-1] = p[:,-2]  #dp/dy = 0 at y = 2
        p[:,0]  = p[:,1]   #dp/dy = 0 at y = 0

    return p

#meshing

nx = 41            #Grid Points along X direction
ny = 41            #Grid Points along Y direction
nt = 100           #Number of Time Step

iteration = input('Enter the number of Iterations to Solve: ')          

if iteration.isdigit() == False:
    print("Please provide an integer\n")
else:
    iteration = int(iteration)

#Grid Spacing
    
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

Y,X = numpy.meshgrid(x,y)           #Generating a 2D Mesh

#fluid property and timestep and source

rho = 1                            #Density
nu  = 0.1                          #Viscosity

#Source Term to the U-Momemtum Equation
F   = 1                            

dt  = 0.01                         #time step size

#innitial conditions

u = numpy.zeros((nx, ny))
un = numpy.zeros((nx, ny))

v = numpy.zeros((nx, ny))
vn = numpy.zeros((nx, ny))

p = numpy.ones((nx, ny))
pn = numpy.ones((nx, ny))

b = numpy.zeros((nx, ny))

residual =  1       #Initial error
iterations = 0

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)   

while residual > 0.01:     #Convergence Criteria
    un = u.copy()
    vn = v.copy()

    #Calling the Pressure Field

    b = build_up_b(rho, dt, dx, dy, u, v)
    p = pressure_poisson_periodic(p, dx, dy)

    #Solving X Momentum

    u[1:-1,1:-1] = (un[1:-1,1:-1] - un[1:-1,1:-1]*((dt/dx)*(un[1:-1,1:-1]-un[:-2,1:-1]))
    - vn[1:-1,1:-1]*((dt/dy)*(un[1:-1,1:-1]-un[1:-1,:-2])) - ((dt/rho*2*dx)*(p[2:,1:-1]-p[:-2,1:-1]))
    + nu*(((dt/dx**2)*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1]))
                      + ((dt/dy**2)*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]))) + F * dt)

    #Solving Y Momentum

    v[1:-1,1:-1] = (vn[1:-1,1:-1] - un[1:-1,1:-1]*((dt/dx)*(vn[1:-1,1:-1]-vn[:-2,1:-1]))
    - vn[1:-1,1:-1]*((dt/dy)*(vn[1:-1,1:-1]-vn[1:-1,:-2])) - ((dt/rho*2*dy)*(p[1:-1,2:]-p[1:-1,:-2]))
    + nu*(((dt/dx**2)*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1]))
                      + ((dt/dy**2)*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2]))))

    
    # Periodic BC u @ x = 2

    u[-1,1:-1] = (un[-1,1:-1] - un[-1,1:-1]*((dt/dx)*(un[-1,1:-1]-un[-2,1:-1]))
    - vn[-1,1:-1]*((dt/dy)*(un[-1,1:-1]-un[-1,:-2])) - ((dt/rho*2*dx)*(p[0,1:-1]-p[-2,1:-1]))
    + nu*(((dt/dx**2)*(un[0,1:-1]-2*un[-1,1:-1]+un[-2,1:-1]))
                      + ((dt/dy**2)*(un[-1,2:]-2*un[-1,1:-1]+un[-1,:-2]))) + F * dt)

    # Periodic BC u @ x = 0

    u[0,1:-1] = (un[0,1:-1] - un[0,1:-1]*((dt/dx)*(un[0,1:-1]-un[-1,1:-1]))
    - vn[0,1:-1]*((dt/dy)*(un[0,1:-1]-un[0,:-2])) - ((dt/rho*2*dx)*(p[1,1:-1]-p[-1,1:-1]))
    + nu*(((dt/dx**2)*(un[1,1:-1]-2*un[0,1:-1]+un[-1,1:-1]))
                      + ((dt/dy**2)*(un[0,2:]-2*un[0,1:-1]+un[0,:-2]))) + F * dt)
    
    # Periodic BC v @ X = 2

    v[-1,1:-1] = (vn[-1,1:-1] - un[-1,1:-1]*((dt/dx)*(vn[-1,1:-1]-vn[-2,1:-1]))
    - vn[-1,1:-1]*((dt/dy)*(vn[-1,1:-1]-vn[-1,:-2])) - ((dt/rho*2*dy)*(p[-1,2:]-p[-1,:-2]))
    + nu*(((dt/dx**2)*(vn[0,1:-1]-2*vn[-1,1:-1]+vn[-2,1:-1]))
                      + ((dt/dy**2)*(vn[-1,2:]-2*vn[-1,1:-1]+vn[-1,:-2]))))

    # Periodic BC v @ X = 0

    v[0,1:-1] = (vn[0,1:-1] - un[0,1:-1]*((dt/dx)*(vn[0,1:-1]-vn[-1,1:-1]))
    - vn[0,1:-1]*((dt/dy)*(vn[0,1:-1]-vn[0,:-2])) - ((dt/rho*2*dy)*(p[0,2:]-p[0,:-2]))
    + nu*(((dt/dx**2)*(vn[1,1:-1]-2*vn[0,1:-1]+vn[-1,1:-1]))
                      + ((dt/dy**2)*(vn[0,2:]-2*vn[0,1:-1]+vn[0,:-2]))))

    #WALL no slip condition

    u[:,0] = 0
    u[:,-1]= 0

    #WALL no penetration condition
    
    v[:,0] = 0
    v[:,-1]= 0

    residual = (numpy.sum(u) - numpy.sum(un)) / numpy.sum(u)
    iterations += 1

print(lineSingle)
print("Solution Converged!")
print(lineSingle)

print('number of iterations :', iteration)

print(lineSingle)
print("Plotting Velocity Vectors & Contour")
print(lineSingle)

fig = pyplot.figure(figsize = (11,7), dpi=100)
pyplot.contourf(X,Y, u, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X,Y, u, cmap=cm.viridis)
pyplot.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
pyplot.xlabel('X')
pyplot.ylabel('Y')
pyplot.show()



     
    


        

                          
