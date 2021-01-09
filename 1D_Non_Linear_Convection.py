import numpy
from matplotlib import pyplot

def non_linear_conv(nx):             #grip point
    dx  = 2/ (nx - 1)                #distance between 2 nodes
    nt  = 20                         #number of timesteps
    cfl = 0.5                        

    dt  = cfl * dx                   #timestep size

    #innitial condition
    
    u   = numpy.ones(nx)
    u[int(0.5/dx):int(1/dx+1)] = 2  #Square Wave Profile
    print(u)
    pyplot.plot(numpy.linspace(0,2,nx), u, label='Initial Solution')

    #discritization of the convection equation


    un  = numpy.ones(nx) 
    for n in range(nt):                                              #time marching
        un = u.copy() 
        for i in range(1,nx):                                        #Space marching
            u[i] = un[i] - un[i]*dt/dx* (un[i] - un[i-1])            #Backward Differnece Scheme

    pyplot.plot(numpy.linspace(0,2,nx), u, label='Convected Solution')
    pyplot.title('1D Non Linear Convecction')
    pyplot.xlabel('Grid Space')
    pyplot.ylabel('Velocity')

    print(u)
    pyplot.legend()
    pyplot.show()


