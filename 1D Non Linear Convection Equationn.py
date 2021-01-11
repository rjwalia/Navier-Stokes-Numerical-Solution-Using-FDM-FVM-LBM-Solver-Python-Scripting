
import numpy
from matplotlib import pyplot

lineSingle = '------------------------------------------------'

print("Solving 1D Non Linear Equation using  Finite Difference Method\n")

while True:
    nx = input('Enter the number of Grid Points: ')           #grip points

    if nx.isdigit() == False:
        print("Please provide an integer\n")
        continue
    else:
        nx = int(nx)
            
    dx  = 2/ (nx - 1)                                       #grid spacing  
    nt  = 20                                                #number of timesteps
    cfl = 0.5                        

    dt  = cfl * dx                   #timestep size

    #innitial condition

    print(lineSingle)
    print("Computing Innitial Solution...")
        
    u   = numpy.ones(nx)
    u[int(0.5/dx):int(1/dx+1)] = 2  #Square Wave Profile

    print("Printing Innitial Solution...")
    print(lineSingle)
    print(u)
    
    pyplot.plot(numpy.linspace(0,2,nx), u, label='Initial Solution')

    print(lineSingle)
    print("Calculating Numerical Solution......")
    print(lineSingle)

    #discritization
    
    un  = numpy.ones(nx) 
    for n in range(nt):                                              #time marching
        un = u.copy() 
        for i in range(1,nx):                                        #Space marching
            u[i] = un[i] - un[i]*dt/dx* (un[i] - un[i-1])            #Backward Differnece Scheme
            
    print(lineSingle)
    print("Printing Numerical Solution......")
    print(lineSingle)
        
    print(u)

    print(lineSingle)
    print("Plotting Innitial & Numerical Solution")
    print(lineSingle)
    pyplot.plot(numpy.linspace(0,2,nx), u, label='Convected Solution')
    pyplot.title('1D Non Linear Convecction')
    pyplot.xlabel('Grid Space')
    pyplot.ylabel('Velocity')

    pyplot.legend()
    pyplot.show()
    break
    


