import numpy  #give mathamatical or matrix expressions
import sympy  #SymPy is the symbolic math library for Python. 

from sympy import init_printing       
init_printing(use_latex = True) #Output Rendering using LATEX.

lineSingle = '------------------------------------------------'

print("Solving 1D Burgers Equation using Finite Difference Scheme\n")

#setting up symbolic variables for the three variables in our initial condition then type full eqn of phi

#innitial condition

x, nu, t = sympy.symbols('x nu t')
phi      = (sympy.exp(-(x - 4*t)**2/(4*nu*(t + 1))) + sympy.exp(-(x - 4*t - 2*sympy.pi)**2/(4*nu*(t+1))))       #phi expression

print(lineSingle)
print("printing phi expression")
print(lineSingle)
print(phi)


phiprime = phi.diff(x)        #differentiation wrt x
phiprime

from sympy.utilities.lambdify import lambdify       #translate sympy symbolic initial condition equation into a usable Python expression.

u = -2*nu*(phiprime/phi) + 4                        #Initial condition expression

print(lineSingle)
print("Initial Condition expression")
print(lineSingle)
print(u)

ufunc = lambdify((t, x, nu), u)    #putting value of t, x and nu in u and getting the output


from matplotlib import pyplot     

#seting up the grid

nx = 101                                #grid points
nt =  20                                #number of timesteps
dx =  2 * numpy.pi / (nx - 1)           #grid spacing
nu = .07                                #viscosity
dt = dx * nu                            #timestep size

x  = numpy.linspace(0, 2 * numpy.pi, nx)
un = numpy.empty(nx)
t  = 0                                  #initial time t = 0

u  = numpy.asarray([ufunc(t, x0, nu) for x0 in x])        #initial condition plot using our lambdify-ed function

print(lineSingle)
print("Computing Innitial Solution")
print(lineSingle)
print(u)

print(lineSingle)
print("Plotting Innitial Solution")
print(lineSingle)

pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.plot(x, u, marker='o', lw=2, label='Initial Solution')
pyplot.title('1D Burgers Convection')
pyplot.xlabel('Grid Space')
pyplot.ylabel('Velocity')
pyplot.xlim([0, 2 * numpy.pi])
pyplot.ylim([0, 10]);

pyplot.legend();
pyplot.show()

print(lineSingle)
print("Computing Analytical Solution")
print(lineSingle)

u_analytical = numpy.asarray([ufunc(nt * dt, xi, nu) for xi in x])            #computing analytical solution

print(lineSingle)
print("Printing Analytical Solution")
print(lineSingle)

print(u_analytical)

print(lineSingle)
print("Calculating Numerical Solution......")
print(lineSingle)

for n in range(nt):                 #time marching
    un = u.copy()                   
    for i in range(1, nx - 1):      #space marching

        #Backward Difference for Convection Term
        #Central Difference for Diffusion Term
        
        u[i] = un[i] - un[i]*dt/dx*(un[i] - un[i-1]) + nu*dt/dx**2*(un[i+1]-2*un[i]+un[i-1])

        #periodic boudnary condition
        
        u[0] = un[0] - un[0]*dt/dx*(un[0] - un[-2]) + nu*dt/dx**2*(un[1]-2*un[0]+un[-2])
        u[-1] = u[0]
        
print(lineSingle)
print("Printing Numerical Solution......")
print(lineSingle)

print(u)


print(lineSingle)
print("Plotting Innitial, Analytical & Numerical Solution")
print(lineSingle)

pyplot.figure(figsize=(11,7), dpi=100)

pyplot.plot(x,u, marker = 'o', lw = 2, label='Computational Solution')
pyplot.plot(x, u_analytical, label = 'Analytical Solution')
pyplot.xlim([0, 2*numpy.pi])
pyplot.ylim([0, 10])

pyplot.title('1D Burgers Convecction')
pyplot.xlabel('Grid Space')
pyplot.ylabel('Velocity')

pyplot.legend();
pyplot.show()
