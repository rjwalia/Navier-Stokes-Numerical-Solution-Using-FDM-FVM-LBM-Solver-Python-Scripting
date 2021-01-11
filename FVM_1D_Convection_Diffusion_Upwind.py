import numpy
import matplotlib.pyplot as plt

print("\n")
print("Finite Volume Method\n")
print("Solving 1D Heat Convection Diffusion Equation with Dirichlet Boundary Condition\n")
print("Discretization for Diffusion Term : Central Difference Scheme")
print("Discretization for Convection Term: Upwind Scheme\n")

k   = 100
print("Conductivity of the Material:",k,'W/m-K')

rho = 1
print("Density of the Fluid:",rho,'Kg/m3')

cp  = 1000
print("Specific Heat Capacity of the Fluid:",cp,'J/Kg-K')
print("\n")

u = float(input('Enter the flow velocity: '))

area             = 0.1
print("Cross Section Area of Rod:",area,'m2')

barLength        = 5
print("\nLength of the rod:",barLength,'m')

tempLeft         = 100
tempRight        = 200
print("Temperature at the Left End of the Rod:",tempLeft,'C')
print("Temperature at the Right End of the Rod:",tempRight,'C')

heatSourcePerVol = 1000
print("Heat Source in the Rod:",heatSourcePerVol,'W/m3')
print("\n")

nCell = int(input('Enter the number of Cells for Meshing the Rod: '))


print ('------------------------------------------------')
print (' Creating Mesh')
print ('------------------------------------------------')

#cell coordinates
xFace      = numpy.linspace(0, barLength, nCell+1)

#cell centroids
xCentroid  = 0.5*(xFace[1:] + xFace[:-1])

#cell length
cellLength = xFace[1:] - xFace[:-1]

#distance between cell centroids
dCentroid  = xCentroid[1:] - xCentroid[:-1]

# For the boundary cell on the left, the distance is double the distance 
# from the cell centroid to the boundary face
dLeft      = 2*(xCentroid[0] - xFace[0])

# For the boundary cell on the right, the distance is double the distance
#from the cell centroid to the boundary cell face
dRight     = 2*(xFace[-1] - xCentroid[-1])

# Append these to the vector of distances
dCentroid  = numpy.hstack([dLeft, dCentroid, dRight])

#cellVolume
cellVolume = area*cellLength

print ('------------------------------------------------')
print (' Calculating Matrix Coefficients')
print ('------------------------------------------------')

#diffusive flux
DA = area*numpy.divide(k, dCentroid)

#convective flux 
velocityVector = u*numpy.ones(nCell+1)
F              = velocityVector*rho*area*cp

#peclet no.
Pe = F/DA

#source term Sp
Sp = numpy.zeros(nCell)
Sp[0] = -(2*numpy.copy(DA[0]) + numpy.maximum(numpy.copy(F[0]),0))
Sp[-1] = -(2*numpy.copy(DA[-1]) + numpy.maximum(-numpy.copy(F[-1]),0))

#souce term Su
Su     = heatSourcePerVol*cellVolume
Su[0]  =Su[0] + tempLeft*(2*numpy.copy(DA[0]) + numpy.maximum(numpy.copy(F[0]),0))
Su[-1] =Su[-1] + tempRight*(2*numpy.copy(DA[-1]) + numpy.maximum(-numpy.copy(F[-1]),0))

#left and right coefficient
aL     = numpy.copy(DA[0:-1]) + numpy.maximum(numpy.copy(F[0:-1]),numpy.zeros(nCell))
aR     = numpy.copy(DA[0:-1]) + numpy.maximum(-numpy.copy(F[0:-1]),numpy.zeros(nCell))
aL[0]  = 0
aR[-1] = 0

#central coeff Ap
aP = numpy.copy(aL) + numpy.copy(aR) - numpy.copy(Sp)

print ('------------------------------------------------')
print (' Assembling Matrices')
print ('------------------------------------------------')

Amatrix = numpy.zeros([nCell, nCell])
Bvector = numpy.zeros(nCell)

for i in range(nCell):

    if i == 0:
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i+1] = -1*aR[i]

    elif i == nCell - 1:
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i-1] = -1*aL[i]

    else:
        Amatrix[i,i-1] = -1*aL[i]
        Amatrix[i,i]   =  aP[i]
        Amatrix[i,i+1] =  -1*aR[i]

    Bvector[i] = Su[i]



print('aL:',aL)
print('aR:',aR)
print('aP:',aP)
print('Sp:',Sp)
print('Su:',Su)
print('\nCell Peclet Number:',Pe)
print('\nAmatrix:')
print(Amatrix)
print('\nBvector:',Bvector)

print ('------------------------------------------------')
print (' Solving ...')
print ('------------------------------------------------')

Tvector = numpy.linalg.solve(Amatrix, Bvector)

print ('------------------------------------------------')
print (' Equations Solved')
print ('------------------------------------------------')

print('---------------------------------------------')
print('Solution: Temperature Field')
Tvector = numpy.around(Tvector, decimals = 2) 
print(Tvector)
print("\n")

print ('------------------------------------------------')
print (' Plotting ...')
print ('------------------------------------------------')

xPlotting = numpy.hstack([xFace[0], xCentroid, xFace[-1]])
temperaturePlotting = numpy.hstack([tempLeft, Tvector, tempRight])

tickPad = 8
tickPad2 = 16
labelPadY = 10
labelPadX = 8
boxPad = 5
darkBlue = (0.0,0.129,0.2784)
darkRed = (0.7176, 0.0705, 0.207)

fig1 = plt.figure()
ax = fig1.add_subplot()
fig1.tight_layout(pad=boxPad)

ax.plot(xPlotting , temperaturePlotting, 'b-o',linewidth = 2, label='CFD', color=darkBlue)

plt.xlabel(r'$x$ [m]', fontsize=14, labelpad = labelPadX)
plt.ylabel(r'$T$ [$^{\circ}$C]', fontsize=14, labelpad = labelPadY)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.xlim([xFace[0], xFace[-1]])

leg = plt.legend(fontsize = 14, loc='best', fancybox=False, edgecolor = 'k')
leg.get_frame().set_linewidth(2)
ax.tick_params(which = 'both', direction='in', length=6,width=2, gridOn = False)

ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(pad=tickPad)
plt.show()
