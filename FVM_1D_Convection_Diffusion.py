import numpy as np
import matplotlib.pyplot as plt

print("\n")
print("Finite Volume Method\n")
print("Solving 1D Heat Convection Diffusion Equation with Dirichlet Boundary Condition\n")
print("Discretization for Diffusion Term : Central Difference Scheme")
print("Discretization for Convection Term: Central Difference Scheme\n")

k    = 100
print("Conductivity of the Material:",k,'W/m-K')

rho  = 1
print("Density of the Fluid:",rho,'Kg/m3')

cp   = 1000
print("Specific Heat Capacity of the Fluid:",cp,'J/Kg-K')
print("\n")

u = float(input('Enter the flow velocity: '))

area = 0.1
print("Cross Section Area of Rod:",area,'m2')

barLength = 5
print("\nLength of the rod:",barLength,'m')

tempLeft = 100
tempRight = 200
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
xFace = np.linspace(0, barLength, nCell+1)

#cell centroids
xCentroid = 0.5*(xFace[1:] + xFace[0:-1])

#cell length
cellLength = xFace[1:] - xFace[:-1]

#distance between cell centroids
dCentroid = xCentroid[1:] - xCentroid[:-1]

# For the boundary cell on the left, the distance is double the distance 
# from the cell centroid to the boundary face
dLeft = 2*(xCentroid[0] - xFace[0])

# For the boundary cell on the right, the distance is double the distance
#from the cell centroid to the boundary cell face
dRight = 2*(xFace[-1] - xCentroid[-1])

# Append these to the vector of distances
dCentroid = np.hstack([dLeft, dCentroid, dRight])

dCentroidLeft  = dCentroid[0:-1]
dCentroidRight = dCentroid[1:]
areaLeftFaces  = area*np.ones(nCell)
areaRightFaces = area*np.ones(nCell)

conductivityFaces      = k*np.ones(nCell+1)
conductivityLeftFaces  = conductivityFaces[0:-1]
conductivityRightFaces = conductivityFaces[1:]


#cellVolume
cellVolume = cellLength*area

print ('------------------------------------------------')
print (' Calculating Matrix Coefficients')
print ('------------------------------------------------')

#diffusive flux
DA = area*np.divide(k, dCentroid)
DA_LeftFaces = np.divide(np.multiply(conductivityLeftFaces,areaLeftFaces),dCentroidLeft)
DA_RightFaces= np.divide(np.multiply(conductivityRightFaces,areaRightFaces),dCentroidRight)

#convective flux 
velocityVector = u*np.ones(nCell+1)
densityVector = rho*np.ones(nCell+1)
F = np.multiply(velocityVector, densityVector)*area*cp
F_LeftFaces  = F[0:-1]
F_RightFaces = F[1:]

#peclet no.
Pe = F/DA

#source term Sp
Sp = np.zeros(nCell)

#Sp is not zero in left and right boundaries
Sp[0]  = -(2*(DA[0]) + F[0])
Sp[-1] = -(2*(DA[-1]) - F[-1])

#souce term Su
Su = heatSourcePerVol*cellVolume

#source term on left and right boundary
Su[0]  = Su[0] + tempLeft*(2*DA[0] + F[0])
Su[-1] = Su[-1] + tempRight*(2*DA[-1] - F[-1])

#left and right coefficient
aL = DA[0:-1] + 0.5*(F[:-1])
aR = DA[0:-1] - 0.5*(F[:-1])

aL[0] = 0
aR[-1] = 0

#central coeff Ap
aP = aL + aR - Sp

#create matrix

print ('------------------------------------------------')
print (' Assembling Matrices')
print ('------------------------------------------------')

# create empty matrix A & empty vector B
Amatrix = np.zeros([nCell, nCell])
Bvector = np.zeros(nCell)

for i in range(nCell):

    #leftboundary
    if i == 0:
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i+1] = -1*aR[i]
        
    #rightboundary
    elif i == nCell-1:
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i-1] = -1*aL[i]

    #interior cells
    else:
        Amatrix[i,i-1] =  -1*aL[i]
        Amatrix[i,i]   =  aP[i]
        Amatrix[i,i+1] =  -1*aR[i]

    Bvector[i] = Su[i]

#Print the setup



print('aL:',aL)
print('aR:',aR)
print('aP:',aP)
print('Sp:',Sp)
print('Su:',Su)
print('\nCell Peclet Number:',Pe)
print('\nAmatrix:')
print(Amatrix)
print('\nBvector:',Bvector)
print('\n')

#Solve the matrices

print ('------------------------------------------------')
print (' Solving ...')
print ('------------------------------------------------')

# Use the built-in python solution module 
Tvector = np.linalg.solve(Amatrix, Bvector)

print ('------------------------------------------------')
print (' Equations Solved')
print ('------------------------------------------------')

#print result

print('---------------------------------------------')
print('Solution: Temperature Field')
Tvector = np.around(Tvector, decimals = 2)
print(Tvector)
print("\n")

temperatureStack = np.hstack([tempLeft, np.copy(Tvector), tempRight])

temperatureDifferenceLeft  = temperatureStack[1:-1] - temperatureStack[0:-2]
temperatureDifferenceRight = temperatureStack[2:] - temperatureStack[1:-1]

temperatureFaces = 0.5*(Tvector[0:-1]+Tvector[1:])
temperatureFacesStack = np.hstack([tempLeft, np.copy(temperatureFaces), tempRight])
temperatureLeftFaces = temperatureFacesStack[0:-1]
temperatureRightFaces = temperatureFacesStack[1:]

normalLeft  = -1*np.ones(nCell)
normalRight =  1*np.ones(nCell)

heatfluxLeftCond  = -1*np.prod([normalLeft, temperatureDifferenceLeft, DA_LeftFaces],0) 
heatfluxRightCond = -1*np.prod([normalRight, temperatureDifferenceRight, DA_RightFaces],0) 

heatfluxLeftConv  =   1 * np.prod([temperatureLeftFaces, F_LeftFaces],0)
heatfluxRightConv =   1 * np.prod([temperatureRightFaces, F_RightFaces],0)

heatfluxLeftCond[0]   *= 2.0
heatfluxRightCond[-1] *= 2.0

heatSource = heatSourcePerVol*cellVolume
heatBalanceError = heatSource - heatfluxLeftCond - heatfluxRightCond + heatfluxLeftConv - heatfluxRightConv



print ('Energy Balance:')
print ('------------------------------------------------')
print ('Cell |  QL_Cond   |  QR_Cond   |  QL_Conv   |  QR_Conv   |  SV   |  Error')
print ('------------------------------------------------')
for i in range(nCell):
    print('%4i %10.1f %12.1f %12.1f %11.1f %10.1f %7.1f' %(i+1, heatfluxLeftCond[i], heatfluxRightCond[i], heatfluxLeftConv[i], heatfluxRightConv[i],
            heatSource[i], heatBalanceError[i]))

if Pe[i]>2:
    print("\n")
    print("Solution is diverged!")
    print("\n")
    print("Osillation observed in the solution!")
    print("This is due to High Convective Strength.")
    print("Central Differnce Scheme is not stable for highly Convective Flow.")
    print("Try to reduce Flow Velocity or Refine the Mesh or Switch to Upwind Scheme for Convection Term.\n")
    print("Use another code where Upwind Scheme for Convection Term is used.\n") 




#plot result
    

print ('------------------------------------------------')
print ('Plotting the results ...')
print ('------------------------------------------------')

#Append the boundary temperature values to the vector, so we can 
#plot the complete solution
    
xPlotting = np.hstack([xFace[0], np.copy(xCentroid), xFace[-1]])
temperaturePlotting = np.hstack([tempLeft, np.copy(Tvector), tempRight])

# Configure the plot to look how you want
tickPad = 8
tickPad2 = 16
labelPadY = 10
labelPadX = 8
boxPad = 5
darkBlue = (0.0,0.129,0.2784)
darkRed = (0.7176, 0.0705, 0.207)

plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = (6.2,4.2)
plt.rcParams['axes.linewidth'] = 2

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

ax.tick_params(which = 'both', direction='in', length=6, width=2, gridOn = False)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')

ax.tick_params(pad=tickPad)
plt.show()
