import numpy as np
import matplotlib.pyplot as plt

print("\n")
print("Finite Volume Method\n")
print("Solving 1D Heat Diffusion Equation with Neumann Boundary Condition\n")
print("Discretization for Diffusion Term: Central Difference Scheme\n")

cond             = 100
print("Conductivity of the Material:",cond,'W/m-K')

area             = 0.1
print("Cross Section Area of Rod:",area,'m2')

barLength        = 5
print("\nLength of the rod:",barLength,'m')

heatfluxLeftbc   = 200
print("Heat Flux at the Left End of the Rod:",heatfluxLeftbc,'C/m')

tempRight        = 200
print("Temperature at the Right End of the Rod:",tempRight,'C')

heatSourcePerVol = 1000     
print("Heat Source in the Rod:",heatSourcePerVol,'W/m3')
print("\n")

nCell = int(input('Enter the number of Cells for Meshing the Rod: '))   #Number of Cells

lineSingle = '------------------------------------------------'
lineDouble = '================================================'

print('Creating Mesh')
print(lineSingle)

#cell coordinates
xFace = np.linspace(0, barLength, nCell+1)

#cell centroids
xCentroid = 0.5*(xFace[1:] + xFace[0:-1])

#cell length
cellLength = xFace[1:] - xFace[0:-1]

#distance between cell centroids
dCentroid = xCentroid[1:] - xCentroid[0:-1]

# For the boundary cell on the left, the distance is double the distance 
# from the cell centroid to the boundary face
dLeft = 2*(xCentroid[0] - xFace[0])

# For the boundary cell on the right, the distance is double the distance
#from the cell centroid to the boundary cell face
dRight = 2*(xFace[-1] - xCentroid[-1])

# Append these to the vector of distances
dCentroid      = np.hstack([dLeft, dCentroid, dRight])
dCentroidLeft  = dCentroid[0:-1]
dCentroidRight = dCentroid[1:]

#Area of cell Faces
areaLeftFaces  = area*np.ones(nCell)
areaRightFaces = area*np.ones(nCell)

#cellvolume
cellVolume     = cellLength*area

print('Assigning Material Properties')
print(lineSingle)

conductivityFaces      = cond*np.ones(nCell+1)
conductivityLeftFaces  = conductivityFaces[0:-1]
conductivityRightFaces = conductivityFaces[1:]

print ('Calculating Matrix Coefficients')
print(lineSingle)

#diffusive flux per unit area
DA_LeftFaces = np.divide(np.multiply(conductivityLeftFaces,areaLeftFaces),dCentroidLeft)
DA_RightFaces= np.divide(np.multiply(conductivityRightFaces,areaRightFaces),dCentroidRight)

#souce term Su
Su     = heatSourcePerVol*cellVolume
Su[0]  = Su[0] - heatfluxLeftbc*area
Su[-1] = Su[-1] + 2*tempRight*np.copy(DA_RightFaces[-1])

#source term Sp
Sp     = np.zeros(nCell)
Sp[0]  = 0
Sp[-1] = -2*np.copy(DA_RightFaces[-1])

#left coefficient aL
aL = np.copy(DA_LeftFaces)

#right coefficient aR
aR = np.copy(DA_RightFaces)

aL[0]  = 0
aR[-1] = 0

#central coeff Ap
aP = np.copy(aL) + np.copy(aR) - np.copy(Sp)

print ('Assembling Matrices')
print(lineSingle)

# creat empty matrix A & empty vector B

Amatrix = np.zeros([nCell, nCell])
BVector = np.zeros(nCell)

for i in range(nCell):

    if i == 0:
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i+1] = -aR[i]

    elif i == nCell-1:
        Amatrix[i,i-1] = -aL[i]
        Amatrix[i,i]   = aP[i]

    else:
        Amatrix[i,i-1] = -aL[i]
        Amatrix[i,i]   = aP[i]
        Amatrix[i,i+1] = -aR[i]

    BVector[i] = Su[i]


print ('Summary: Set Up')
print(lineSingle)
print ('Cell | aL | aR  | ap  |  Sp  |   Su ')
print(lineSingle)
for i in range(nCell):
    print ('%4i %5.1f %5.1f %5.1f %5.1f %7.1f ' % (
        i+1, aL[i], aR[i], aP[i], Sp[i], Su[i]))
print(lineSingle)
np.set_printoptions(linewidth=np.inf)
print ('A matrix:')
print(lineSingle)
print(Amatrix)
print ('B vector')
print(lineSingle)
print(BVector)
print(lineSingle)

print ('Solving ...')
print(lineSingle)

Tvector = np.linalg.solve(Amatrix, BVector)

print ('Equations Solved')
print(lineSingle)



print('Solution: Temperature Vector')
print(lineSingle)
Tvector = np.around(Tvector, decimals = 2) 
print(Tvector)
print(lineSingle)

tempLeft = (Tvector[0] - (heatfluxLeftbc*area)/(2*np.copy(DA_LeftFaces[0])))

print('Left Boudnary Temperature:')
print(lineSingle)
print(tempLeft)

temperatureStack = np.hstack([tempLeft, np.copy(Tvector), tempRight])

temperatureDifferenceLeft  = temperatureStack[1:-1] - temperatureStack[0:-2]
temperatureDifferenceRight = temperatureStack[2:] - temperatureStack[1:-1]

normalLeft  = -1*np.ones(nCell)
normalRight =  1*np.ones(nCell)

heatfluxLeft  = -1*np.prod([normalLeft, temperatureDifferenceLeft, DA_LeftFaces],0)
heatfluxRight = -1*np.prod([normalRight, temperatureDifferenceRight, DA_RightFaces],0)

heatfluxLeft[0]   *= 2.0
heatfluxRight[-1] *= 2.0

heatSource = heatSourcePerVol*cellVolume
heatBalanceError = heatSource - heatfluxLeft - heatfluxRight


print ('\nHeat Balance:')
print(lineSingle)
print ('Cell |  QL   |  QR   |  SV   |  Error')
print(lineSingle)
for i in range(nCell):
    print('%4i %7.1f %7.1f %7.1f %7.1f' %(i+1, heatfluxLeft[i], heatfluxRight[i],
              heatSource[i], heatBalanceError[i]))
print(lineSingle)



print (' Plotting ...')
print(lineSingle)

xPlotting = np.hstack([xFace[0], np.copy(xCentroid), xFace[-1]])
    
xAnalytical = np.linspace(0, barLength, 100)
temperatureAnalytical = ((heatSourcePerVol/(2.0*cond))
                            *(barLength*barLength*np.ones(len(xAnalytical)) 
                            - np.square(xAnalytical)) 
                            + (heatfluxLeftbc/cond)*(xAnalytical 
                            - barLength*np.ones(len(xAnalytical)))
                            + tempRight)

fontSize = 14
fontSizeLegend = 14
lineWidth = 2.0
tickPad = 8
tickPad2 = 16
labelPadY = 1
labelPadX = 2
boxPad = 5
darkBlue = (0.0,0.129,0.2784)
darkRed = (0.7176, 0.0705, 0.207)


plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = (6.2,4.2)
plt.rcParams['axes.linewidth'] = lineWidth

fig1 = plt.figure()
ax = fig1.add_subplot()
fig1.tight_layout(pad=boxPad)
ax.plot(xPlotting , temperatureStack, 'bo',linewidth = lineWidth, label='CFD', color=darkRed)
ax.plot(xAnalytical, temperatureAnalytical, 'k--',linewidth = lineWidth, label = 'Analytical', color=darkBlue)
plt.xlabel(r'$x$ [m]', fontsize=fontSize, labelpad = labelPadX)
plt.ylabel(r'$T$ [$^{\circ}$ C]', fontsize=fontSize, labelpad = labelPadY)
plt.title('Temperature Distribution Along the Bar')
plt.yticks(fontsize = fontSize)
plt.xticks(fontsize = fontSize)
plt.xlim([xFace[0], xFace[-1]])
leg = plt.legend(fontsize = fontSizeLegend, loc='best', edgecolor = 'r')
leg.get_frame().set_linewidth(lineWidth)
ax.tick_params(which = 'both', direction='in', length=6,width=lineWidth, gridOn = False)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.tick_params(pad=tickPad)
plt.show()
