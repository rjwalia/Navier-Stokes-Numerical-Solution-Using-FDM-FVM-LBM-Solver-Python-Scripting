import numpy as np
import matplotlib.pyplot as  plt

print("\n")
print("Finite Volume Method\n")
print("Standard Wall Function Approach\n")
print("Solving 1D Heat Diffusion Equation with Dirichlet BC\n")
print("Discretization for Diffusion Term: Central Difference Scheme\n")

cond = 100             
print("Conductivity of the Material:",cond,'W/m-K')

area = 0.1              
print("Cross Section Area of Rod:",area,'m2')

barLength = 5          
print("\nLength of the rod:",barLength,'m')

nCells = int(input('Enter the number of Cells for Meshing the Rod: '))

heatfluxLeftEnd = 100
print("Heat Flux at the Left End of the Rod:",heatfluxLeftEnd,'C/m')

TRightEnd = 200
print("Temperature at the Right End of the Rod:",TRightEnd,'C')

heatSourcePerVol = 1000     
print("Heat Source in the Rod:",heatSourcePerVol,'W/m3')
print("\n")

density = 8000
print("Density of the Rod:",density,'Kg/m3')

cp = 500
print("Specific Heat Capacity of the Rod:",density,'J/kg-C')

yPlus = int(input('Enter the value of Target Y+: '))

Pr  = 0.71

lineSingle = '------------------------------------------------'
lineDouble = '================================================'

print ('Creating Mesh')
print(lineSingle)

xFaces           = np.linspace(0, barLength, nCells+1)
xCentroid        = 0.5*(xFaces[1:] + xFaces[0:-1])
dCentroid        = xCentroid[1:] - xCentroid [0:-1]
dLeft            = 2*(xCentroid[0] - xFaces[0])
dRight           = 2*(xFaces[-1] - xCentroid[-1])

dCentroids       = np.hstack([dLeft, dCentroid, dRight])
dCentroidsLeft   = dCentroids[0:-1]
dCentroidsRight  = dCentroids[1:]

areaLeftFaces    = area*np.ones(nCells)
areaRightFaces   = area*np.ones(nCells)

cellLength       = xFaces[1:] -xFaces[0:-1]
cellVolume       = cellLength*area

print ('Computing Wall Function')
print(lineSingle)

Prt    = 0.85

E      = 9.7983     
kappa  = 0.4187

P_Pr   = Pr/Prt
P      = 9.24*(np.power(P_Pr, 0.75)-1)*(1 + 0.28*np.exp(-0.007*P_Pr))


yPlusL = 11.0
for i in range(10):
    f          = ((Pr*yPlusL) - (Prt*(np.log(E*yPlusL)/kappa + P)))
    df         = Pr - (Prt/(kappa*yPlusL))
    yPlusLNew  = yPlusL - f/df
    if (np.abs(yPlusLNew - yPlusL) < 1e-6):
        break
    else:
        yPlusL = yPlusLNew

alpha = cond/(density*cp)

if yPlus < yPlusL:
    alphaWall = alpha
else:
    alphaWall = alpha*((Pr*yPlus)/(Prt*(((1/kappa)*(np.log(E*yPlus))+P))))



print('Wall Function: Summmary')
print(lineSingle)
print('Pr              = %6.3f'%Pr)
print('Prt             = %6.3f'%Prt)
print('P               = %5.3f'% P)
print('y+              = %5.3f'%yPlus)
print('y+_L            = %5.3f'%yPlusL)
print('alphaWall/alpha = %6.3f'%(alphaWall/alpha))
print(lineSingle)

print (' Assigning Material Properties')
print(lineSingle)
 
conductivityFaces      = cond*np.ones(nCells+1)

kWall                  = alphaWall*density*cp
conductivityFaces[-1]  = kWall

conductivityLeftFaces  = conductivityFaces[0:-1]
conductivityRightFaces = conductivityFaces[1:]

print (' Calculating Matrix Coefficients')
print(lineSingle)

DA_LeftFaces  = np.divide(
               np.multiply(conductivityLeftFaces, areaLeftFaces), dCentroidsLeft) 
DA_RightFaces = np.divide(
               np.multiply(conductivityRightFaces, areaRightFaces),dCentroidsRight)


Su     = heatSourcePerVol*cellVolume

Su[0]  = Su[0] - heatfluxLeftEnd*area
Su[-1] = Su[-1] + TRightEnd*(2*np.copy(DA_RightFaces[-1]))

Sp     =  np.zeros(nCells)
Sp[0]  =  0
Sp[-1] = -2*np.copy(DA_RightFaces[-1])

aL     = np.copy(DA_LeftFaces)
aR     = np.copy(DA_RightFaces)

aL[0]  = 0
aR[-1] = 0

aP     = np.around(np.copy(aL) + np.copy(aR) - np.copy(Sp),decimals = 2)

print(' Assembling Matrices')
print(lineSingle)

Amatrix = np.zeros([nCells, nCells])
BVector = np.zeros(nCells)

for i in range(nCells):

    if i == 0:
        Amatrix[i,i]   =  aP[i]
        Amatrix[i,i+1] = -aR[i]

    if i == nCells-1:
        Amatrix[i,i]   =  aP[i]
        Amatrix[i,i-1] = -aL[i]

    else:
        Amatrix[i,i+1] = -aR[i]
        Amatrix[i,i]   =  aP[i]
        Amatrix[i,i-1] = -aL[i]

    BVector[i] = np.around(Su[i],decimals = 2)



print (' Summary: Set Up')
print(lineSingle)
print ('Cell | aL | aR  | ap  |  Sp  |   Su ')
print(lineSingle)
for i in range(nCells):
    print('%4i %5.1f %5.1f %5.1f %5.1f %8.1f ' 
        % (i+1, aL[i], aR[i], aP[i], Sp[i], Su[i]))
print(lineSingle)
np.set_printoptions(linewidth=np.inf)
print ('A matrix:')
print(lineSingle)
print(Amatrix)
print('B vector')
print(lineSingle)
print(BVector)
print(lineSingle)


print (' Solving ...')
print(lineSingle)

# Use the built-in python solution module 
Tvector = np.around(np.linalg.solve(Amatrix, BVector),decimals = 2)

print (' Equations Solved')
print(lineSingle)



print (' Solution: Temperature Vector')
print(lineSingle)
print(Tvector)
print(lineSingle)

print (' Calculating Heat Fluxes ...')
print(lineSingle)

tempLeft = (Tvector[0] - (heatfluxLeftEnd*area)/(2*np.copy(DA_LeftFaces[0])))

temperatureStack = np.hstack([tempLeft, np.copy(Tvector), TRightEnd])

temperatureDifferenceLeft  = temperatureStack[1:-1] - temperatureStack[0:-2]
temperatureDifferenceRight = temperatureStack[2:] - temperatureStack[1:-1]

normalsLeft  = -1.0*np.ones(nCells)
normalsRight = np.ones(nCells)

heatFluxLeft  = -1*np.prod([normalsLeft,temperatureDifferenceLeft,DA_LeftFaces],0)
heatFluxRight = -1*np.prod([normalsRight,temperatureDifferenceRight,DA_RightFaces],0)

heatFluxLeft[0] *= 2.0
heatFluxRight[-1] *= 2.0

heatSource       = heatSourcePerVol*cellVolume*np.ones(nCells)
heatBalanceError = heatSource - heatFluxLeft - heatFluxRight


print(' Heat Fluxes')
print(lineSingle)
print ('Cell |  QL   |  QR   |  SV   |  Error')
print(lineSingle)
for i in range(nCells):
    print ('%4i %7.1f %7.1f %7.1f %7.1f' % (
        i+1, heatFluxLeft[i], heatFluxRight[i], 
        heatSource[i], heatBalanceError[i]))
print(lineSingle)



print (' Plotting ...')
print (lineSingle)

xPlotting = np.hstack([xFaces[0], np.copy(xCentroid), xFaces[-1]])

fontSize = 11
fontSizeLegend = 11
lineWidth = 1.5
tickPad = 8
tickPad2 = 16
labelPadY = 3
labelPadX = 2
boxPad = 2
tickLength = 4
markerSize = 4

lightBlue = '#bfc8d1'
shadeBlue = '#8091a4'
darkBlue  = '#002147'


plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["figure.figsize"] = (3.1,2.5)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
fig1.tight_layout(pad=boxPad)
ax.plot(xPlotting, temperatureStack, 'b-o',markersize=markerSize,linewidth = 1.5, label='CFD', color=darkBlue)
plt.xlabel(r'$x$ [m]', fontsize=fontSize, labelpad = labelPadX)
plt.ylabel(r'$T$ [$^{\circ}$C]', fontsize=fontSize, labelpad = labelPadY)
plt.title('Temperature Distribution Along the Bar')
plt.yticks(fontsize = fontSize)
plt.xticks(np.linspace(xFaces[0], xFaces[-1], int(barLength)+1), fontsize = fontSize)
plt.xlim([xFaces[0], xFaces[-1]])
ax.tick_params(which = 'both', direction='in', length=tickLength,width=1.5, gridOn = False, pad=tickPad, color=darkBlue)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
ax.spines['bottom'].set_color(darkBlue)
ax.spines['top'].set_color(darkBlue) 
ax.spines['right'].set_color(darkBlue)
ax.spines['left'].set_color(darkBlue)


if yPlus < 11.0:
    print("\n!!!!!!!!! ----- NOTE -----!!!!!!!!!") 
    print("\nTemperature Gradient is resolved as y+ is less than 11\n")
else:
    print("\n!!!!!!!!! ----- NOTE -----!!!!!!!!!") 
    print("\nTemperature Gradient is not resolved as y+ is more than < 11")
    print("Heat Flux is corrected by using Standard Wall Function.\n")

plt.show()
