import numpy
import matplotlib.pyplot as plt

cond = 100
area = 0.1
barLength = 5
nCells = 8
tempLeft = 100 #bc
tempRight = 200 #bc
heatSourcePerVol = 1000

# Plot the data?
plotOutput = 'true'
# Print the set up data? (table of coefficients and matrices)
printSetup = 'true'
# Print the solution output (the temperature vector)
printSolution = 'true'

#create mesh

print ('------------------------------------------------')
print (' Creating Mesh')
print ('------------------------------------------------')

#cell coordinates
xFace = numpy.linspace(0, barLength, nCells+1)

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
dCentroid = numpy.hstack([dLeft, dCentroid, dRight])

#cellvolume
cellVolume = cellLength*area

#Calculate the Matrix Coefficients

print ('------------------------------------------------')
print (' Calculating Matrix Coefficients')
print ('------------------------------------------------')

#diffusive flux per unit area
DA = area*numpy.divide(cond, dCentroid)

#source term Sp
Sp = numpy.zeros(nCells)

#Sp is not zero in left and right bounndaries
Sp[0] = -2*numpy.copy(DA[0])
Sp[-1] = -2*numpy.copy(DA[-1])

#souce term Su
Su = heatSourcePerVol*cellVolume

#source term on left & right boundary
Su[0] = Su[0] + 2*tempLeft*numpy.copy(DA[0])
Su[-1] = Su[-1] + 2*tempRight*numpy.copy(DA[-1])

#left coefficient aL
aL = numpy.copy(DA[0:-1])

#right coefficient aR
aR = numpy.copy(DA[0:-1])

aL[0]  = 0
aR[-1] = 0

#central coeff Ap
aP = numpy.copy(aL) + numpy.copy(aR) - numpy.copy(Sp)

#creat matrix

print ('------------------------------------------------')
print (' Assembling Matrices')
print ('------------------------------------------------')


# creat empty matrix A & empty vector B
Amatrix = numpy.zeros([nCells, nCells])
BVector = numpy.zeros(nCells)

for i in range(nCells):

    #left boundary
    if i == 0:
        Amatrix[i,i]   =    aP[i]
        Amatrix[i,i+1] = -1*aR[i]

    #right boundary
    elif i == nCells - 1:
        Amatrix[i,i-1]   = -1*aL[i]
        Amatrix[i,i]     =    aP[i]

    #interior cells
    else:

        Amatrix[i,i-1] = -1*aL[i]
        Amatrix[i,i]   =    aP[i]
        Amatrix[i,i+1] = -1*aR[i]

    BVector[i] = Su[i]
#Print the setup

if printSetup == 'true':

    print('aL:',aL)
    print('aR:',aR)
    print('aP:',aP)
    print('Sp:',Sp)
    print('Su:',Su)
    print('Amatrix:')
    print(Amatrix)
    print('BVector:',numpy.around(BVector,decimals=2))

for i in range(nCells):
        print('%4i %5.1f %5.1f %5.1f %7.1f %5.1f' %(i+1, aL[i], aR[i], Sp[i], Su[i], aP[i]))

#Solve the matrices

print ('------------------------------------------------')
print (' Solving ...')
print ('------------------------------------------------')

# Use the built-in python solution module 
Tvector = numpy.linalg.solve(Amatrix, BVector)

print ('------------------------------------------------')
print (' Equations Solved')
print ('------------------------------------------------')

#print the result

if printSolution == 'true':
    print ('------------------------------------------------')
    print (' Solution: Temperature Vector')
    Tvector = numpy.around(Tvector, decimals = 2)
    print(Tvector)

#plot result
    
if (plotOutput  =='true'):

    print ('------------------------------------------------')
    print (' Plotting ...')
    print ('------------------------------------------------')


    #append the boundary temperature & interior temperature
    #plot the solution
    xPlotting = numpy.hstack([xFace[0], numpy.copy(xCentroid), xFace[-1]])
    tempPlotting = numpy.hstack([tempLeft, numpy.copy(Tvector), tempRight])

    #assemble analytical solution for comparison
    xAnalytical = numpy.linspace(0, barLength, 100)
    tempAnalytical = (tempLeft + ((tempRight-tempLeft)*(xAnalytical/barLength)) +
                      (heatSourcePerVol/(2*cond))*xAnalytical*(barLength*numpy.ones(len(xAnalytical)) - xAnalytical))

   # Configure the plot to look how you want
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

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["figure.figsize"] = (6.2,4.2)
    plt.rcParams['axes.linewidth'] = lineWidth

    fig1 = plt.figure()
    ax = fig1.add_subplot()
    fig1.tight_layout(pad=boxPad)
    ax.plot(xPlotting , tempPlotting, 'bo',linewidth = lineWidth, label='CFD', color=darkRed)
    ax.plot(xAnalytical, tempAnalytical, 'k--',linewidth = lineWidth, label = 'Analytical', color=darkBlue)
    plt.xlabel(r'$x$ [m]', fontsize=fontSize, labelpad = labelPadX)
    plt.ylabel(r'$T$ [$^{\circ}$ C]', fontsize=fontSize, labelpad = labelPadY)
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
    

    






    







