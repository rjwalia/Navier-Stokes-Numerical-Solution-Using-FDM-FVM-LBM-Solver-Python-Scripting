import numpy
import matplotlib.pyplot as plt

print("\n")
print("Finite Volume Method\n")
print("Solving 1D Heat Diffusion Equation with Dirichlet Boundary Condition\n")
print("Discretization for Diffusion Term: Central Difference Scheme\n")

cond = 100             
print("Conductivity of the Material:",cond,'W/m-K')

area = 0.1              
print("Cross Section Area of Rod:",area,'m2')

barLength = 5          
print("\nLength of the rod:",barLength,'m')

#Dirichlet BC

tempLeft = 100              
tempRight = 200             
print("Temperature at the Left End of the Rod:",tempLeft,'C')
print("Temperature at the Right End of the Rod:",tempRight,'C')

heatSourcePerVol = 1000     
print("Heat Source in the Rod:",heatSourcePerVol,'W/m3')
print("\n")

nCells = int(input('Enter the number of Cells for Meshing the Rod: '))   #Number of Cells

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
    
print('aL:',aL)
print('aR:',aR)
print('aP:',aP)
print('Sp:',Sp)
print('\nSu:',Su)
print('\nAmatrix:')
print(Amatrix)
print('\nBVector:',numpy.around(BVector,decimals=2))
print('\n')

print ('------------------------------------------------')
print ('Printing Cell Summary..')
print ('------------------------------------------------')

print('          aL    aR    Sp    Su     aP')
for i in range(nCells):
        print('Cell%2i %5.1f %5.1f %5.1f %7.1f %5.1f' %(i+1, aL[i], aR[i], Sp[i], Su[i], aP[i]))

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


print ('------------------------------------------------')
print (' Solution: Temperature Field')
Tvector = numpy.around(Tvector, decimals = 2)
print(Tvector)

#plot result
    


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

#Configure the plot to look how you want
tickPad = 8
tickPad2 = 16
labelPadY = 1
labelPadX = 2
boxPad = 5
darkBlue = (0.0,0.129,0.2784)
darkRed = (0.7176, 0.0705, 0.207)


plt.rc('font', family='serif')
plt.rcParams["figure.figsize"] = (6.2,4.2)
plt.rcParams['axes.linewidth'] = 2

fig1 = plt.figure()
ax = fig1.add_subplot()
fig1.tight_layout(pad=boxPad)

ax.plot(xPlotting , tempPlotting, 'bo',linewidth = 2, label='CFD', color=darkRed)
ax.plot(xAnalytical, tempAnalytical, 'k--',linewidth = 2, label = 'Analytical', color=darkBlue)

plt.xlabel(r'$x$ [m]', fontsize=14, labelpad = labelPadX)
plt.ylabel(r'$T$ [$^{\circ}$C]', fontsize=14, labelpad = labelPadY)

plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)
plt.xlim([xFace[0], xFace[-1]])
leg = plt.legend(fontsize = 14, loc='best', edgecolor = 'r')
leg.get_frame().set_linewidth(2)
plt.show()
