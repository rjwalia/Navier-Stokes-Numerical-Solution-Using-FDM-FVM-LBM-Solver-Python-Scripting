import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print("\n")
print("Finite Volume Method\n")
print("Solving 2D Heat Diffusion Equation with Dirichlet Boundary Condition\n")
print("Discretization for Diffusion Term: Central Difference Scheme\n")

conductivity     = 100
print("Conductivity of the Material:",conductivity,'W/m-K')

thickness        = 0.1
print("Thickness of the Plate:",thickness ,'m2')

plateLength      = 4
print("Length of the Plate:",plateLength ,'m2')

plateWidth       = 4
print("Width of the Plate:",plateWidth ,'m2')

tempLeft         = 100
tempBottom       = 150
tempRight        = 200
tempTop          = 250
print("Temperature at the Left End of the Rod:",tempLeft,'C')
print("Temperature at the Bottom End of the Rod:",tempBottom,'C')
print("Temperature at the Right End of the Rod:",tempRight,'C')
print("Temperature at the Top End of the Rod:",tempTop,'C')

heatSourcePerVol = 1000     
print("Heat Source in the Rod:",heatSourcePerVol,'W/m3')
print("\n")

lineSingle = '------------------------------------------------'
lineDouble = '================================================'

nCellsLength     = int(input('Enter the number of Cells along the length for Meshing the Plate: '))
nCellsWidth      = int(input('Enter the number of Cells along the width for Meshing the Plate: '))

print(lineDouble)
print('Creating Mesh')
print(lineSingle)

#Total No. of Cells & Cell Faces
numCells = nCellsLength*nCellsWidth
numFaces = (nCellsLength+1)*(nCellsWidth+1)

#coordinate of faces(x,y)

xFacesPattern = np.linspace(0, plateLength, nCellsLength+1)
yFacesPattern = np.linspace(0, plateWidth, nCellsWidth+1)
xFaces = np.tile(xFacesPattern, nCellsWidth+1)
yFaces = np.repeat(yFacesPattern, nCellsLength+1)


#coordinate of centroids(x,y)

xCentroidPattern = 0.5*(xFacesPattern[1:] + xFacesPattern[0:-1])
yCentroidPattern = 0.5*(yFacesPattern[1:] + yFacesPattern[0:-1])
xCentroid = np.tile(xCentroidPattern,nCellsWidth+1)
yCentroid = np.repeat(yCentroidPattern,nCellsLength+1)


# Distance between the cell centroids and the boundaries

dLeftBoundary      = 2*(xCentroid[0] - xFacesPattern[0])
dRightBoundary     = 2*(xFacesPattern[-1] - xCentroid[-1])
dTopBoundary       = 2*(yCentroid[0] - yFacesPattern[0])
dBottomBoundary    = 2*(yFacesPattern[-1] - yCentroid[-1])


# Assemble the distance vectors

dLeftPattern   = np.hstack([dLeftBoundary, xCentroidPattern[1:] - xCentroidPattern[0:-1]])
dRightPattern  = np.hstack([xCentroidPattern[1:] - xCentroidPattern[0:-1], dRightBoundary])
dBottomPattern = np.hstack([yCentroidPattern[1:] - yCentroidPattern[0:-1], dBottomBoundary])
dTopPattern    = np.hstack([dTopBoundary, yCentroidPattern[1:] - yCentroidPattern[0:-1]])

dLeft   = np.tile(dLeftPattern, nCellsWidth)
dRight  = np.tile(dRightPattern, nCellsWidth)
dBottom = np.repeat(dBottomPattern, nCellsLength)
dTop    = np.repeat(dTopPattern, nCellsLength)

CellLength = plateLength/nCellsLength
CellWidth  = plateWidth/nCellsWidth
CellVolume = CellLength*CellWidth*thickness    #cellvolume

areaX = CellWidth*thickness
areaY = CellLength*thickness

# Identify the cells which have boundary faces. Give them an ID of 1. 

topBoundaryID   = np.hstack([np.ones(nCellsLength), np.tile(np.zeros(nCellsLength), nCellsWidth-1)])
bottomBoundaryID= np.hstack([np.tile(np.zeros(nCellsLength), nCellsWidth-1), np.ones(nCellsLength)])
leftBoundaryID  = np.tile(np.hstack([1, np.zeros(nCellsLength-1)]), nCellsWidth)
rightBoundaryID = np.tile(np.hstack([np.zeros(nCellsLength-1), 1]), nCellsWidth)

print(' Calculating Matrix Coefficients')
print(lineSingle)

#diffusive flux per unit area
DA_Left   = np.divide(conductivity*areaX, dLeft)  
DA_Right  = np.divide(conductivity*areaX, dRight)
DA_Top    = np.divide(conductivity*areaY, dTop)
DA_Bottom = np.divide(conductivity*areaY, dBottom)

Su = CellVolume*np.ones(numCells)*heatSourcePerVol

# Add the contribution from each of the boundary faces
Su += 2*tempLeft*np.multiply(leftBoundaryID, DA_Left)
Su += 2*tempRight*np.multiply(rightBoundaryID, DA_Right)
Su += 2*tempTop*np.multiply(topBoundaryID, DA_Top)
Su += 2*tempBottom*np.multiply(bottomBoundaryID, DA_Bottom)

# The source term is zero for interior cells
Sp = np.zeros(numCells)

# Add the contribution from each of the boundary faces

Sp += -2*DA_Left*leftBoundaryID
Sp += -2*DA_Right*rightBoundaryID
Sp += -2*DA_Top*topBoundaryID
Sp += -2*DA_Bottom*bottomBoundaryID

# Only add contributions for interior cells

aL = np.multiply(DA_Left, 1 - leftBoundaryID)
aR = np.multiply(DA_Right,1 - rightBoundaryID)
aT = np.multiply(DA_Top, 1 - topBoundaryID)
aB = np.multiply(DA_Bottom, 1 - bottomBoundaryID)

#central coeff Ap
aP = aL + aR + aT + aB - Sp

print(' Summary: Set Up')
print(lineSingle)
print('Cell |  aL |  aR |  aB |  aT |  Sp |   Su |   aP ')
print(lineSingle)
for i in range(numCells):
    print('%4i %5.0f %5.0f %5.0f %5.0f %5.0f %7.0f %5.0f' %(i+1, aL[i], aR[i], aB[i], aT[i], Sp[i], Su[i], aP[i]))
print(lineSingle)


print(' Assembling Matrices')
print(lineSingle)

Amatrix = np.zeros([numCells, numCells])
Bvector = np.zeros(numCells)

for i in range(numCells):

    Amatrix[i,i]    = aP[i]
    Bvector[i]      = Su[i]

    if leftBoundaryID[i] == 0:
        Amatrix[i,i-1] = -aL[i]

    if rightBoundaryID[i] == 0:
        Amatrix[i,i+1] = -aR[i]

    if bottomBoundaryID[i] == 0:
        Amatrix[i, i+nCellsLength] = -aB[i]

    if topBoundaryID[i] == 0:
        Amatrix[i, i-nCellsLength] = -aT[i]

np.set_printoptions(linewidth=85)
print(Amatrix)
print(lineSingle)

print('Solving ...')
print(lineSingle)

Tvector = np.linalg.solve(Amatrix, Bvector)
Tvector = np.around(Tvector, decimals = 3)

print('Equations Solved')
print(lineSingle)

Tgrid = Tvector.reshape(nCellsWidth, nCellsLength)

print ('\nSolution: Temperature Field')
print(lineSingle)
print(Tgrid)
print(lineSingle)

#Heat Fluxes
# Calculate the temperature differences
# - To do this we need to stack on the boundary temperatures onto the grid

Tleftrightshift  = np.hstack([tempLeft*np.ones([nCellsWidth,1]), Tgrid,
                            tempRight*np.ones([nCellsWidth,1])])
Ttopbottomshift = np.vstack([tempTop*np.ones([nCellsLength]), Tgrid,
                             tempBottom*np.ones([nCellsLength])])

# Now we can calculate the temperature differences

deltaTleft   = Tleftrightshift[:,1:-1] - Tleftrightshift[:,0:-2]
deltaTright  = Tleftrightshift[:,2:] - Tleftrightshift[:,1:-1]
deltaTtop    = Ttopbottomshift[0:-2,:] - Ttopbottomshift[1:-1,:]
deltaTbottom = Ttopbottomshift[1:-1,:] - Ttopbottomshift[2:,:]

# We now need to calculate the diffusive heat flux (DA) on each face
# - Start by reshaping the DA vectors into a grid of the correct size

DA_left_grid   = DA_Left.reshape(nCellsWidth, nCellsLength)
DA_right_grid  = DA_Right.reshape(nCellsWidth, nCellsLength)
DA_top_grid    = DA_Top.reshape(nCellsWidth, nCellsLength)
DA_bottom_grid = DA_Bottom.reshape(nCellsWidth, nCellsLength)

# Calculate the boundary face fluxes

DA_left_boundary   = (2*conductivity*areaX/dLeftBoundary)*np.ones([nCellsWidth,1])
DA_right_boundary  = (2*conductivity*areaX/dRightBoundary)*np.ones([nCellsWidth,1])
DA_top_boundary    = (2*conductivity*areaY/dTopBoundary)*np.ones([nCellsLength])
DA_bottom_boundary = (2*conductivity*areaY/dBottomBoundary)*np.ones([nCellsLength])

# Now stack on the boundary face fluxes to the grid

DA_left_shift   = np.hstack([DA_left_boundary, DA_left_grid[:,1:]])
DA_right_shift  = np.hstack([DA_right_grid[:,:-1], DA_right_boundary])
DA_top_shift    = np.vstack([DA_top_boundary, DA_top_grid[1:,:]])
DA_bottom_shift = np.vstack([DA_bottom_grid[0:-1,:],DA_bottom_boundary])

#unit normals

normalsLeftGrid   = -1*np.ones([nCellsWidth, nCellsLength])
normalsRightGrid  =  1*np.ones([nCellsWidth, nCellsLength])
normalsBottomGrid = -1*np.ones([nCellsWidth, nCellsLength]) 
normalsTopGrid    =  1*np.ones([nCellsWidth, nCellsLength])

#calculating heat flux across faces

heatFluxLeft   = -np.multiply(np.multiply(DA_left_shift,deltaTleft),normalsLeftGrid)
heatFluxRight  = -np.multiply(DA_right_shift,deltaTright,normalsRightGrid)
heatFluxTop    = -np.multiply(DA_top_shift,deltaTtop,normalsTopGrid)
heatFluxBottom = -np.multiply(np.multiply(DA_bottom_shift,deltaTbottom),normalsBottomGrid)

#calculating vol heat generation in each cell

sourceVol = heatSourcePerVol*CellVolume*np.ones([nCellsWidth, nCellsLength])

# Calculate the error in the heat flux balance in each cell

error = (sourceVol - heatFluxLeft - heatFluxRight - heatFluxTop - heatFluxBottom)

heatFluxLeftVector   = heatFluxLeft.flatten()
heatFluxRightVector  = heatFluxRight.flatten()
heatFluxTopVector    = heatFluxTop.flatten()
heatFluxBottomVector = heatFluxBottom.flatten()
sourceVolVector      = sourceVol.flatten()
errorVector          = error.flatten()



print('\nHeat Balance:')
print(lineSingle)
print('Cell |  QL  |   QR  |   QT  |   QB   |  SV   |   Err')
print(lineSingle)
for i in range(numCells):
    print('%4i %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f' % (i+1, heatFluxLeftVector[i],
        heatFluxRightVector[i], heatFluxTopVector[i],
        heatFluxBottomVector[i], sourceVolVector[i], errorVector[i]))
    print(lineSingle)

#Sum the heat fluxes across the boundary faces to give the total heat flux
#across each boundary

heatFluxLeftBoundaryTotal   = np.sum(np.multiply(leftBoundaryID, heatFluxLeftVector))
heatFluxRightBoundaryTotal  = np.sum(np.multiply(rightBoundaryID, heatFluxRightVector))
heatFluxBottomBoundaryTotal = np.sum(np.multiply(bottomBoundaryID, heatFluxBottomVector))
heatFluxTopBoundaryTotal    = np.sum(np.multiply(topBoundaryID, heatFluxTopVector))

heatFluxBoundaryTotal = (heatFluxLeftBoundaryTotal + heatFluxRightBoundaryTotal
                         + heatFluxBottomBoundaryTotal + heatFluxTopBoundaryTotal)
heatGenerationTotal   = np.sum(sourceVolVector)

print ('Boundary Heat Flux')
print(lineSingle)
print('Left      :  %5.1f [W]'%(heatFluxLeftBoundaryTotal))
print('Right     :  %6.1f [W]'% heatFluxRightBoundaryTotal)
print('Bottom    :  %6.1f [W]'% heatFluxBottomBoundaryTotal)
print('Top       : %6.1f [W]'% heatFluxTopBoundaryTotal)
print('Total     : %7.1f [W]'% heatFluxBoundaryTotal)
print(lineSingle)
print('Generated : %7.1f [W]'% heatGenerationTotal)
print(lineSingle)
print('Error     : %7.1f [W]'% (heatFluxBoundaryTotal - heatGenerationTotal))
print(lineSingle)

# Interpolate the solution on the interior nodes from the CFD solution

temperatureTopLeftCorner     = 0.5*(tempTop + tempLeft)
temperatureTopRightCorner    = 0.5*(tempRight + tempTop)
temperatureBottomLeftCorner  = 0.5*(tempBottom + tempLeft)
temperatureBottomRightCorner = 0.5*(tempRight + tempBottom)

# Interpolate the solution on the interior nodes from the CFD solution

Tleftrightnodes = 0.5*(Tgrid[:,1:]+Tgrid[:,:-1])
Tinternalnodes = 0.5*(Tleftrightnodes[1:,:] + Tleftrightnodes[:-1,:])


# Assemble the temperatures on all the boundary nodes

temperatureTopVector    = np.hstack([temperatureTopLeftCorner,
                         tempTop*np.ones(nCellsLength-1),temperatureTopRightCorner])

temperatureBottomVector = np.hstack([temperatureBottomLeftCorner,
                         tempBottom*np.ones(nCellsLength-1),temperatureBottomRightCorner])

temperatureLeftVector   = tempLeft*np.ones([nCellsWidth-1,1])
temperatureRightVector  = tempRight*np.ones([nCellsWidth-1,1])


Tnodes = np.vstack([temperatureTopVector, np.hstack([temperatureLeftVector,
         Tinternalnodes,temperatureRightVector]), temperatureBottomVector])
Tnodes = np.around(Tnodes, decimals = 2)

print ('Solution: Temperature Field on nodes')
print(lineSingle)
print(Tnodes)
print(lineSingle)

xNodes = xFaces.reshape([nCellsWidth+1, nCellsLength+1])
yNodes = np.flipud(yFaces.reshape([nCellsWidth+1, nCellsLength+1]))

#Plot the solution

# Plot the data if desired

print ('Plotting ...')
print(lineSingle)

tickPad = 8
tickPad2 = 16
labelPadY = 3
labelPadX = 2
boxPad = 2
tickLength = 4
markerSize = 4

# Colours - Can use rgb or html
lightBlue = '#bfc8d1'
shadeBlue = '#8091a4'
darkBlue = '#002147'   

# Use 'CMU sans-serif' font in the plots. 
plt.rc('font', family='serif')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams["figure.figsize"] = (3.1,2.5)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
fig1.tight_layout(pad=boxPad)
cmap_reversed = cm.get_cmap('autumn_r')
CS = ax.contourf(xNodes, yNodes, Tnodes, cmap=cmap_reversed)
CS2 = ax.contour(CS, colors='k', linewidth=1.5)
ax.set_xlabel(r'$x$ [m]', fontsize=11, labelpad = labelPadX)
ax.set_ylabel(r'$y$ [m]', fontsize=11, labelpad = labelPadY)
ax.set_title('Temperature Contour')
plt.yticks(np.arange(0,plateLength+1,1), fontsize = 11)
plt.xticks(np.arange(0,plateWidth+1,1), fontsize = 11)
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Temperature [C]', fontsize=11, labelpad = labelPadX)
cbar.ax.tick_params(size = 0, width = 1.5)
cbar.add_lines(CS2)
cbar.ax.tick_params(labelsize=11)
ax.tick_params(which = 'both', direction='in', length=6,width=1.5, gridOn = False, pad=tickPad)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
plt.show()
