###############################################################################
#
# Postprocessor script for ALPACA outputs using Paraview
# Created by Daniel Nagy (nagyd@edu.bme.hu)
# Date: 27/04/2026
# Version: DBv1.0
# Required paraview version: 5.11.2
# HDS Sonochemistry Research Group
#
###############################################################################
#
# Usage:
# use with pvpython
# Argument 1     : String of the input file folder (domain folder)
# Argument 2     : String of the output file where the png-s are saved
# Argument 3     : Frequency of the sound wave
# Argument 4     : Pressure amplitude of the sound wave
# Argument 5     : Radius of the bubble at rest
# --debug        : Shows debug information
# --help         : Prints out the help menu
#
###############################################################################

import os
from os import listdir
from os.path import isfile, join
from sys import exit
import time
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import eval_legendre
from paraview.simple import *

print("----------------------------------------------------------------------")
print("                Start of the postprocessor code")
print("----------------------------------------------------------------------")

#CONSTANTS
pi = 3.141592653589793238462
c0 = 1482.0 #speed of sound in water at 20 degrees Celsius
rho0 = 998.2 #density of water at 20 degrees Celsius
p0 = 101325.0 #pressure of water at 20 degrees Celsius

#PARAMETERS
volume_calculation_phi_threshold = 0.0 #threshold used for volume calculation
modes_calculation_phi_lower = -1.5 #threshold used for mode calculation
modes_calculation_phi_upper = 1.5 #threshold used for mode calculation
calculate_with_geometric_y0 = True #if true, the bubble center is calculated and used
max_mode_number = 10 #maximum mode to be calculated
points_vertical_line_full = 1000 #number of points in the vertical line for the full domain
points_horizontal_line_full = 500 #number of points in the horizontal line for the full domain
points_vertical_line = 1000 #number of points in the vertical line for the near bubble domain
points_horizontal_line = 500 #number of points in the horizontal line for the near bubble domain
saveDomainSize = 8 #size of the domain to be saved around the bubble (in bubble radii)
saveVars = ['pressure','density','velocity_X','velocity_Y','levelset']
line_header = "coord[m],pressure[Pa],density[kg/m3],velocity_X[m/s],velocity_Y[m/s],levelset[1]"
velScaleFull = [0.001,1.0]
velScaleSub  = [0.001,1000.0]
presScaleSub = [1e4,1e7]

#USER INTERFACE
parser = argparse.ArgumentParser(description='Calculates important data from bubble simulations')
parser.add_argument('inputfolder', metavar='inp', type=str, help='Input folder')
parser.add_argument('outputfolder', metavar='out', type=str, help='Output file name')
parser.add_argument('frequency', metavar='freq', type=float, help='Frequency of the sound wave')
parser.add_argument('pressure_amplitude', metavar='pamp', type=float, help='Pressure amplitude of the sound wave')
parser.add_argument('radius', metavar='rad', type=float, help='Radius of the bubble at rest')
parser.add_argument('--debug',action='store_const', default=0, const=1, help='Debugger (1) on, (0) off')

#USER ARGUMENTS
args = parser.parse_args()
debug = args.debug

#CALCULATE VARS
x0 = 0.0
f = args.frequency
pA = args.pressure_amplitude
lamb = c0 / f
R0 = args.radius
y0 = lamb + R0

print("Lambda: ",lamb,"m")
print("y0: ",y0,"m")

#INPUT AND OUTPUT FOLDERS
workingDirectory = os.getcwd()
inputFolder = workingDirectory + '/' +  args.inputfolder + '/subdomain'
inputFolderFull = workingDirectory + '/' +  args.inputfolder + '/domain'
outputFile = workingDirectory + '/' + args.outputfolder + '/bubble_data.csv'
outputFileDetailed = workingDirectory + '/' + args.outputfolder + '/bubble_data_highres.csv'
outputFileDetailed2 = workingDirectory + '/' + args.outputfolder + '/bubble_data_medres.csv'
outputFileFull = workingDirectory + '/' + args.outputfolder + '/domain_data.csv'
print("Input folder1: \n   ",inputFolder)
print("Input folder2: \n   ",inputFolderFull)
print("Output folder: \n   ",args.outputfolder)
print("Output file  : \n   ",outputFile)

if not os.path.exists(args.outputfolder):
    os.makedirs(args.outputfolder)
    if debug:
        print("Output folder created\n")


#OUTPUT FOLDER FOR PROFILES
folderVerticalLineFull = workingDirectory + '/' + args.outputfolder + '/profile_y_full'
if not os.path.exists(folderVerticalLineFull):
    os.makedirs(folderVerticalLineFull)
folderHorizontalLineFull = workingDirectory + '/' + args.outputfolder + '/profile_x_full'
if not os.path.exists(folderHorizontalLineFull):
    os.makedirs(folderHorizontalLineFull)
folderVerticalLine = workingDirectory + '/' + args.outputfolder + '/profile_y_sub'
if not os.path.exists(folderVerticalLine):
    os.makedirs(folderVerticalLine)
folderHorizontalLine = workingDirectory + '/' + args.outputfolder + '/profile_x_sub'
if not os.path.exists(folderHorizontalLine):
    os.makedirs(folderHorizontalLine)
folderSubdomain = workingDirectory + '/' + args.outputfolder + '/images_sub'
if not os.path.exists(folderSubdomain):
    os.makedirs(folderSubdomain)
folderFulldomain = workingDirectory + '/' + args.outputfolder + '/images_full'
if not os.path.exists(folderFulldomain):
    os.makedirs(folderFulldomain)


print("----------------------------------------------------------------------")
print("     Processing of the monitor quantities file...")
print("----------------------------------------------------------------------")

#MONITOR QUANTITIES FILE
monitorQuantitiesFile = list(Path(workingDirectory + '/' +  args.inputfolder).glob("*_monitor_quantities.csv"))[0]
print("Monitored quant.: ",monitorQuantitiesFile)

df = pd.read_csv(monitorQuantitiesFile, skipinitialspace=True)
df.columns = df.columns.str.strip()
time_array = df["time"].to_numpy()
volume_array = df["total_volume0"].to_numpy()
total_mass = df["total_mass"].to_numpy()
total_mass = total_mass/total_mass[0]
radius_array = (3.0 * volume_array / (4*pi))**(1.0/3.0)
out_array = np.column_stack((time_array, radius_array, volume_array, total_mass))
out_array2 = np.column_stack((time_array[::20], radius_array[::20], volume_array[::20], total_mass[::20]))
np.savetxt(
    outputFileDetailed,
    out_array,
    delimiter=",",
    header="time[s],radius[m],volume[m3],mass[1]",
    comments=""
)
np.savetxt(
    outputFileDetailed2,
    out_array2,
    delimiter=",",
    header="time[s],radius[m],volume[m3],mass[1]",
    comments=""
)

resMassEnd = total_mass[-1]

#find min/max in monitor quantities
resMaxT = 0.0
resMaxR = R0
resMaxM = 1.0
resMinT = 0.0
resMinR = R0
resMinM = 1.0
resMassMax = 1.0
resMassMin = 1.0
for i in range(np.size(time_array)):  
    if radius_array[i] > resMaxR:
        resMaxR = radius_array[i]
        resMaxT = time_array[i]
        resMaxM = total_mass[i]
    if radius_array[i] < resMinR:
        resMinR = radius_array[i]
        resMinT = time_array[i]
        resMinM = total_mass[i]
    if total_mass[i] > resMassMax:
        resMassMax = total_mass[i]
    if total_mass[i] < resMassMin:
        resMassMin = total_mass[i]

print("Max: t=",resMaxT,"\t R=",resMaxR,"\t M=",resMaxM)
print("Min: t=",resMinT,"\t R=",resMinR,"\t M=",resMinM)


#READ ALL STEPS
files = [f for f in listdir(inputFolder) if isfile(join(inputFolder, f))]
files = sorted([inputFolder+'/'+i for i in files if '.xdmf' in i and 'time_series' not in i])
length = len(files)

filesFull = [f for f in listdir(inputFolderFull) if isfile(join(inputFolderFull, f))]
filesFull = sorted([inputFolderFull+'/'+i for i in filesFull if '.xdmf' in i and 'time_series' not in i])
lengthFull = len(filesFull)

if debug:
    print("First file: \n   ",files[0])
    print("Number of files:\n   ",length)
    print("First file (full): \n   ",filesFull[0])
    print("Number of files (full):\n   ",lengthFull)

#APPLY THE NECESSARY FILTERS FOR THE FULL DOMAIN
listOfStepsFull = XDMFReader(registrationName='allsteps', FileNames=filesFull)
timestepsFull = listOfStepsFull.TimestepValues
if debug:
    print("First timestep (Full) :",timestepsFull[0])
    print("Last timestep (Full) :",timestepsFull[-1])

# get animation scene
animationFull = GetAnimationScene()
animationFull.UpdateAnimationUsingDataTimeSteps()

# create a new 'Slice'
slice1Full = Slice(registrationName='Slice1Full', Input=listOfStepsFull)
slice1Full.SliceType.Origin = [0.0, 0.0, 1e-15]
slice1Full.SliceType.Normal = [0.0, 0.0, 1.0]

# find maximum x value
UpdatePipeline(time=timestepsFull[0], proxy=slice1Full)
slice1Data = paraview.servermanager.Fetch(slice1Full)
slice1Points = slice1Data.GetPoints()
maxX = 0.0
maxY = 0.0
for i in range(slice1Points.GetNumberOfPoints()):
    x = slice1Points.GetPoint(i)[0]
    y = slice1Points.GetPoint(i)[1]
    if x > maxX:
        maxX = x
    if y > maxY:
        maxY = y
volumeDomain = 2*pi*maxX*lamb
if debug:
    print("Maximum x, y values in the slice:",maxX, maxY)
    print("Volume of the domain:",volumeDomain)

# create a new 'Slice'
slice2Full = Clip(registrationName='Slice2Full', Input=slice1Full)
slice2Full.ClipType = 'Box'
slice2Full.ClipType.Position = [0.0, 0.0, 0.0]
slice2Full.ClipType.Length = [maxX, lamb, 0.1]

# create a new 'Cell Data to Point Data'
cellDatatoPointData1Full = CellDatatoPointData(registrationName='CellDatatoPointData1Full', Input=slice2Full)
cellDatatoPointData1Full.ProcessAllArrays = 0
cellDatatoPointData1Full.CellDataArraytoprocess = ['pressure', 'density','velocity']

# Calculate the energy of the pulse
calculator1Full = Calculator(registrationName='Calculator1Full', Input=cellDatatoPointData1Full)
calculator1Full.ResultArrayName = 'E_int'
calculator1Full.Function = ' ( (' + str(p0) + '- pressure)*(' + str(p0) + '- pressure)/'+str(2*rho0*c0*c0)+ ' + 0.5*'+ str(rho0) + '*(velocity_X*velocity_X + velocity_Y*velocity_Y) ) * coordsX'
if debug:
    print("Energy function: ",calculator1Full.Function)

# create a new 'Point Data to Cell Data'
pointDatatoCellData1Full  = PointDatatoCellData(registrationName='PointDatatoCellData1Full', Input=calculator1Full)
pointDatatoCellData1Full.ProcessAllArrays = 0
pointDatatoCellData1Full.PointDataArraytoprocess = ['E_int']

# Integrate over the bubble
integratedFull= IntegrateVariables(registrationName='IntegrateVariables1Full', Input=pointDatatoCellData1Full)
integratedFull.DivideCellDataByVolume = 0 

# Create line to save data
verticalLine1Full = PlotOverLine(registrationName='PlotOverLine1', Input=listOfStepsFull)
verticalLine1Full.Point1 = [0.0, 0.0, 1e-15]
verticalLine1Full.Point2 = [0.0, maxY-1e-8, 1e-15]
verticalLine1Full.SamplingPattern = 'Sample At Segment Centers'

# Create line to save data
horizontalLine1Full = PlotOverLine(registrationName='PlotOverLine2', Input=listOfStepsFull)
horizontalLine1Full.Point1 = [0.0, y0, 1e-15]
horizontalLine1Full.Point2 = [maxX-1e-8, y0, 1e-15]
horizontalLine1Full.SamplingPattern = 'Sample At Segment Centers'

#PREPARE FOR PNG OUTPUT
renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.InteractionMode = '2D'
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.OrientationAxesVisibility = 0

fullDisplay = Show(listOfStepsFull,renderView1,'UnstructuredGridRepresentation')
ColorBy(fullDisplay,('Cells','pressure'))

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=listOfStepsFull)
transform1.Transform = 'Transform'
transform1.Transform.Scale = [-1.0, 1.0, 1.0]

transform1Display = Show(transform1,renderView1,'UnstructuredGridRepresentation')
ColorBy(transform1Display,('Cells','velocity','Magnitude'))
transform1Display.RescaleTransferFunctionToDataRange(True, False)

# create a new 'Line'
line1 = Line(registrationName='Line1')
line1.Point1 = [0.0, 0.0, 1e-3]
line1.Point2 = [0.0, maxY, 1e-3]

colorsP = GetColorTransferFunction('pressure')
colorsV = GetColorTransferFunction('velocity')

colorsP.RescaleTransferFunction(p0-pA,p0+pA)
colorsV.UseLogScale = 1
colorsV.MapControlPointsToLogSpace()
colorsV.RescaleTransferFunction(velScaleFull[0],velScaleFull[1])
colorsV.ApplyPreset('X Ray',True)

line1Display = Show(line1, renderView1, 'GeometryRepresentation')
line1Display.LineWidth = 2.0
line1Display.AmbientColor = [0.0, 0.0, 0.0]
line1Display.DiffuseColor = [0.0, 0.0, 0.0]

renderView1.ResetCamera(True)
renderView1.Update()

layout1 = GetLayout()
layout1.SetSize(1600, 1600)

#CREATE FILE TO SAVE DATA
file = open(outputFileFull,'w')
file.write("time[s],acoustic_energy[J]\n")

def GetInterpolatedArray(linePoints, arrays, NpointsInterp):
    Npoints = lineData.GetNumberOfPoints()
    Narrays = len(arrays)
    line_raw = np.zeros((Npoints,1 + Narrays))
    line_inp = np.zeros((NpointsInterp,1 + Narrays))

    for j in range(Npoints):
        line_raw[j,0] = linePoints.GetArray('arc_length').GetValue(j)
        for k in range(Narrays):
            if arrays[k] == 'velocity_X':
                line_raw[j,k+1] = linePoints.GetArray('velocity').GetComponent(j,0)
            elif arrays[k] == 'velocity_Y':
                line_raw[j,k+1] = linePoints.GetArray('velocity').GetComponent(j,1)
            else:
                line_raw[j,k+1] = linePoints.GetArray(arrays[k]).GetValue(j)
    #clear nan-s
    mask = ~np.isnan(line_raw).any(axis=1)
    line_raw = line_raw[mask]

    max_size = line_raw[-1,0]

    for j in range(NpointsInterp):
        line_inp[j,0] = j/(NpointsInterp-1)*max_size
        for k in range(Narrays):
            line_inp[j,k+1] = np.interp(line_inp[j,0], line_raw[:,0], line_raw[:,k+1])
    return line_inp

print("----------------------------------------------------------------------")
print("                Processing of full domain...")
print("----------------------------------------------------------------------")

E_init = 0.0
E_final = 0.0
t_final = 0.0
E_final_saved = False
counter = 0.0
for i in range(lengthFull):
    animationFull.AnimationTime = timestepsFull[i]

    if counter < i/lengthFull*100:
        print("Progress ",counter,"% t = ",timestepsFull[i])
        counter += 5

    if debug:
        print("Timesteps #",i," t=",timestepsFull[i])
        Show(pointDatatoCellData1Full)

    # Update the box clip
    dy = timestepsFull[i]*c0
    slice2Full.ClipType.Position = [0.0, 0.0 + dy, 0.0]
    slice2Full.ClipType.Length = [maxX, lamb + dy, 0.1]
    
    # Calculate integrated quantities
    UpdatePipeline(time=timestepsFull[i], proxy=integratedFull)
    integrateData = paraview.servermanager.Fetch(integratedFull)
    if timestepsFull[i] > 2.0/f: 
        E_pulse = 0  #After two periods, the pulse should have left the domain, so we set the energy to zero to avoid numerical issues
    else:
        E_pulse = integrateData.GetCellData().GetArray('E_int').GetValue(0)*2*pi
    file.write("{:.15f},{:.15f}\n".format(timestepsFull[i],E_pulse))

    if i == 0:
        E_init = E_pulse
    if (i == lengthFull-1 or timestepsFull[i]> 1.2/f) and not E_final_saved:
        E_final = E_pulse
        t_final = timestepsFull[i]
        E_final_saved = True

    # Calculate line data - vertical line in the full domain
    UpdatePipeline(time=timestepsFull[i], proxy=verticalLine1Full)
    lineData = paraview.servermanager.Fetch(verticalLine1Full)
    linePoints = lineData.GetPointData()
    line_inp = GetInterpolatedArray(linePoints, saveVars, points_vertical_line_full)
    np.savetxt(folderVerticalLineFull+'/step_'+str(i)+'.csv', line_inp, delimiter=',',header=line_header,comments='')

    # Calculate line data - horizontal line in the full domain
    UpdatePipeline(time=timestepsFull[i], proxy=horizontalLine1Full)
    lineData = paraview.servermanager.Fetch(horizontalLine1Full)
    linePoints = lineData.GetPointData()
    line_inp = GetInterpolatedArray(linePoints, saveVars, points_horizontal_line_full)
    np.savetxt(folderHorizontalLineFull+'/step_'+str(i)+'.csv', line_inp, delimiter=',',header=line_header,comments='')

    SaveScreenshot(folderFulldomain + "/step_" + str(i) + ".png",renderView1,ImageResolution=[1600,1600])

file.close()
if debug:
    print("   Filewriters closed")




#APPLY THE NECESSARY FILTERS
# create a new 'XDMF Reader'
listOfSteps = XDMFReader(registrationName='allsteps', FileNames=files)
timesteps = listOfSteps.TimestepValues
if debug:
    print("First timestep :",timesteps[0])
    print("Last timestep  :",timesteps[-1])

# get animation scene
animation = GetAnimationScene()
animation.UpdateAnimationUsingDataTimeSteps()

# Select the bubble  with alpha=0.999
threshold1 = Threshold(registrationName='Threshold1', Input=listOfSteps)
threshold1.Scalars = ['CELLS', 'levelset']
threshold1.LowerThreshold = -100.0
threshold1.UpperThreshold = volume_calculation_phi_threshold

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=threshold1)
slice1.SliceType.Origin = [0.0, 0.0, 1e-15]
slice1.SliceType.Normal = [0.0, 0.0, 1.0]


# create a new 'Cell Data to Point Data'
cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=slice1)
cellDatatoPointData1.ProcessAllArrays = 0
cellDatatoPointData1.CellDataArraytoprocess = ['levelset', 'pressure', 'density']

# Volume - Recalculate the field to account for axisymmetry
calculator1 = Calculator(registrationName='Calculator1', Input=cellDatatoPointData1)
calculator1.ResultArrayName = 'V_int'
calculator1.Function = '2*3.141592653589793238462*coordsX'

# Pressure - Recalculate the field to account for axisymmetry
calculator2 = Calculator(registrationName='Calculator2', Input=calculator1)
calculator2.ResultArrayName = 'p_int'
calculator2.Function = 'pressure*coordsX'

# Density - Recalculate the field to account for axisymmetry
calculator3 = Calculator(registrationName='Calculator3', Input=calculator2)
calculator3.ResultArrayName = 'rho_int'
calculator3.Function = 'density*coordsX'

# Center Volume - Recalculate the field to account for axisymmetry
calculator4 = Calculator(registrationName='Calculator4', Input=calculator3)
calculator4.ResultArrayName = 'CV_int'
calculator4.Function = '2*3.141592653589793238462*coordsX'

# Center on the y-axis 
calculator5 = Calculator(registrationName='Calculator5', Input=calculator4)
calculator5.ResultArrayName = 'y_int'
calculator5.Function = '2*3.141592653589793238462*coordsX * coordsY'

# create a new 'Point Data to Cell Data'
pointDatatoCellData1 = PointDatatoCellData(registrationName='PointDatatoCellData1', Input=calculator5)
pointDatatoCellData1.ProcessAllArrays = 0
pointDatatoCellData1.PointDataArraytoprocess = ['rho_int', 'V_int', 'CV_int', 'p_int','y_int']

# Integrate over the bubble
integrated= IntegrateVariables(registrationName='IntegrateVariables1', Input=pointDatatoCellData1)
integrated.DivideCellDataByVolume = 0 


#APPLY THE NECESSARY FILTERS TO FIND THE MODES

# Select the bubble interface
threshold2 = Threshold(registrationName='Threshold2', Input=listOfSteps)
threshold2.Scalars = ['CELLS', 'levelset']
threshold2.LowerThreshold = modes_calculation_phi_lower
threshold2.UpperThreshold = modes_calculation_phi_upper

# create a new 'Cell Data to Point Data'
cellCenters = CellCenters(registrationName='CellCenters1', Input=threshold2)

#SAVE LINE DATA

# Create line to save data
verticalLine1 = PlotOverLine(registrationName='PlotOverLine1', Input=listOfSteps)
verticalLine1.Point1 = [0.0, y0 - saveDomainSize*R0, 1e-15]
verticalLine1.Point2 = [0.0, y0 + saveDomainSize*R0, 1e-15]
verticalLine1.SamplingPattern = 'Sample At Segment Centers'

# Create line to save data
horizontalLine1 = PlotOverLine(registrationName='PlotOverLine2', Input=listOfSteps)
horizontalLine1.Point1 = [0.0, y0, 1e-15]
horizontalLine1.Point2 = [saveDomainSize*R0, y0, 1e-15]
horizontalLine1.SamplingPattern = 'Sample At Segment Centers'

#FILTERS FOR IMAGES
clipSC1 = Clip(registrationName='ClipSC1', Input=listOfSteps)
clipSC1.ClipType = 'Box'
clipSC1.ClipType.Position = [0, y0-saveDomainSize*R0, 0.0]
clipSC1.ClipType.Length = [saveDomainSize*R0, 2*saveDomainSize*R0, 1.0]
clipSC1.Crinkleclip = 1

Hide(line1,renderView1)
Hide(transform1,renderView1)
Hide(listOfStepsFull,renderView1)

renderView2 = GetActiveViewOrCreate('RenderView')
renderView2.InteractionMode = '2D'
renderView2.UseColorPaletteForBackground = 0
renderView2.Background = [1.0, 1.0, 1.0]
renderView2.OrientationAxesVisibility = 0

subDisplay = Show(clipSC1,renderView2,'UnstructuredGridRepresentation')
ColorBy(subDisplay,('Cells','pressure'))

# create a new 'Transform'
transform2 = Transform(registrationName='Transform2', Input=clipSC1)
transform2.Transform = 'Transform'
transform2.Transform.Scale = [-1.0, 1.0, 1.0]

transform2Display = Show(transform2,renderView2,'UnstructuredGridRepresentation')
ColorBy(transform2Display,('Cells','velocity','Magnitude'))
transform2Display.RescaleTransferFunctionToDataRange(True, False)

# create a new 'Line'
line2 = Line(registrationName='Line2')
line2.Point1 = [0.0, y0-saveDomainSize*R0, 1e-4]
line2.Point2 = [0.0, y0+saveDomainSize*R0, 1e-4]

colorsP.MapControlPointsToLogSpace()
colorsP.UseLogScale = 1
colorsP.RescaleTransferFunction(presScaleSub[0],presScaleSub[1])

"""# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(colorsP, renderView1)
pressureLUTColorBar.Position = [0.7, 0.4]
pressureLUTColorBar.ScalarBarLength = 0.3"""

colorsV.RescaleTransferFunction(velScaleSub[0],velScaleSub[1])
colorsV.MapControlPointsToLogSpace()
colorsV.UseLogScale = 1
colorsV.ApplyPreset('X Ray',True)

# Select the interface=0.999
thresholdSC1 = Threshold(registrationName='Threshold1', Input=clipSC1)
thresholdSC1.Scalars = ['CELLS', 'levelset']
thresholdSC1.LowerThreshold = -1.5
thresholdSC1.UpperThreshold = 1.5
thresholdSC1Display = Show(thresholdSC1,renderView2,'UnstructuredGridRepresentation')
ColorBy(thresholdSC1Display, None)
thresholdSC1Display.AmbientColor = [0.0, 0.0, 0.0]
thresholdSC1Display.DiffuseColor = [0.0, 0.0, 0.0]

# create a new 'Transform'
transform3 = Transform(registrationName='Transform2', Input=thresholdSC1)
transform3.Transform = 'Transform'
transform3.Transform.Scale = [-1.0, 1.0, 1.0]
transform3Display = Show(transform3,renderView2,'UnstructuredGridRepresentation')
#ColorBy(transform3Display, None)
transform3Display.AmbientColor = [0.0, 0.0, 0.0]
transform3Display.DiffuseColor = [0.0, 0.0, 0.0]

line2Display = Show(line2, renderView2, 'GeometryRepresentation')
line2Display.LineWidth = 2.0
line2Display.AmbientColor = [0.0, 0.0, 0.0]
line2Display.DiffuseColor = [0.0, 0.0, 0.0]

renderView2.ResetCamera(True)
renderView2.Update()

layout2 = GetLayout()
layout2.SetSize(1600, 1600)

print("----------------------------------------------------------------------")
print("                Processing of close subdomain...")
print("----------------------------------------------------------------------")


#CREATE FILE TO SAVE DATA
file = open(outputFile,'w')
file.write("time[s],radius[m],volume[m3],center_volume[m3],center_y[m],pressure[Pa],density[kg/m3]")
for n in range(max_mode_number+1):
    file.write(",a"+str(n)+"[m]")
file.write("\n")


counter = 0.0
jet_impact_happened = False
jet_impact_T = 0.0
jet_impact_R = R0
jet_impact_V = 0.0
for i in range(length):
    animation.AnimationTime = timesteps[i]

    if counter < i/length*100:
        print("Progress ",counter,"% t = ",timesteps[i])
        counter += 5

    if debug:
        print("Timesteps #",i," t=",timesteps[i])
        Show(pointDatatoCellData1)

    
    # Calculate integrated quantities
    UpdatePipeline(time=timesteps[i], proxy=integrated)
    integrateBubbleData = paraview.servermanager.Fetch(integrated)
    VB = integrateBubbleData.GetCellData().GetArray('V_int').GetValue(0)
    CVB = integrateBubbleData.GetCellData().GetArray('CV_int').GetValue(0)
    if CVB <= 0.0:
        CVB = -1
        print("WARNING: Center bubble volume is invalid!\n")
    pB = integrateBubbleData.GetCellData().GetArray('p_int').GetValue(0)*2*pi/CVB
    rhoB = integrateBubbleData.GetCellData().GetArray('rho_int').GetValue(0)*2*pi/CVB
    y0B = integrateBubbleData.GetCellData().GetArray('y_int').GetValue(0) / VB
    RB = (3.0/(4.0*pi)*VB)**(1.0/3.0)


    # Calculate modes
    UpdatePipeline(time=timesteps[i], proxy=cellCenters)
    surface_data = paraview.servermanager.Fetch(cellCenters)
    surface_points = surface_data.GetPoints()
    Npoints = surface_points.GetNumberOfPoints()

    #initialize array to save r and phi values
    rPhiVals = np.zeros((Npoints,2))

    #y0 used in calculations
    y0calc = y0B if calculate_with_geometric_y0 else y0
    if debug:
        print("y0_calc:",y0calc,"y0B:",y0B,"y0:",y0)

    for j in range(Npoints):
        x = surface_points.GetPoint(j)[0]
        y = surface_points.GetPoint(j)[1]
        # Polar coordinate transformation
        r = np.sqrt((x-x0)**2+(y-y0calc)**2)
        phi = np.arctan2(x-x0,y-y0calc)
        # Save result
        rPhiVals[j, 0] = phi
        rPhiVals[j, 1] = r

    # Sort the r and phi values based on phi
    rPhiSorted = rPhiVals[rPhiVals[:, 0].argsort()]

    #calculate the mode amplitudes
    an = np.zeros(max_mode_number+1)
    for j in range(Npoints-1):
        phi = rPhiSorted[j, 0]
        dphi = rPhiSorted[j+1, 0] - rPhiSorted[j, 0]
        r = rPhiSorted[j, 1]
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        for n in range(max_mode_number+1):
            an[n] += eval_legendre(n, cos_phi) * r * sin_phi * dphi
    for n in range(max_mode_number+1):
        an[n] *= (2.0*n+1.0)/2.0

    # Write file
    file.write("{:.15f},{:.15f},{:.24f},{:.24f},{:.15f},{:.15f},{:.15f}".format(timesteps[i],RB,VB,CVB,y0B,pB,rhoB))
    for n in range(max_mode_number+1):
        file.write(",{:.15f}".format(an[n]))
    file.write("\n")

    # Calculate line data - vertical line in the full domain
    UpdatePipeline(time=timesteps[i], proxy=verticalLine1)
    lineData = paraview.servermanager.Fetch(verticalLine1)
    linePoints = lineData.GetPointData()
    line_inp = GetInterpolatedArray(linePoints, saveVars, points_vertical_line)
    line_inp[:,0] += y0 - saveDomainSize*R0
    np.savetxt(folderVerticalLine+'/step_'+str(i)+'.csv', line_inp, delimiter=',',header=line_header,comments='')

    #find jet impact time and jet velocity
    if not jet_impact_happened:
        only_water = True
        for j in range(points_vertical_line):
            if line_inp[j,5] < 0.05:
                only_water = False
                break
        
        #if there is only water along the y-axis, then jeting has happened!
        if only_water:
            jet_impact_happened = True
            velocity_max = 0.0
            for j in range(points_vertical_line):
                if line_inp[j,4] > velocity_max:
                    velocity_max = line_inp[j,4]
            
            jet_impact_R = RB
            jet_impact_T = timesteps[i]
            jet_impact_V = velocity_max

    # Calculate line data - horizontal line in the full domain
    UpdatePipeline(time=timesteps[i], proxy=horizontalLine1)
    lineData = paraview.servermanager.Fetch(horizontalLine1)
    linePoints = lineData.GetPointData()
    line_inp = GetInterpolatedArray(linePoints, saveVars, points_horizontal_line)
    np.savetxt(folderHorizontalLine+'/step_'+str(i)+'.csv', line_inp, delimiter=',',header=line_header,comments='')


    SaveScreenshot(folderSubdomain + "/step_" + str(i) + ".png",renderView2,ImageResolution=[1600,1600])

file.close()
if debug:
    print("   Filewriters closed")

print("----------------------------------------------------------------------")
print("                 Writing the Metadata")
print("----------------------------------------------------------------------")

case_id = "f" + str(int(1.0e-3*f)) + "_R" + str(int(1.0e6*R0)) + "_pA" + str(int(1.0e-3*pA))

np.savetxt(workingDirectory + '/' + args.outputfolder + "/timesteps_full.csv",np.transpose(np.array([range(lengthFull),timestepsFull])),delimiter=',',header='id,time[s]',comments='',fmt=['%d','%.15e'])
np.savetxt(workingDirectory + '/' + args.outputfolder + "/timesteps_sub.csv" ,np.transpose(np.array([range(length),timesteps])),delimiter=',',header='id,time[s]',comments='',fmt=['%d','%.15e'])

inputFile = list(Path(workingDirectory + '/' +  args.inputfolder).glob("*.xml"))[0]
print("Inputfile: ",inputFile)
shutil.copy(inputFile,workingDirectory + '/' +  args.outputfolder + "/inputfile_" + case_id + ".xml")

metadata = {
    "cased_id": case_id + ".csv",
    "control_parameters": {
        "frequency": f,
        "pressure_amplitude": pA,
        "radius": R0,
        "bubble_center": y0
    },
    "parameters": "parameter_set_1.json",
    "validity": {
        "mass_min": resMassMin,
        "mass_max": resMassMax,
        "mass_end": resMassEnd,
        "collapse_captured": "True",
        "accepted": "True"
    },
    "results": {
        "max_size": {
            "time": resMaxT,
            "radius": resMaxR,
            "mass": resMaxM
        },
        "min_size": {
            "time": resMinT,
            "radius": resMinR,
            "mass": resMinM
        },
        "jet_impact": {
            "happened:": jet_impact_happened,
            "time": jet_impact_T,
            "radius": jet_impact_R,
            "jet_velocity": jet_impact_V
        },
        "compression_ratio":
        {
            "RE_based": R0 / resMinR,
            "Rmax_based": resMaxR / resMinR
        },
        "energy_dissipation": {
            "E_init": E_init,
            "t_init": timestepsFull[0],
            "E_final": E_final,
            "t_final": t_final,
            "E_dissipation": E_init - E_final,
            "pulse_passed": t_final > 1.05/f
        },
        "notes:":
        {
            "jet_type": "empty",
            "comment": "empty"
        }
    },
    "timesteps": {
        "full": {
            "list": "timesteps_full.csv",
            "N": lengthFull
        },
        "sub": {
            "list": "timesteps_sub.csv",
            "N": length
        }
    },
    "colorbars": {
        "pressure": {
            "colormap": "cool_to_warm",
            "range_full": [p0-pA,p0+pA],
            "type_full": "linear",
            "range_sub": presScaleSub,
            "type_sub": "log"
        },
        "velocity": {
            "colormap": "X_ray",
            "range_full": velScaleFull,
            "type_full": "log",
            "range_sub": velScaleSub,
            "type_sub": "log"
        }
    },
    "input_file": "inputfile_" + case_id + ".xml"
}

with open(workingDirectory + '/' +  args.outputfolder + "/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)


print("----------------------------------------------------------------------")
print("                 End of the postprocessor code")
print("----------------------------------------------------------------------")
