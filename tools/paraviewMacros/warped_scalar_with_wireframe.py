# This is paraview macro to setup the visualisation of the solutions 
# with the warp-by-scalar and wireframe grid. Add this file to paraview 
# by using the macro menu and selecting 'import new macro...'. Then, 
# open the .vtu file of interest, click 'Apply', and make sure that it 
# is active. Then click 'warped_scalar_with_wireframe' in the top-right.
# This will activate the macro and go through the needed step to get the 
# visualisation.


# Do NOT run this file directly in Python, it has to be adapted for that 
# to work.






# trace generated using paraview version 5.12.0-RC1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 12

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
originalSource = GetActiveSource()#FindSource('Poisson-2-D-p(3, 2)-k(2, 0)-m100-case-sine2d.vtu')

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=originalSource)

# Properties modified on extractSurface1
extractSurface1.NonlinearSubdivisionLevel = 4

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
extractSurface1Display.Representation = 'Surface'

# hide data in view
Hide(originalSource, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Warp By Scalar'
warpByScalar1 = WarpByScalar(registrationName='WarpByScalar1', Input=extractSurface1)

# Properties modified on warpByScalar1
warpByScalar1.ScaleFactor = 0.2

# show data in view
warpByScalar1Display = Show(warpByScalar1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
warpByScalar1Display.Representation = 'Surface'

# hide data in view
Hide(extractSurface1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(warpByScalar1Display, ('POINTS', 'point_data'))

# rescale color and/or opacity maps used to include current data range
warpByScalar1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
warpByScalar1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'point_data'
point_dataLUT = GetColorTransferFunction('point_data')

# get opacity transfer function/opacity map for 'point_data'
point_dataPWF = GetOpacityTransferFunction('point_data')

# get 2D transfer function for 'point_data'
point_dataTF2D = GetTransferFunction2D('point_data')

# get color legend/bar for point_dataLUT in view renderView1
point_dataLUTColorBar = GetScalarBar(point_dataLUT, renderView1)

# Properties modified on point_dataLUTColorBar
point_dataLUTColorBar.RangeLabelFormat = '%-#6.1f'

#change interaction mode for render view
renderView1.InteractionMode = '3D'

# get the material library
materialLibrary1 = GetMaterialLibrary()

# set active source
SetActiveSource(originalSource)

# get display properties
originalSourceDisplay = GetDisplayProperties(originalSource, view=renderView1)

# change representation type
originalSourceDisplay.SetRepresentationType('Wireframe')

# set active source
SetActiveSource(originalSource)

# show data in view
originalSourceDisplay = Show(originalSource, renderView1, 'UnstructuredGridRepresentation')

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1495, 794)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [-0.2779832135382434, -0.8803666873316901, 0.5430975307968678]
renderView1.CameraFocalPoint = [0.5, 0.5, -5.572524046316893e-18]
renderView1.CameraViewUp = [-0.08889546344245762, 0.4076624524024501, 0.90879531330249]
renderView1.CameraParallelScale = 0.3535533905932738


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://kitware.github.io/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------