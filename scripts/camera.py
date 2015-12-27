import bpy
import math

scene = bpy.context.scene
render = bpy.context.scene.render

#----------------------------------------------- 
# view port 

render.engine = 'CYCLES'
render.filepath = '//'
render.resolution_x = 1920
render.resolution_y = 1080
render.resolution_percentage = 100

scene.cycles.device = 'GPU'

#----------------------------------------------- 
# K 
camdata = bpy.data.cameras.new('cameraData')
camdata.lens_unit = 'FOV'

f = float(1000)
fx = float(1000)
fy = float(1000)
ppx = float(958)
ppy = float(562)

maxdim = max(render.resolution_x,render.resolution_y) 

#camdata.angle = 2*math.atan(0.5*maxdim/f)
camdata.angle_x=2*math.atan(0.5*render.resolution_x/fx);
camdata.angle_y=2*math.atan(0.5*render.resolution_y/fy);

# the unit of shiftXY is FOV unit (Lens Shift)
camdata.shift_x = (ppx - render.resolution_x/2.0)/maxdim
camdata.shift_y = (ppy- render.resolution_y/2.0)/maxdim

camdata.dof_distance = 0.0
camdata.clip_end = 1000.0

#-----------------------------------------
# OBJECT TYPE
cam = bpy.data.objects.new('camera', camdata)
#
# flip axis to top-left
cam.scale=(1,-1,-1); 

scene.objects.link(cam)
scene.objects.active = cam
cam.select = True