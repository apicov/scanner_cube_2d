import numpy as np
import json
import os
import bpy
import mathutils

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def setup_camera(w, h, f):
    """
    :input w image width
    :input h image height
    :input f focal length (equiv. 35mm)
    """
    scene = bpy.data.scenes["Scene"]

    fov = 90.0

    # Set render resolution
    #scene.render.tile_x = 256
    #scene.render.tile_y = 256
    
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    # Set camera fov in degrees
    scene.camera.data.angle = 2*np.arctan(35/f)
    scene.camera.data.clip_end = 10000


def get_K():
    scene = bpy.data.scenes["Scene"]
    camd = scene.camera.data
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = [[s_u, skew, u_0],
        [   0,  s_v, v_0],
        [   0,    0,   1]]
    return K

def get_RT():
    scene = bpy.data.scenes["Scene"]
    cam = scene.camera
    # bcam stands for blender camera
    
    R_bcam2cv = mathutils.Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))
    
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    R = np.matrix(R_world2cv)
    T = np.array(T_world2cv)

    return R.tolist(), T.tolist()
    

def get_bounding_box():
    scene = bpy.context.scene
            
    xmin, ymin, zmin = 10000, 10000, 10000
    xmax, ymax, zmax = -10000, -10000, -10000
    for o in bpy.data.objects:
        m = o.matrix_world
        if not(o.name in ['Camera','Light']):
            for b in o.bound_box:
                x,y,z = b
                xmin, ymin, zmin = np.minimum([xmin, ymin, zmin], [x,y,z])
                xmax, ymax, zmax = np.maximum([xmax, ymax, zmax], [x,y,z])

    bbox ={
        "x" : [xmin, xmax],
        "y" : [ymin, ymax],
        "z" : [zmin, zmax]
    }
    return bbox


def get_camera_intrinsics():
    scene = bpy.data.scenes["Scene"]
    K=get_K()
    camera_model = {
        "width" : scene.render.resolution_x,
        "height" : scene.render.resolution_y,
        "model" : "OPENCV",
        "params" : [ K[0][0], K[1][1], K[0][2], K[1][2], 0.0, 0.0, 0.0, 0.0 ]
    }

    return camera_model

def get_camera_extrinsics():
    R,T= get_RT()
    return  {"R":R,"T":T}

def move_camera(tx=None, ty=None, tz=None):
    #if "--hdri" in argv:
    #    bpy.context.scene.cycles.film_transparent = False #hdri visible

    # Set camera translation
    scene = bpy.data.scenes["Scene"]
    if tx is not None:
        scene.camera.location[0] = float(tx)
    if ty is not None:
        scene.camera.location[1] = float(ty)
    if tz is not None:
        scene.camera.location[2] = float(tz)
 

def rotate_camera(rx=None, ry=None, rz=None):
    scene = bpy.data.scenes["Scene"]

    # Set camera rotation in euler angles
    scene.camera.rotation_mode = 'XYZ'
    if rx is not None:
        scene.camera.rotation_euler[0] = float(rx)*(np.pi/180.0)
    if ry is not None:
        scene.camera.rotation_euler[1] = float(ry)*(np.pi/180.0)
    if rz is not None:
        scene.camera.rotation_euler[2] = float(rz)*(np.pi/180.0)
    #bpy.context.scene.cycles.film_transparent = False

    
def goto(i_theta, i_phi, r=5,n_theta=180,n_phi=4):
    theta=i_theta*2*np.pi/n_theta
    phi=i_phi*.5*np.pi/n_phi
    x = r *np.cos(phi) * np.cos(theta) #x pos of camera
    y = r *np.cos(phi) * np.sin(theta) #y pos of camera   
    z = r *np.sin(phi)
    rx = - phi*180/np.pi + 90 #camera tilt
    rz = theta * 180/np.pi + 90
    move_camera(x, y, z)
    rotate_camera(rx, 0, rz)

def goto_cylinder( i_theta, i_phi, r=5,n_theta=180,n_phi=4):
    #for cylinder
    max_height = 6
    theta=i_theta*2*np.pi/n_theta
    x = r * np.cos(theta) #x pos of camera
    y = r * np.sin(theta) #y pos of camera   
    z = (max_height/n_phi)*i_phi
    #self.rx = - self.phi*180/np.pi + 90 #camera tilt
    rz = theta * 180/np.pi + 90
    #rx = -np.arctan(z/r) * 180/np.pi + 90
    if i_phi == n_phi-1:
        rx = -np.arctan(z/r) * 180/np.pi + 90
    else:
        rx = 90
    move_camera(x, y, z)
    rotate_camera(rx, 0, rz)
    #fr = get_frame_color(im)
    #return fr
    

def extrinsics2pose(r,t):
     R_bcam2cv = mathutils.Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

     R_world2cv = mathutils.Matrix(r)
     T_world2cv = mathutils.Vector(t)

     rotation = (R_bcam2cv.transposed() @ R_world2cv ).transposed()
     print(rotation)

     T_world2bcam = R_bcam2cv.transposed() @ T_world2cv
     location = -1 *  rotation @ T_world2bcam

     euler = rotation.to_euler()

     return [[np.rad2deg(x) for x in euler],location]


def load_object(fname, dx = None, dy = None, dz = None):
    """move object by dx, dy, dz if specified"""
    scene = bpy.data.scenes["Scene"]
    loc = scene.camera.location
    rot = scene.camera.rotation_euler
    print("loc=%s"%str(loc))
    print("rot=%s"%str(rot))

    for x in bpy.data.objects:
        if "SHAPEID" in x.name:
            bpy.data.objects.remove(x, True)
    bpy.ops.import_scene.obj(
        filepath=fname)
    imported_object = bpy.context.scene.objects
    clear_all_rotation()

    for obj in imported_object: #Move object
        if dx is not None:
            obj.location.x = float(dx)
        if dy is not None:
            obj.location.y = float(dy)
        if dz is not None:
            obj.location.x = float(dz)

    scene.camera.location = loc
    scene.camera.rotation_euler = rot
    
    #HDRI background
    if "--hdri" in argv:
        current_bg_image = bpy.data.images.load(hdri_folder + hdri_list[int(time.time())%L]) #random hdri background from folder
        env_text_node.image = current_bg_image   
        bpy.context.scene.cycles.film_transparent = False #hdri visible


    if 'Cube' in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects['Cube'], do_unlink=True)




#exec(compile(open('/home/pico/uni/romi/blender_scripts/blender_test.py').read(), '/home/pico/uni/romi/blender_scripts/blender_test.py', 'exec'))
#exec(compile(open('/home/pico/uni/romi/scanner_cube_2d/blender_scripts/blender_test.py').read(), '/home/pico/uni/romi/scanner_cube_2d/blender_scripts/blender_test.py', 'exec'))

if __name__ == "__main__":
    params_path = "/home/pico/uni/romi/scanner_cube_2d/blender_scripts/params_image_collection.json"
    #object_path = "/home/pico/uni/romi/3d_obj_models_scanner-gym_v2/216.obj"
    
    #load parameters
    params=json.load(open(params_path))
    #horizontal resolution
    w=params['scanner']['w']
    #vertical resolution
    h=params['scanner']['h']
    #focal length in mm
    f=params['scanner']['f']
    #number of images to take in horizontal
    N_theta = params["traj"]["N_theta"]
    #n images vertical
    N_phi = params["traj"]["N_phi"]
    #radial distance from [x, y] center
    R = params["traj"]["R"] 

    setup_camera(w, h, f)

    #import object
    #bpy.ops.import_scene.obj(filepath=object_path)


    # Parent Directory path 
    #parent_dir = "/home/pico/uni/romi/render_test/"
    parent_dir = '/home/pico/uni/romi/scanner-gym_models_v3'
    # Directory 
    directory =  str('213_2d')
    # Path 
    path = os.path.join(parent_dir, directory)
    
    print(path)
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    img_dir  = os.path.join(path, 'imgs')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
            
            
    ext_dir  = os.path.join(path, 'extrinsics')
    if not os.path.exists(ext_dir):
        os.makedirs(ext_dir)
                
    masks_dir  = os.path.join(path, 'masks')
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    #get and save bbox and intrinsics
    bbox = get_bounding_box()
    camera_model = get_camera_intrinsics()

    with open(os.path.join(path, 'params.json'), 'w') as json_file:
        json.dump(params, json_file)
            
    with open(os.path.join(path, 'bbox.json'), 'w') as json_file:
        json.dump(bbox, json_file)

    with open(os.path.join(path, 'camera_model.json'), 'w') as json_file:
        json.dump(camera_model, json_file)



        
    
    '''# Create a Scene:
    scene       = bpy.data.scenes.new("Scene")
    camera_data = bpy.data.cameras.new("Camera")
    light_data  = bpy.data.lights.new(name="Light", type='SUN')
    # Get handles for all objects:
    maplet = bpy.context.selected_objects[0] 
    scene = bpy.data.scenes.new("Scene")
    camera = bpy.data.objects.new("Camera", camera_data)
    sun    = bpy.data.objects.new(name="Light", object_data=light_data)
    
    

    # Setup camera parameters:
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    camera.location = (-2.0, 3.0, 3.0)
    camera.rotation_euler = ([np.deg2rad(a) for a in (422.0, 0.0, 149)])

    # Setup scene:
    sun.location = (2.0,2.0,2.0)
    sun.rotation_euler = (np.deg2rad(10), np.deg2rad(15), np.deg2rad(20))
    maplet.location = (0,0,0)
    maplet.rotation_euler = (0,0,0)
        
    # Add camera to the scene:
    scene.camera = camera

    # Update the scene:
    bpy.context.view_layer.update()'''

    for z in range(params['traj']['N_phi']):
        for theta in range(params['traj']['N_theta']):
            image = goto(theta,z,R,N_theta,N_phi)
            #image = goto_cylinder(theta,z,R,N_theta,N_phi)
            im_path = img_dir + '/' + str(z).zfill(3) + '_' + str(theta).zfill(3) +'.png'
            mask_path =  masks_dir + '/' + str(z).zfill(3) + '_' + str(theta).zfill(3) +'.png'
            ext_path = ext_dir + '/' + str(z).zfill(3) + '_' + str(theta).zfill(3) +'.json'
            print (im_path)


            #render cam image 
            sce = bpy.context.scene.name
            bpy.data.scenes[sce].render.filepath = im_path
            # go into camera-view 
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    # change perspective to camera
                    area.spaces[0].region_3d.view_perspective = 'CAMERA'
                    
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            #hide overlays
                            space.overlay.show_overlays = False
                            break
                    break
            # hide others
            scene = bpy.context.scene
            #lamp_objects = [o for o in scene.objects if o.type == 'LAMP']
            #for ob in lamp_objects: ob.hide_render = True
            # Render image through viewport
            bpy.ops.render.opengl(write_still=True)
            #bpy.context.scene.render.engine = 'CYCLES'
            #bpy.ops.render.render(write_still=True)
            #for ob in lamp_objects: ob.hide_render = False


            #image.save(im_path)
            #mask = get_mask(image)
            #cv2.imwrite(mask_path,mask)

            extrinsics = get_camera_extrinsics()
            with open(ext_path, 'w') as json_file:
                json.dump(extrinsics, json_file)
            print(extrinsics)
            
            #bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            #cv2.imwrite( im_path,  bgr)
            #plt.imshow(image)
            #plt.savefig( path + '/' + str(theta).zfill(3) + '_' + str(z).zfill(z) +'.png')
            #plt.clf()
    

    #bpy.ops.wm.quit_blender()

