import bpy
import random
import numpy as np
import argparse
import os
import sys
import json
import time
from mathutils import Vector

sys.path.insert(0, ".")
import blender_utils as bu


def parse_args(argv=None):
    """
    Parse input arguments
    """
    if argv is not None:

        parser = argparse.ArgumentParser()
        parser.add_argument('--cad-path', dest='cad_path', help='Path to all PLY CAD models', type=str)
        parser.add_argument('--dtd-path', dest='dtd_path', help='Path to DTD images for plane texture', type=str)
        parser.add_argument('--plane-distance', dest='plane_distance', help='Camera to middle of table distance', type=float, default=1.2)
        parser.add_argument('--theta_x', dest='theta_x', help='Euler angles of object in degrees (X, Y, Z)', type=float)
        parser.add_argument('--theta_y', dest='theta_y', help='Euler angles of object in degrees (X, Y, Z)', type=float)
        parser.add_argument('--theta_z', dest='theta_z', help='Euler angles of object in degrees (X, Y, Z)', type=float)
        parser.add_argument('--lamp-theta', dest='lamp_theta', help='Lamp location (x, y, z)', type=float)
        parser.add_argument('--lamp-phi', dest='lamp_phi', help='Lamp location (x, y, z)', type=float)
        parser.add_argument('--lamp-color', dest='lamp_color', help='Lamp color (S in HSV)', type=float)
        parser.add_argument('--ao', dest='ambient_occlusion', help='Proportion of ambiant light (0 to 1)', default=0.5,
                            type=float)
        parser.add_argument('--num_objects', dest='num_objects', type=int, default=1)
        parser.add_argument('--lamp-strength', dest='lamp_strength', help='Lamp  strength', type=float)
        parser.add_argument('--focal', dest='focal_length', help='Focal length', default=107.4, type=float)
        parser.add_argument('--savepath', dest='save_path', help='Output path for the dataset', default='', type=str)
        parser.add_argument('--image-index', dest='idx', help='Image index in the dataset', type=int, default=1)
        parser.add_argument('--normals', action='store_true')
        parser.add_argument('--depth', action='store_true')
        parser.add_argument('--cuda', dest='cuda_visible_devices', type=str, default=0)
        parser.add_argument('--nocuda', action='store_true')

        args = parser.parse_args(argv)
    else:
        args = None

    return args

def set_scene(cad_path, num_objects=1, plane_image_path='',
              lamp_phi=0, lamp_theta=0,
              lamp_strength=100,
              lamp_hsv_color=0,
              theta_x=0, theta_y=0, theta_z=0,
              plane_distance=2,
              focal_length=100):

    # Clear scene objects and materials

    print('Here')
    scene = bpy.data.scenes['Scene']
    scene_tree = scene.node_tree
    nodes = scene_tree.nodes

    print('======== using {} objects =========='.format(num_objects))

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)

    # Set table plane
    plane = bpy.ops.mesh.primitive_cube_add(radius=10, view_align=False, enter_editmode=False,
                                             location=(0, 0, float(plane_distance)))

    plane = bpy.data.objects['Cube']
    plane.name = 'Plane'

    plane.location[0] = 0
    plane.location[1] = 0
    plane.location[2] = float(plane_distance)

    plane.dimensions[0] = 30
    plane.dimensions[1] = 30

    plane.scale[0] = 2
    plane.scale[1] = 2
    plane.scale[2] = 0.01

    bpy.ops.rigidbody.object_add()
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.collision_shape = 'MESH'

    # Material and texture
    bpy.ops.object.material_slot_add()
    plane_mat = bpy.data.materials.new(name='PlaneMat')
    plane_mat.use_nodes = True
    plane.material_slots[0].material = plane_mat
    plane_nodetree = plane_mat.node_tree

    for n in plane_nodetree.nodes:
        plane_nodetree.nodes.remove(n)
    plane_bsdf = plane_nodetree.nodes.new("ShaderNodeBsdfDiffuse")
    plane_out = plane_nodetree.nodes.new("ShaderNodeOutputMaterial")

    plane_tc = plane_nodetree.nodes.new("ShaderNodeTexCoord")
    tex_mapping = plane_nodetree.nodes.new("ShaderNodeMapping")
    plane_tex = plane_nodetree.nodes.new("ShaderNodeTexImage")
    tex_mapping.vector_type = 'TEXTURE'
    tex_mapping.scale[0] = 0.2
    tex_mapping.scale[1] = 0.2

    plane_nodetree.links.new(plane_tc.outputs[0], tex_mapping.inputs[0])
    plane_nodetree.links.new(tex_mapping.outputs[0], plane_tex.inputs[0])
    plane_nodetree.links.new(plane_tex.outputs[0], plane_bsdf.inputs[0])
    plane_nodetree.links.new(plane_bsdf.outputs[0], plane_out.inputs[0])

    # Set table texture
    img = bpy.data.images.load(os.path.abspath(plane_image_path))
    plane_tex.image = img

    theta = [theta_x, theta_y, theta_z]
    bu.apply_transform_to_selected_obj(plane, theta, [0, 0, 0])

    bpy.context.scene.objects.active = None

    obj_list = {}

    # GUIDING TUBE

    tube_sz = 3.5 * np.max([0.7, np.min([17*plane_distance / theta_x, 1.15])])

    bpy.ops.mesh.primitive_cone_add(radius1=tube_sz, radius2=0, depth=50, view_align=False, enter_editmode=False,
                                    location=(0, 0, float(plane_distance)))

    cone = bpy.data.objects['Cone']
    bu.apply_transform_to_selected_obj(cone, theta, [0, 0, 0])

    cone.dimensions[0] = tube_sz
    cone.dimensions[1] = tube_sz
    cone.dimensions[2] = 50
    cone.scale[0] = 1
    cone.scale[1] = 1
    cone.scale[2] = 1


    bu.apply_transform_to_selected_obj(cone, [0, 0, 0], [0, 0, tube_sz], translate_first=False)
    z_cone = cone.location[-1]
    cone.location = [0, 0, z_cone]

    # make invisible
    bpy.context.object.cycles_visibility.camera = False
    bpy.context.object.cycles_visibility.diffuse = False
    bpy.context.object.cycles_visibility.glossy = False
    bpy.context.object.cycles_visibility.transmission = False
    bpy.context.object.cycles_visibility.scatter = False
    bpy.context.object.cycles_visibility.shadow = False

    bpy.ops.rigidbody.object_add()
    cone.rigid_body.type = 'PASSIVE'
    cone.rigid_body.collision_shape = 'MESH'
    bpy.context.object.rigid_body.friction = 0.0

    # OBJECTS

    for obj in bpy.data.objects:
        obj.select = False

    instance_nodes = []

    for i in range(num_objects):
        obj_idx = np.random.randint(30)
        # choose object from cad path
        # if obj_idx not in range(18, 23):
        #     obj_path = os.path.join(cad_path.replace('models_cad', 'models_reconst'),
        #                             'obj_' + '%02d' % (obj_idx + 1) + '.ply')
        # else:
        print('here')
        obj_path = os.path.join(cad_path, 'obj_' + '%02d' % (obj_idx + 1) + '.ply')

        # import object
        obj = bpy.ops.import_mesh.ply(filepath=os.path.abspath(obj_path))
        obj = bpy.context.active_object
        print(obj)
        obj.name = 'obj%d' % i
        obj_list[obj.name] = obj_idx
        obj.location = [0, 0, float(plane_distance)]
        obj.pass_index = i+1  # the plane has index 1

        # set at random initial pose
        T_x = random.uniform(-1, 1) * 0.075
        T_y = random.uniform(-1, 1) * 0.075
        T_z = np.max([-4.0, (-4.0 * (i+1)/num_objects)])

        theta_obj_x = random.uniform(-1, 1) * 180
        theta_obj_y = random.uniform(-1, 1) * 180
        theta_obj_z = random.uniform(-1, 1) * 180
        theta_obj = [theta_obj_x, theta_obj_y, theta_obj_z]
        bu.apply_transform_to_selected_obj(obj, theta=[theta_x, theta_y, theta_z], T=[0, 0, 0], translate_first=False)
        bu.apply_transform_to_selected_obj(obj, theta=[0, 0, 0], T=[T_x, T_y, T_z], translate_first=False)

        bu.apply_transform_to_selected_obj(obj, theta=theta_obj, T=[0, 0, 0], translate_first=True)

        obj.dimensions = obj.dimensions * 0.01

        # set physics
        bpy.ops.rigidbody.object_add()

        obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.mass = 1.0


        # set material
        bpy.ops.object.material_slot_add()
        t_less_mat = bpy.data.materials.new(name='TLessMaterial' + str(i))
        t_less_mat.use_nodes = True
        obj.material_slots[0].material = t_less_mat
        t_less_nodetree = t_less_mat.node_tree

        obj_gray_level = random.uniform(0.3, 1)

        for n in t_less_nodetree.nodes:
            t_less_nodetree.nodes.remove(n)
        tless_bsdf = t_less_nodetree.nodes.new("ShaderNodeBsdfDiffuse")
        tless_out = t_less_nodetree.nodes.new("ShaderNodeOutputMaterial")

        if obj_idx not in range(18, 23):
            tless_color = t_less_nodetree.nodes.new("ShaderNodeAttribute")
            tless_color.attribute_name = "Col"
            tless_color_mix = t_less_nodetree.nodes.new("ShaderNodeMixRGB")
            tless_color_mix.inputs[0].default_value = random.uniform(0, 0.7)
            tless_color_mix.inputs[2].default_value = (1-obj_gray_level,
                                                       1 - obj_gray_level,
                                                       1 - obj_gray_level, 1)
            t_less_nodetree.links.new(tless_color.outputs[0], tless_color_mix.inputs[1])
            t_less_nodetree.links.new(tless_color_mix.outputs[0], tless_bsdf.inputs[0])
        else:
            tless_bsdf.inputs[0].default_value = (obj_gray_level, obj_gray_level, obj_gray_level, 1)

        t_less_nodetree.links.new(tless_bsdf.outputs[0], tless_out.inputs[0])


    # Apply animation and get all object poses
    scene.rigidbody_world.time_scale = 60

    # Set gravity as -1 * plane normal
    R = np.array(plane.matrix_world)[:3, :3]
    g = -200 * np.array(bpy.context.scene.gravity)

    g = np.dot(R, g)

    bpy.context.scene.gravity = tuple(g)

    # Run physics simulation
    bpy.context.scene.frame_set(1)
    for i in range(50):
        print("Physics: {} / {}".format(i+1, 50))
        bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
        bpy.data.scenes['Scene'].frame_set(bpy.context.scene.frame_current + 1)

    bpy.context.scene.frame_current = 49
    bpy.context.scene.frame_set(49)

    success = True

    # because applying bpy.ops.transform.resize does not work as expected when rigid bodies are on...
    for obj in bpy.data.objects:
        bpy.context.scene.objects.active = obj

        # save rotation and translation now, because blender is about to forget everything...
        r = [a for a in bpy.context.object.matrix_world.to_euler()]
        t = [a for a in bpy.context.object.matrix_world.to_translation()]


        for trans in t:
            if np.abs(trans) > 10:
                success = False

        if success is True:
            bpy.ops.rigidbody.object_remove()
            # because blender resets everything when removing rigid body...
            bpy.context.object.rotation_euler = r
            bpy.context.object.location = t

    if success:

        # get barycenter point
        bpy.ops.object.select_all(action='SELECT')

        bpy.ops.transform.resize(value=(0.1, 0.1, 0.1), constraint_axis=(False, False, False),
                                 constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                 proportional_edit_falloff='SMOOTH', proportional_size=1)
        #
        obj_sel = [ob.location for ob in bpy.data.objects if ob.select and ob.name.startswith('obj')]#and not ob.name.startswith('wall'))]
        centroid = sum(obj_sel, Vector()) / len(obj_sel)

        bpy.ops.transform.translate(value=(-1.0 * centroid[0], 0, 0),
                                    constraint_axis=(True, False, False),
                                    constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                    proportional_edit_falloff='SMOOTH', proportional_size=1)

        bpy.ops.transform.translate(value=(0, -1.0 * centroid[1], 0),
                                    constraint_axis=(False, True, False),
                                    constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                    proportional_edit_falloff='SMOOTH', proportional_size=1)

        bpy.ops.transform.translate(value=(0, 0, plane_distance - centroid[2]),
                                    constraint_axis=(False, False, True),
                                    constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                    proportional_edit_falloff='SMOOTH', proportional_size=1)

        # I don't know why, but objects are shifted by a small distance along Y if we don't do this
        bpy.context.scene.objects.active = None
        for obj in bpy.data.objects:
            obj.select = False
        for obj in bpy.data.objects:
            if obj.name != 'Plane':
                obj.select = True
                bpy.context.scene.objects.active = obj
                bpy.ops.transform.translate(value=(0, -0.0045, 0),
                                            constraint_axis=(False, True, False),
                                            constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED',
                                            proportional_edit_falloff='SMOOTH', proportional_size=1)
                bpy.context.scene.objects.active = None
                obj.select = False

        cone.location[0] = 0
        cone.location[1] = 0

        # Set camera
        camera = bu.add_camera((0, 0, 0), (0, np.pi, 0), 'camera', 'PERSP',
                               focal_length, sensor_fit='HORIZONTAL',
                               sensor_width=40)

        # Set lamp
        lamp = bu.add_lamp(xyz=(0, 0, 0), strength=lamp_strength, s_value=lamp_hsv_color)

        # convert spherical to cartesian coordinates
        lamp_location = (np.cos(lamp_theta * (np.pi / 180)) * np.cos(lamp_phi * (np.pi / 180)),
                         np.sin(lamp_theta * (np.pi / 180)) * np.cos(lamp_phi * (np.pi / 180)),
                         -np.sin(lamp_phi * (np.pi / 180)))

        lamp.location = [0, 0, plane_distance]

        # bu.apply_transform_to_selected_obj(lamp, [theta_x, theta_y, theta_z], [0, 0, 0], translate_first=False)
        bu.apply_transform_to_selected_obj(lamp, [0, 0, 0], [1.3 * lamp_location[0], 1.3 * lamp_location[1], 1.3 * lamp_location[2]], translate_first=False)

        return_yml = {}
        for name in obj_list.keys():
            obj_dict = {}
            obj_dict['type'] = obj_list[name]
            obj_dict['T'] = list(bpy.data.objects[name].matrix_world.to_translation())
            obj_dict['Euler'] = [(180 / np.pi) * a for a in bpy.data.objects[name].matrix_world.to_euler()]

            return_yml[name.split('obj')[-1]] = obj_dict

        return_yml['Plane'] = {'Distance': plane_distance, 'theta': theta_z, 'phi': theta_x}

    else:
        return_yml = {}

    return return_yml


if __name__ == '__main__':

    if '--' not in sys.argv:
        argv = None
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]

    args = parse_args(argv)

    if not args.nocuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    if args is None:
        print('Specify arguments')
        sys.exit(0)

    lamp_theta = args.lamp_theta
    lamp_phi = args.lamp_phi
    lamp_strength = args.lamp_strength
    lamp_color = args.lamp_color

    cad_path = args.cad_path

    theta_x = args.theta_x
    theta_y = args.theta_y
    theta_z = args.theta_z

    focal_length = args.focal_length

    cad_path = args.cad_path
    dtd_path = args.dtd_path
    num_objects = int(args.num_objects)
    plane_distance = args.plane_distance


    bu.set_cycles(w=400, h=400, n_samples=250, gpu_id=args.cuda_visible_devices,
                  outpath=os.path.join(args.save_path, str(args.idx)),
                  use_normals=args.normals, use_depth=args.depth, use_obj_instances=True, num_objects=num_objects)

    yml = set_scene(cad_path, plane_image_path=dtd_path,
              focal_length=focal_length, num_objects=num_objects,
              lamp_phi=lamp_phi, lamp_theta=lamp_theta,
              lamp_hsv_color=lamp_color, lamp_strength=lamp_strength,
              theta_x=theta_x, theta_y=theta_y, theta_z=theta_z, plane_distance=plane_distance)

    # Set ambient light
    bpy.context.scene.world.light_settings.ao_factor = args.ambient_occlusion

    if not os.path.exists(os.path.join(args.save_path, 'images')):
        os.makedirs(os.path.join(args.save_path, 'images'))
    if not os.path.exists(os.path.join(args.save_path, 'gt_poses')):
        os.makedirs(os.path.join(args.save_path, 'gt_poses'))
    bpy.context.scene.render.filepath = os.path.join(args.save_path, 'images', 'img'+'_%05d' % args.idx + '.png')
    bpy.ops.render.render(write_still=True)

    with open(os.path.join(args.save_path, 'gt_poses', 'tmp_%05d' % int(args.idx) + '.json'), 'w') as f:
        json.dump(yml, f)
