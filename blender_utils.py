import os
import bpy, _cycles
import numpy as np
import colorsys

def set_cycles(w=None, h=None, n_samples=None, gpu_id="0", outpath='',
               use_normals=False, use_depth=False, use_obj_instances=False, num_objects=1, high_quality=True):

    print('OUTPATH: {}'.format(outpath))

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    scene.use_nodes = True

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    prefs = bpy.context.user_preferences
    cprefs = prefs.addons['cycles'].preferences
    try:
        cprefs.compute_device_type = 'CUDA'
    except Exception as e:
        print("Could not use GPU" + gpu_id + " as device. Using CPU")
        cprefs.compute_device_type = 'NONE'

    for device in cprefs.devices:
        device.use = True

    cycles = scene.cycles

    cycles.use_progressive_refine = True
    if n_samples is not None:
        cycles.samples = n_samples
    else:
        cycles.samples = 500

    cycles.max_bounces = 200 if high_quality else 20
    cycles.min_bounces = 10 if high_quality else 3
    cycles.caustics_reflective = True
    cycles.caustics_refractive = False
    cycles.diffuse_bounces = 100 if high_quality else 10
    cycles.glossy_bounces = 20 if high_quality else 4
    cycles.transmission_bounces = 0
    cycles.volume_bounces = 0
    cycles.transparent_min_bounces = 0
    cycles.transparent_max_bounces = 0

    # remove anti-aliasing
    # cycles.filter_width = 0.01

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.debug_use_spatial_splits = True

    # cycles.blur_glossy = 3
    # cycles.sample_clamp_indirect = 3

    # Ensure no background node
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    scene.render.tile_x = 256
    scene.render.tile_y = 256
    if w is not None:
        scene.render.resolution_x = w
    if h is not None:
        scene.render.resolution_y = h
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'

    s = outpath.rsplit('/', 1)
    print(s)
    print("Outpath set to: {}".format(outpath))

    nodes = scene.node_tree.nodes
    scene.render.layers["RenderLayer"].use_pass_combined = False
    render_layers = nodes["Render Layers"]

    if use_normals:

        if not os.path.exists(outpath.rsplit('/', 1)[0]):
            os.makedirs(outpath.rsplit('/', 1)[0])

        # scene.render.filepath = outpath

        scene.render.layers["RenderLayer"].use_pass_normal = True
        out_normals = nodes.new("CompositorNodeOutputFile")

        out_normals.base_path = os.path.join(s[0], 'normals/') + s[1] + '_'
        out_normals.format.file_format = "OPEN_EXR_MULTILAYER"
        scene.node_tree.links.new(render_layers.outputs["Normal"], out_normals.inputs["Image"])
        out_normals.format.color_depth = "32"
        out_normals.format.color_mode = "RGBA"
        out_normals.format.exr_codec = "NONE"

    if use_depth:
        scene.render.layers["RenderLayer"].use_pass_z = True
        out_depth = nodes.new("CompositorNodeOutputFile")

        out_depth.base_path = os.path.join(s[0], 'depth/') + s[1] + '_'

        scene.node_tree.links.new(render_layers.outputs["Depth"], out_depth.inputs["Image"])
        out_depth.format.file_format = "OPEN_EXR_MULTILAYER"

        out_depth.format.color_depth = "32"
        out_depth.format.color_mode = "RGB"
        out_depth.format.exr_codec = "NONE"

    if use_obj_instances:
        scene.render.layers["RenderLayer"].use_pass_object_index = True
        # set object instance index
        for i in range(num_objects):
            node_id = nodes.new("CompositorNodeIDMask")
            node_id.index = i + 1   # plane has index 1
            scale_node = nodes.new("CompositorNodeMath")
            scale_node.operation = 'DIVIDE'
            add_node = nodes.new("CompositorNodeMath")
            add_node.operation = 'ADD'
            scene.node_tree.links.new(render_layers.outputs['IndexOB'], node_id.inputs[0])
            scene.node_tree.links.new(node_id.outputs[0], scale_node.inputs[0])
            scale_node.inputs[1].default_value = float(256 / (i + 1))

            if i == 0:
                scene.node_tree.links.new(scale_node.outputs[0], add_node.inputs[0])
            else:
                scene.node_tree.links.new(scale_node.outputs[0], previous_add_node.inputs[1])
                scene.node_tree.links.new(previous_add_node.outputs[0], add_node.inputs[0])

            if i == (num_objects - 1):
                add_node.inputs[1].default_value = 0

            previous_add_node = add_node

        # object instances
        out_obj = nodes.new("CompositorNodeOutputFile")
        scene.node_tree.links.new(previous_add_node.outputs[0], out_obj.inputs[0])
        out_obj.base_path = os.path.join(s[0], 'instances/')
        out_obj.file_slots[0].path = s[1] + '_instances_'
        scene.render.filepath = os.path.join(s[0], 'instances/') + ('%05d' % int(s[1])) + '.png'
        out_obj.format.file_format = "PNG"
        out_obj.format.color_depth = "8"
        out_obj.format.color_mode = "BW"

    return 1


def apply_transform_to_selected_obj(obj, theta, T, translate_first=False):
    for ob in bpy.data.objects:
        ob.select = False
    bpy.context.scene.objects.active = None
    bpy.context.scene.objects.active = obj

    bpy.data.objects[obj.name].select = True

    if translate_first:
        bpy.ops.transform.translate(value=(T[0], T[1], T[2]), constraint_axis=(False, False, False),
                                    constraint_orientation="LOCAL")

    bpy.ops.transform.rotate(value=theta[0] * np.pi / 180, axis=(1, 0, 0), constraint_axis=(True, False, False),
                             constraint_orientation='LOCAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)

    bpy.ops.transform.rotate(value=theta[1] * np.pi / 180, axis=(0, 1, 0), constraint_axis=(False, True, False),
                             constraint_orientation='LOCAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)

    bpy.ops.transform.rotate(value=theta[2] * np.pi / 180, axis=(0, 0, 1), constraint_axis=(False, False, True),
                             constraint_orientation='LOCAL', mirror=False, proportional='DISABLED',
                             proportional_edit_falloff='SMOOTH', proportional_size=1)

    if not translate_first:
        bpy.ops.transform.translate(value=(T[0], 0, 0), constraint_axis=(True, False, False),
                                    constraint_orientation="LOCAL")
        bpy.ops.transform.translate(value=(0, T[1], 0), constraint_axis=(False, True, False),
                                    constraint_orientation="LOCAL")
        bpy.ops.transform.translate(value=(0, 0, T[2]), constraint_axis=(False, False, True),
                                    constraint_orientation="LOCAL")


def pick_random_file_from_txt_filelist(filelist_path, num_picks=1):
    picked_list = []

    with open(filelist_path, 'r') as f:
        file_list = [x.strip() for x in f.readlines()]

    if num_picks == 1:
        file_idx = np.random.randint(len(file_list))
        file_path = file_list[file_idx]
        return [file_path]
    else:
        for i in range(num_picks):
            file_idx = np.random.randint(len(file_list))
            file_path = file_list[file_idx]
            picked_list.append(file_path)

        print('Picked List: \n {}'.format(picked_list))
        return picked_list


def add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None,
               proj_model='PERSP', f=35, sensor_fit='HORIZONTAL',
               sensor_width=32, sensor_height=18):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    return cam


def add_lamp(xyz=(0, 0, 0), strength=100, s_value=0):
    bpy.ops.object.lamp_add(location=xyz)
    lamp = bpy.context.active_object

    lamp.location = xyz
    lamp.data.node_tree.nodes['Emission'].inputs['Strength'].default_value = strength

    c = colorsys.hsv_to_rgb(0.140, s_value, 1.0)
    c = c[:] + (1.0,)

    lamp.data.node_tree.nodes['Emission'].inputs['Color'].default_value = c

    return lamp