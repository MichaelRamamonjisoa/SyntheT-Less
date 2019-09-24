import argparse
import numpy as np
import random
import os
import multiprocessing
import json
import sys
try:
    from scipy.misc import imread, imsave
except:
    from imageio import imread, imsave
import cv2
import geometry_utils as gu

'''
Some json functions to avoid unicode problems with Python 2
'''

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data


def process_chunk(index_list, args):
    params = {}
    if len(index_list) <= 0 or index_list is None:
        return {}
    for i, idx in enumerate(index_list):
        # Choose random parameters
        # light is set on hemisphere of radius 1
        lamp_theta = round(random.uniform(0, 1) * 360, 5)
        lamp_phi = round(random.uniform(30, 80), 5)

        # object pose
        # Rotation angles in degrees
        table_theta = round(random.uniform(0, 1) * 360, 5)
        table_phi = round(random.uniform(1, 80), 5)
        theta = (table_phi, 0, table_theta)

        # Translation in meters
        T_z = round(random.uniform(0.6, 1.4), 5)

        # Light parameters
        lamp_strength = round(random.uniform(45, 110), 5)
        lamp_color = round(random.uniform(0, 0.65), 5)
        ao_ratio = round(random.uniform(0.4, 1), 5)

        num_objects = int(np.random.randint(1, 9))

        with open(os.path.join(args.dtd_rootdir, 'table_image_list.txt'), 'r') as f:
            table_image_list = [x.strip() for x in f.readlines()]

        table_image_idx = np.random.randint(len(table_image_list))
        table_image = os.path.join(args.dtd_rootdir, table_image_list[table_image_idx])

        background = '--background' #set to empty string to debug in blender
        no_cuda = ' --nocuda ' if args.nocuda else ''

        outdir = 'multi_objects'
        script_name = 'render_sample_multi_object.py'

        command = args.blender_path + '/blender ' + background + ' --python ' + script_name + ' -- ' + \
                  ' --cad-path {} --dtd-path {}'.format(os.path.join(args.models_path), os.path.abspath(table_image)) + \
                  ' --plane-distance {:5f}'.format(T_z) + \
                  ' --num_objects {}'.format(num_objects) + \
                  ' --cuda ' + str(args.cuda_visible_devices) + \
                  no_cuda + \
                  ' --lamp-theta {:.5f} --lamp-phi {:.5f} --lamp-color {:.5f} --lamp-strength {:.5f}'.format(lamp_theta,
                                                                                                             lamp_phi,
                                                                                                             lamp_color,
                                                                                                             lamp_strength) + \
                  ' --ao {:.2f} '.format(ao_ratio) + \
                  ' --theta_x {:.5f} --theta_y {:.5f} --theta_z {:.5f}'.format(theta[0], theta[1], theta[2]) + \
                  ' --savepath {}'.format(outdir) + \
                  ' --image-index {} '.format(str(int(idx))) + \
                  ' --normals' +\
                  ' --depth'

        normals_path = os.path.join(outdir, 'normals')
        if not os.path.exists(normals_path):
            os.makedirs(normals_path)

        depth_path = os.path.join(outdir, 'depth')
        if not os.path.exists(depth_path):
            os.makedirs(depth_path)

        mask_path = os.path.join(outdir, 'mask')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

        contours_path = os.path.join(outdir, 'contours')
        if not os.path.exists(contours_path):
            os.makedirs(contours_path)

        instances_path = os.path.join(outdir, 'instances')
        if not os.path.exists(instances_path):
            os.makedirs(instances_path)

        try:
            os.system(command)
            json_path = os.path.join(outdir, 'gt_poses', 'tmp_%05d' % int(idx) + '.json')
            if sys.version_info[0] == 2:
                # script executed with python2
                with open(json_path, 'r') as f:
                    params_tmp = json_load_byteified(f)
            else:
                # script executed with python3
                with open(json_path, 'r') as f:
                    params_tmp = json.load(f)

            # os.remove(json_path)
            success = True

        except Exception as e:
            # CUDA memory error, json does not exist ...
            print(e)
            success = False
            break
            sys.exit(0)

        if success:
            params_tmp.update({'Lamp': {'phi': lamp_phi, 'theta': lamp_theta,
                                        'strength': lamp_strength},
                               'Ambient Light': ao_ratio,
                               'Table': {'distance': T_z, 'Euler': [theta[0], theta[1], theta[2]]}})
            with open(json_path, 'w') as f:
                json.dump(params_tmp, f)

            normals_exr_path = os.path.join(normals_path, str(int(idx)) + '_0049.exr')
            normals = gu.process_normals(normals_exr_path)

            imsave(os.path.join(normals_path, ('%05d' % int(idx)) + '.png'), normals)

            depth_exr_path = os.path.join(depth_path, str(int(idx)) + '_0049.exr')
            depth, mask = gu.process_depth(depth_exr_path)

            imsave(os.path.join(mask_path, ('%05d' % int(idx)) + '.png'), mask)
            cv2.imwrite(os.path.join(depth_path, ('%05d' % int(idx)) + '.png'), depth)

            # Rename instances file
            instances_savepath = os.path.join(instances_path, str(idx) + '_instances_0049.png')
            instances_newpath = os.path.join(instances_path, ('%05d' % int(idx)) + '.png')
            os.rename(instances_savepath, instances_newpath)

            # compute contours
            instances = imread(instances_newpath)

            contours = gu.compute_all_contours(depth, instances, normals,
                                               low_threshold=round(0.1 * np.min([np.max([0.1, table_phi / 45]), 0.5]),
                                                                   5),
                                               high_threshold=round(0.3 * np.min([np.max([0.3, table_phi / 45]), 0.8]),
                                                                    5))
            imsave(os.path.join(contours_path, ('%05d' % int(idx)) + '.png'), contours)

    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--blender_path', dest='blender_path', help='Path to blender executable', type=str)
    parser.add_argument('--models_path', dest='models_path', help='Path to T-Less CAD models', type=str)
    parser.add_argument('--dtd-rootdir', dest='dtd_rootdir', help='Rootdir of DTD dataset', type=str)
    parser.add_argument('--cpus', dest='num_cpus', default=1, type=int,
                        help='Number of processors for parallel computation')
    parser.add_argument('--cuda', dest='cuda_visible_devices')
    parser.add_argument('--nocuda', action='store_true')
    parser.add_argument('--pose_outpath', dest='pose_outpath', type=str, default='gt_poses')
    parser.add_argument('--size', dest='size', default=50, type=int, help='Size of the dataset')
    parser.add_argument('--start-idx', dest='start_index', default=0, type=int,
                        help='Index of the first generated sample')

    args = parser.parse_args()

    if not args.nocuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    final_yaml = {}
    # choose random parameters
    random.seed(645)
    np.random.seed(645)

    index_list = list(np.arange(start=args.start_index, stop=args.start_index + args.size, step=1))
    index_list = [index for index in index_list if not os.path.exists(os.path.join('multi_objects',
                                                                                   'images', 'img_%05d.png' % int(index)))]

    if args.num_cpus >= 2:
        pool = multiprocessing.Pool(processes=args.num_cpus)

        chunks = [index_list[j::args.num_cpus] for j in range(args.num_cpus)]
        results = [pool.apply_async(process_chunk, args=(chunk, args)) for chunk in chunks]

        r = [p.get() for p in results]
    else:
        r = process_chunk(index_list, args)
