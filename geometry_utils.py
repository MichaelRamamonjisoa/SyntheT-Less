import OpenEXR
import Imath
import numpy as np
import os
from skimage import feature
from skimage.morphology import skeletonize


def process_normals(exr_path):
    # process normals
    normals = OpenEXR.InputFile(exr_path)
    dw = normals.header()['dataWindow']

    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    normals_x = normals.channel('Image.X', Imath.PixelType(Imath.PixelType.FLOAT))
    normals_y = normals.channel('Image.Y', Imath.PixelType(Imath.PixelType.FLOAT))
    normals_z = normals.channel('Image.Z', Imath.PixelType(Imath.PixelType.FLOAT))

    normals = [normals_x, normals_y, normals_z]
    normals = [np.fromstring(norm, dtype='float32') for norm in normals]

    for norm in normals:
        norm.shape = (sz[1], sz[0])

    normals[0] = 0.5 * (normals[0] + 1)  # map [-1,1] to [0, 1]
    normals[1] = 0.5 * (normals[1] + 1)  # map [-1,1] to [0, 1]
    normals[2] = -0.5 * np.clip(normals[2], -1, 0) + 0.5  # map [0,-1] to [0.5, 1]

    norm_map_shape = (normals[0].shape[0], normals[0].shape[1], 3)

    normal_map = np.zeros(norm_map_shape, dtype='float32')

    normal_map[..., 0] = normals[0].astype('float32')
    normal_map[..., 1] = normals[1].astype('float32')
    normal_map[..., 2] = normals[2].astype('float32')

    normal_map = 255 * normal_map
    normal_map = normal_map.astype('uint8')

    os.remove(exr_path)

    return normal_map


def process_depth(exr_path):
    depth_exr = OpenEXR.InputFile(exr_path)
    dw = depth_exr.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    depth = depth_exr.channel('Image.V', Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.fromstring(depth, dtype='float32')
    depth.shape = (sz[1], sz[0])
    depth = depth.astype('float32') * 10000

    depth = depth.astype(np.uint16)
    mask = 255 * np.ones(shape=depth.shape, dtype='uint8')
    mask[depth == 0] = 0

    os.remove(exr_path)

    return depth, mask


def compute_occluding_contours(depth, instances_map, sigma=0.03, low_threshold=0.03, high_threshold=0.15):
    mask_depth = depth != 0
    mask_obj = instances_map != 0
    mask = np.logical_and(mask_depth, mask_obj)
    m = np.min(depth[mask])
    M = np.max(depth[mask])

    depth_norm = (depth.astype('float32') - m) / np.max([(M - m), 2000])
    depth_norm[mask == 0] = 0
    edges_depth = feature.canny(depth_norm, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    edges_instance = feature.canny(instances_map, sigma=0.0001, low_threshold=0.001, high_threshold=0.01)
    edges = np.logical_or(edges_depth, edges_instance)
    edges = skeletonize(edges)
    edges_out = edges.astype('uint8')
    edges_out = edges_out * 255

    return edges_out


def compute_all_contours(depth, instances_map, normals, sigma=0.05, low_threshold=0.03, high_threshold=0.15):
    mask_depth = depth != 0
    mask_obj = instances_map != 0
    mask = np.logical_and(mask_depth, mask_obj)
    m = np.min(depth[mask])
    M = np.max(depth[mask])

    depth_norm = (depth.astype('float32') - m) / np.max([(M - m), 2000])
    depth_norm[mask == 0] = 0
    normals[..., 0][mask == 0] = 127.5
    normals[..., 1][mask == 0] = 127.5
    normals[..., 2][mask == 0] = 127.5
    edges_depth = feature.canny(depth_norm, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    edges_instance = feature.canny(instances_map, sigma=0.0001, low_threshold=0.001, high_threshold=0.01)
    edges_normals_x = feature.canny(normals[..., 0], sigma=2, low_threshold=100, high_threshold=175)
    edges_normals_y = feature.canny(normals[..., 1], sigma=2, low_threshold=100, high_threshold=175)
    edges_normals_z = feature.canny(normals[..., 2], sigma=2, low_threshold=100, high_threshold=175)

    edges_normals = np.logical_or(np.logical_or(edges_normals_x, edges_normals_y), edges_normals_z)

    edges = np.logical_or(np.logical_or(edges_depth, edges_instance), edges_normals)

    edges = skeletonize(edges)
    edges_out = edges.astype('uint8')
    edges_out = edges_out * 255

    return edges_out
