import numpy as np
from numpy.lib.arraysetops import isin
import trimesh
import math
import matplotlib.pyplot as plt
from skimage import io
from transforms3d.euler import euler2mat
#from shapely.geometry import Polygon
import matplotlib.cm as cm
from copy import deepcopy
import matplotlib
import os
import logging
from copy import deepcopy

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender

UVSPH_COUNT = [6, 6]


def render(
    mesh=None,
    shape=(480, 480),
    # pose
    camera_pose=None,
    object_pose=None,
    # color and light of main mesh
    use_mesh_material=False,
    mesh_material_color=[0.9, 0.9, 0.9, 1.0],
    light_intensity=4.0,
    # point cloud
    point_cloud=None,
    point_cloud_r=0.01,
    point_cloud_color=None,
    point_cloud_material_color=[0.0, 0.7, 1.0, 1.0],
    yfov=np.pi / 3.0,
    cam_dst_default=1.0,
    joint=None
):
    """
    :param mesh_fn:
    :param pose: T that transform object from canonical frame to camera frame
    """

    logging.debug("Rendering...")
    r = pyrender.OffscreenRenderer(shape[0], shape[1])

    # plot mesh
    material = (
        pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.0,
            alphaMode="BLEND",
            baseColorFactor=mesh_material_color,
        )
        if use_mesh_material
        else None
    )
    if camera_pose is None:
        # R = euler2mat(0, 0.0, 0, "rzyx")
        # R = euler2mat(0, np.pi / 2, 0, "rzyx")
        R = euler2mat(0, np.pi / 4, -np.pi / 6, "rzyx")
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = np.array([0.0, 0.0, cam_dst_default])
        camera_pose[:3, 3:4] = R @ camera_pose[:3, 3:4]
    scene = pyrender.Scene()
    if mesh is not None:
        if isinstance(mesh, list):
            for ind, m in enumerate(mesh):
                mesh_trimesh = as_mesh(m)  # mesh is a trimesh obj
                if object_pose is not None:
                    mesh_trimesh.apply_transform(object_pose[ind])
                scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, material=material))
        else:
            mesh_trimesh = as_mesh(mesh)  # mesh is a trimesh obj
            if object_pose is not None:
                mesh_trimesh.apply_transform(object_pose)
            scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, material=material))

    # plot point cloud
    if point_cloud is not None:
        # change to list
        if not isinstance(point_cloud, list):
            point_cloud = [point_cloud]
        if object_pose is not None:
            transformed_pc = []
            for pc in point_cloud:
                transformed_pc.append((object_pose[:3, :3] @ pc.T + object_pose[:3, 3:4]).T)
            point_cloud = transformed_pc
        if point_cloud_color is None:
            if not isinstance(point_cloud_material_color[0], list):
                point_cloud_material_color = [point_cloud_material_color]
            for pc, pc_m_c in zip(point_cloud, point_cloud_material_color):
                pc_material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    roughnessFactor=0.0,
                    alphaMode="BLEND",
                    baseColorFactor=pc_m_c,
                )
                sm = trimesh.creation.uv_sphere(radius=point_cloud_r, count=UVSPH_COUNT)
                sm.visual.vertex_colors = [0.0, 1.0, 0.0]
                tfs = np.tile(np.eye(4), (pc.shape[0], 1, 1))
                tfs[:, :3, 3] = pc
                m = pyrender.Mesh.from_trimesh(sm, poses=tfs, material=pc_material)
                scene.add(m)
        else:
            point_cloud_color_1 = 0.5 * np.ones_like(point_cloud_color[:,[0,0,0]],dtype=np.float)
            if point_cloud_color.shape[1] == 2:
                point_cloud_color_1[point_cloud_color[:,0]==1, 0] += 0.5
                point_cloud_color_1[point_cloud_color[:,1]==1, 1] += 0.5
            if point_cloud_color.shape[1] == 3:
                point_cloud_color_1[point_cloud_color[:,0]==1, 0] += 0.5
                point_cloud_color_1[point_cloud_color[:,1]==1, 2] += 0.5
                point_cloud_color_1[point_cloud_color[:,2]==1, 1] += 0.5

            if not isinstance(point_cloud_color_1, list):
                point_cloud_color_1 = [deepcopy(point_cloud_color_1)]
            sm = trimesh.creation.uv_sphere(radius=point_cloud_r, count=UVSPH_COUNT)
            sm.visual.vertex_colors = [1.0, 1.0, 1.0]
            for pc, pc_c in zip(point_cloud, point_cloud_color_1):
                vtx, faces, tex = [], [], []
                _nv = sm.vertices.shape[0]
                for cnt, pt in enumerate(pc):
                    vtx.append(sm.vertices.copy() + pt)
                    faces.append(sm.faces.copy() + cnt * _nv)
                    tex.append(np.ones_like(vtx[-1]) * pc_c[cnt][:3])
                vtx = np.concatenate(vtx, axis=0)
                faces = np.concatenate(faces, axis=0)
                tex = np.concatenate(tex, axis=0)
                pc_mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=tex)
                scene.add(pyrender.Mesh.from_trimesh(pc_mesh, material=None))
    
    if joint is not None:
        for i in range(joint.shape[0]):
            transform_a = np.asarray([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.15],
                [0, 0, 0, 1]
                ], dtype=np.float32)
            a = trimesh.creation.cone(0.02, 0.05, transform=transform_a)
            b = trimesh.creation.cylinder(0.01, height=0.3)
            joint_mesh = trimesh.util.concatenate(a,b)
            transform_joint = np.asarray([
                [1, 0, 0, joint[i,3]],
                [0, 1, 0, joint[i,4]],
                [0, 0, 1, joint[i,5]],
                [0, 0, 0, 1]
                ], dtype=np.float32)
            R = rodrigues_rotation_formula(np.array([0,0,1]), joint[i,:3]/np.linalg.norm(joint[i,:3]))
            transform_joint[:3,:3] = R
            joint_mesh.apply_transform(transform_joint)
            if i == 0:
                joint_mesh.visual.vertex_colors = np.array([0.9,0.9,0.9,1])
            elif i == 1:
                joint_mesh.visual.vertex_colors = np.array([0.1,0.1,0.1,1])
            else:
                joint_mesh.visual.vertex_colors = np.array([0.5,0.5,0.5,1])
            scene.add(pyrender.Mesh.from_trimesh(joint_mesh, material=None))

    # render
    camera = pyrender.PerspectiveCamera(
        yfov=yfov,
        aspectRatio=1.0,
    )

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    logging.debug("Render finished")

    return color, depth


def as_mesh(scene_or_mesh):
    """
    from: https://github.com/mikedh/trimesh/issues/507
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return deepcopy(mesh)

def rodrigues_rotation_formula(origin_vec, target_vec):
    c = np.dot(origin_vec, target_vec)
    n_vec = np.cross(origin_vec, target_vec)
    s = np.linalg.norm(n_vec)
    I = np.eye(3)
    n_vec_invert = np.array((
        [0, -n_vec[2], n_vec[1]],
        [n_vec[2], 0, -n_vec[0]],
        [-n_vec[1], n_vec[0], 0]
    ))

    R = I + n_vec_invert + np.dot(n_vec_invert, n_vec_invert) / (1+c)
    return R

if __name__ == "__main__":
    from transforms3d.euler import euler2mat
    import numpy as np

    R = euler2mat(0, 3 * np.pi / 4, np.pi / 6, "szyx")
    object_pose = np.eye(4)
    object_pose[:3, :3] = R
    object_pose[:3, 3] = np.array([0.0, 0.0, -0.1])
    # object_pose = np.array(
    #     [[1.0, 0, 0, 0.0],
    #      [0.0, 1.0, 0.0, 0.1],
    #      [0.0, 0.0, 1.0, -1.0],
    #      [0.0, 0.0, 0.0, 1.0]]
    # )
    tm = trimesh.load("./debug/bunny.obj")
    tm.visual.vertex_colors = np.random.uniform(size=tm.vertices.shape)
    render(tm, object_pose=object_pose, output_fn="./debug/debug.png")
