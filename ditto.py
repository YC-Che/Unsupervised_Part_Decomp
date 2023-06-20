import scipy.io as sio
import numpy as np
import torch
import glob
from core.models.utils.evaluation import eval_atc_all
import trimesh
import os
import open3d as o3d
import matplotlib.pyplot as plt
from transforms3d.euler import euler2mat
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
#import pyrender

ditto_data_path = '/nhome/yuchen_che/ditto_dataset/train/microwave_train_1K/scenes/'
data = np.load(ditto_data_path + 'a24e299bf36c44308bf5aa6f29bd08bf.npz', allow_pickle=True)

data_pc = data['pc_end']
data_pc /= data['start_mesh_pose_dict'].item()['0_0'][0][1][0]
data_pc /= 2
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(data_pc)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
axis_l = np.cross(data['screw_axis'], data['screw_moment'])
axis_l /= data['start_mesh_pose_dict'].item()['0_0'][0][1]
mesh_frame.translate(axis_l)

print(data['screw_axis'])
o3d.visualization.draw_geometries([pc, mesh_frame])