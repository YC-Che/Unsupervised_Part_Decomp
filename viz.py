import open3d as o3d
import numpy as np
import scipy.io as sio
from glob import glob
# epe

#data = sio.loadmat('resource/data/partMobility/enhance_data/test/valve/split_test_fps_1_0002_5.mat')['data'][0]['input'][0][:,:,:3]
data_1 = np.load('./aligned_pc_1.npz', allow_pickle=True)['arr_0']#T,N,3
data_2 = np.load('./aligned_pc_2.npz', allow_pickle=True)['arr_0']#T,N,3
print(data_1.shape)

pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(data_1[:,:,:].reshape(-1, 3))

pc2 = o3d.geometry.PointCloud()
pc2.points = o3d.utility.Vector3dVector(data_2[:,:,:].reshape(-1, 3))
pc2.translate([0,1,0])
o3d.visualization.draw_geometries([pc, pc2])


#dataset_viz
'''
file_list = glob('resource/data/partMobility/enhance_data/test/washing_machine/*')
for i, file in enumerate(file_list):
    print(i)
    mat = sio.loadmat(file)['data'][0]
    points = mat['input'][0]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    o3d.visualization.draw_geometries([pc])
'''