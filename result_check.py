import open3d as o3d
import numpy as np
import torch

dir = 'log/cadex_pm_eyeglasses_dep/weight/'
b, p = 0, 0
axis_t = torch.load(dir + 'c_trans_481.pt').detach().cpu().numpy()
length = torch.load(dir + 'c_length_481.pt').detach().cpu().numpy()
axis_o = torch.load(dir + 'c_axis_481.pt').detach().cpu().numpy()
query = torch.load(dir + 'query_481.pt').detach().cpu().numpy().reshape(6,3,4,-1,3)
weights = torch.load(dir + 'weights_481.pt').detach().cpu().numpy()

query = query.reshape(6,3,-1,3)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(query[b, p])
o3d.visualization.draw_geometries([pc])