import numpy as np
from sklearn.mixture import GaussianMixture
import open3d as o3d
import pdb
from probreg.gmmtree import GMMTree

def plotOneEllipsoid(mean, cov):

    # _,D,_ = np.linalg.svd(cov)
    D, rot = np.linalg.eig(cov)
    # pdb.set_trace()
    
    a = D[0] * 0.5 # Semi-major axis - x
    b = D[1] * 0.5 # Semi-minor axis - y
    c = D[2] * 0.5 # Semi-minor axis - z
    # pdb.set_trace()

    # Create a mesh grid
    theta = np.linspace(0, np.pi, 20)          # Polar angle
    phi = np.linspace(0, 2 * np.pi, 20)   # Azimuthal angle
    Theta, Phi = np.meshgrid(theta, phi)
    
    # Calculate the coordinates of points on the ellipsoid surface
    X = a * np.sin(Theta) * np.cos(Phi)
    Y = b * np.sin(Theta) * np.sin(Phi)
    Z = c * np.cos(Theta)
    pts = np.dstack([X.reshape(-1),Y.reshape(-1),Z.reshape(-1), np.ones_like(Z.reshape(-1))]).squeeze().T
    T_mat = np.eye(4)
    T_mat[0:3,0:3] = rot
    T_mat[0:3,-1] = mean
    transformed_pts = T_mat @ pts
    # pdb.set_trace()
    return transformed_pts.T[:,0:3]
    


if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud('./src.ply')
    pcd = pcd.voxel_down_sample(2.5)
    points = np.asarray(pcd.points)
    print(points.shape)
    o3d.visualization.draw_geometries([pcd])
    ellipse_pts = []
    gm = GaussianMixture(n_components=1000, random_state=0, init_params='k-means++').fit(points)
    for i in range(len(gm.means_)):
        ellipse_pts.append(plotOneEllipsoid(gm.means_[i,:], gm.covariances_[i,:,:]))

    # pdb.set_trace()
    # gmm_tree = GMMTree(source=points, tree_level=4)
    # for _ell in gmm_tree._nodes:
    #     if _ell[0] != 0:
    #         ellipse_pts.append(plotOneEllipsoid(_ell[1], _ell[2]))
    
    ellipse_pts = np.concatenate(ellipse_pts, axis = 0)
    ellipse_pcd = o3d.geometry.PointCloud()
    ellipse_pcd.points = o3d.utility.Vector3dVector(ellipse_pts)
    o3d.io.write_point_cloud('./GMMS.ply', ellipse_pcd)



