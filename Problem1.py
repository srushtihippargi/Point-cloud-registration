import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans

def register_clouds(xyz_source, xyz_target, trans_init=None):
  if trans_init is None:
    trans_init = np.eye(4)
  threshold = 0.1 
  max_iters = 100 

  source = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(xyz_source)

  target = o3d.geometry.PointCloud()
  target.points = o3d.utility.Vector3dVector(xyz_target)

  evaluation_pre = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
  print("Before registration:", evaluation_pre)

 
  reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)) 

  evaluation_post = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, reg_p2p.transformation)
  print("After registration:", evaluation_post)

 
  reg_mat = reg_p2p.transformation 
  return reg_mat

def combine_clouds(xyz_source, xyz_target, labels_source, labels_target, reg_mat):

  xyz_transformed = np.dot(xyz_source, reg_mat[:3,:3].T) + reg_mat[:3,3].reshape(1, 3) 
  xyz_all = np.concatenate((xyz_transformed, xyz_target), axis = 0) 
  label_all = np.concatenate((labels_source, labels_target), axis = 0) 
  return xyz_all, label_all

def mask_dynamic(xyz, label):

  dynamic_mask = np.logical_or(label == 1, label == 5) #None
  return xyz[dynamic_mask, :], label[dynamic_mask]



def mask_static(xyz, label):

  dynamic_mask = np.logical_or(label == 1, label == 5)
  static_mask = np.logical_not(dynamic_mask) #None
  return xyz[static_mask, :], label[static_mask]


def cluster_dists(xyz, clusters):
  N, __ = xyz.shape
  xyz = xyz.reshape(N, 1, 3)
  C = clusters.shape[0]

  closest_clusts = np.argmin(np.linalg.norm(xyz - clusters, axis=2), axis=1)# None
  return closest_clusts

def new_centroids(xyz, assignments, C):
  new_instances = np.zeros((C, 3))

  for i in range(C):
        new_instances[i] = np.mean(xyz[assignments == i], axis=0)
  return new_instances

def num_instances():

  return 5


def cluster(xyz):
  C = num_instances()
  rng = np.random.default_rng(seed=1)
  instances = xyz[rng.choice(xyz.shape[0], size=C, replace=False), :]
  prev_assignments = rng.choice(C, size=xyz.shape[0])
  while True:
    assignments = cluster_dists(xyz, instances)
    instances = new_centroids(xyz, assignments, C)
    if (assignments == prev_assignments).all():
      return instances, assignments
    prev_assignments = assignments

def cluster_sci(xyz):
  kmeans = KMeans(n_clusters=num_instances(), random_state=5, n_init="auto").fit(xyz)
  clustered_labels = kmeans.predict(xyz)
  return kmeans.cluster_centers_, clustered_labels

def to_pixels(xyz, P, RT):

  transformed_points = np.dot(RT[:3, :3], xyz.T).T + RT[:3, 3].reshape(1, 3)

  d = transformed_points[:, 2] 
  image_points = np.dot(P, transformed_points.T).T 
  image_x = image_points[:, 0] / image_points[:, 2] 
  image_y = image_points[:, 1] / image_points[:, 2] 

  imgpoints = np.stack([image_x, image_y], axis=1)
  return imgpoints, d