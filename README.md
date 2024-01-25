# **LiDAR-Computer Vision Integration**

This task focuses on integrating LiDAR data with computer vision techniques for perception in autonomous driving scenarios. The implementation includes three main components: Point Cloud Registration, Instance Segmentation, and LiDAR to Camera Transformations.

**Point Cloud Registration**

The Problem1.py file contains functions for point cloud registration using the Iterative Closest Point (ICP) algorithm. The combine_clouds() function combines sequential frames using a registration matrix, and the register_clouds() function performs ICP on raw point clouds with noisy initial guesses from odometry. The result is a registered point cloud, improving the accuracy of the LiDAR data.

**Instance Segmentation**

To enhance the perception of moving vehicles, an instance segmentation step is implemented. The get_instances() function creates a mask to isolate dynamic objects on the road, focusing on moving vehicles. The clusters of points belonging to instances of vehicles are identified using the kMeans algorithm. Functions like cluster_dists(), new_centroids(), and num_instances() are implemented to perform unsupervised instance segmentation.

**LiDAR to Camera Transformations**

The final step involves transforming LiDAR points to pixels in the camera frame. The to_pixels() function takes 3D LiDAR points, the intrinsic matrix of the camera, and a transformation from LiDAR frame to camera frame. It returns the pixel values corresponding to the LiDAR points and provides depth information by transforming to the camera frame.

**Notes**<br />
The implementation is designed for autonomous driving scenarios, particularly for off-road driving datasets like RELLIS-3D.<br />
Instance segmentation is crucial for identifying individual moving vehicles on the road.<br />
LiDAR to camera transformations enable a comprehensive understanding of the environment by aligning point cloud data with camera images.
