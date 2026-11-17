#!/usr/bin/env python3

import rospy
import roslaunch
import subprocess
import time
import std_msgs
import std_srvs
import std_srvs.srv  # Add this import at the top of your script
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import os

import time
import os
import signal

MODEL_NAME = "scannet22_swinmtl_onehot_evid" # Replace with actual model name
ROOT_PATH = "/home/user/menon/workspace/ros1/menon25iros_ws/src/evaluation_3d/mapping_scenes/sens"  # Replace with actual root path

OUTPUT_PATH = f"/home/user/menon/workspace/ros1/menon25iros_ws/src/kimera_semantics_evidential/kimera_semantics_evidential_ros/evaluation_results/{MODEL_NAME}"
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.makedirs(f"{ROOT_PATH}/MODEL_NAME", exist_ok=True)

# List of scenes to process
#SCENES = ["scene0013_00", "scene0019_00", "scene0031_02", "scene0093_02", "scene0115_02", "scene0160_03", "scene0275_00",  "scene0369_02", "scene0480_01", "scene0624_00"]  # Replace with actual scene names
#SCENES = [ "scene0031_02", "scene0093_02", "scene0115_02"]#, "scene0160_03", "scene0275_00",  "scene0369_02", "scene0480_01", "scene0624_00"]  # Replace with actual scene names

SCENES = ["scene0013_00", "scene0019_00",  "scene0115_02", "scene0369_02", "scene0480_01"]
#SCENES = [ "scene0480_01"]  # Replace with actual scene names ["",

print("Number of scenes: ", len(SCENES))
# Paths to launch files
KIMERA_LAUNCH = "/home/user/menon/workspace/ros1/menon25iros_ws/src/kimera_semantics_evidential/kimera_semantics_evidential_ros/launch/scannet_kimera_semantics.launch"
SWINMTL_LAUNCH = "/home/user/menon/workspace/ros1/menon25iros_ws/src/swin_mtl_evidential/launch/main_evidential_semantic_cloud_creator.launch"

#!/usr/bin/env python3

import rospy
import roslaunch
import subprocess
import time
import os
import signal

def save_pcd(points, file_name):
    """
    Saves a list of (x, y, z) points to a PCD file in ASCII format.
    
    :param points: List of tuples representing the points.
    :param file_name: The name of the PCD file to write.
    """
    num_points = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH {0}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS {0}\n"
        "DATA ascii\n"
    ).format(num_points)
    
    with open(file_name, MODEL_NAME, "w") as f:
        f.write(header)
        for p in points:
            f.write("{0} {1} {2}\n".format(p[0], p[1], p[2]))

def capture_and_save_pcd(topic_name, pcd_filename="cloud_one_shot.pcd"):
    """
    Waits for a single PointCloud2 message on the specified topic, converts it to points,
    and saves it to a PCD file.

    :param topic_name: ROS topic name to listen for PointCloud2 messages.
    :param pcd_filename: Filename for the saved PCD file.
    """
    # Wait for one message (one-shot capture)
    msg = rospy.wait_for_message(topic_name, PointCloud2)
    rospy.loginfo("Received point cloud: width=%d, height=%d", msg.width, msg.height)
    
    # Convert the PointCloud2 message to a list of (x, y, z) points
    points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
    
    # Save the points to a PCD file
    save_pcd(points, pcd_filename)
    rospy.loginfo("Saved point cloud to %s", pcd_filename)

class LaunchFile:
  def __init__(self, package, file, args=[]):
    self.file = [(roslaunch.rlutil.resolve_launch_arguments([package, file])[0], args)]
    self.is_launched = False

  def launch(self, uuid):
    self.parent = roslaunch.parent.ROSLaunchParent(uuid, self.file)
    self.parent.start()
    self.is_launched = True
    return self.parent

  def spin(self):
    self.parent.spin_once()

  def shutdown(self):
    print('Shutting down')
    self.parent.shutdown()
    print('Shutdown complete')

def is_roscore_running():
  """Check if roscore is already running."""
  try:
    subprocess.check_output(["rosnode", "list"])
    return True
  except subprocess.CalledProcessError:
    return False

def start_roscore():
  """Start roscore if not already running."""
  if not is_roscore_running():
    rospy.loginfo("Starting roscore...")
    subprocess.Popen(["roscore"])
    time.sleep(3)  # Give it time to initialize

def kill_ros_node(node_name):
  """Kill a running ROS node by its name."""
  try:
    node_list = subprocess.check_output(["rosnode", "list"]).decode("utf-8").split("\n")
    for node in node_list:
      if node_name in node:
        rospy.loginfo(f"Shutting down node: {node}")
        subprocess.call(["rosnode", "kill", node])
  except Exception as e:
    rospy.logwarn(f"Could not kill node {node_name}: {e}")

def is_ros_node_running(node_name):
  """Check if a ROS node is still running."""
  try:
    node_list = subprocess.check_output(["rosnode", "list"]).decode("utf-8").split("\n")
    return node_name in node_list
  except Exception:
    return False

def main():
  rospy.init_node("evaluate_mapping_pipeline", anonymous=True)

  # ✅ Ensure ROS Master is running
  start_roscore()

  uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
  roslaunch.configure_logging(uuid)

  for scene in SCENES:
    rospy.loginfo(f"Processing scene: {scene}")

    # Define launch files
    kimera = LaunchFile("kimera_semantics_evidential_ros", "scannet_kimera_semantics.launch", [f"mesh_filename:=$(find kimera_semantics_evidential_ros)/evaluation_results/{MODEL_NAME}/{MODEL_NAME}_{scene}.ply"])
    swinmtl = LaunchFile("swin_mtl_evidential", "main_evidential_semantic_cloud_creator.launch", [f"root_path:=$(find swin_mtl_evidential)/data/mapping_scenes/sens/{scene}"])
    
    rospy.set_param("/kimera_semantics_node/mesh_filename", f"$(find kimera_semantics_evidential_ros)/evaluation_results/{MODEL_NAME}/{MODEL_NAME}_{scene}")
    mesh_filename = rospy.get_param("/kimera_semantics_node/mesh_filename")
    rospy.logwarn(f"Kimera mesh filename settting to: {mesh_filename}")
    # ✅ Start Kimera and SwinMTL in parallel
    kimera_process = kimera.launch(uuid)
    rospy.sleep(5)  # Give it some time to start

    mesh_filename = rospy.get_param("/kimera_semantics_node/mesh_filename")
    rospy.logwarn(f"Kimera mesh filename set to: {mesh_filename}")

    swinmtl_process = swinmtl.launch(uuid)

    rospy.sleep(5)  # Give it some time to start

    # ✅ Wait for SwinMTL to finish first
    swinmtl_node = "/perception_node" 
    while is_ros_node_running(swinmtl_node):
      rospy.sleep(2)

    rospy.loginfo(f"SwinMTL completed for {scene}.")
    rospy.sleep(5)  # Give it some time to finish

    # ✅ Trigger mesh generation in Kimera
    rospy.wait_for_service("/kimera_semantics_node/generate_mesh")
    try:
      generate_mesh = rospy.ServiceProxy("/kimera_semantics_node/generate_mesh", std_srvs.srv.Empty)
      generate_mesh()
      rospy.loginfo("Kimera mesh generation triggered.")
    except rospy.ServiceException as e:
      rospy.logwarn(f"Service call to /kimera_semantics_node/generate_mesh failed: {e}")
    #pcd_filename = f"{ROOT_PATH}/{MODEL_NAME}/{scene}.pcd"
    #capture_and_save_pcd("/kimera_semantics_node/surface_pointcloud", pcd_filename)

    # ✅ Give Kimera some time to generate the mesh
    rospy.sleep(10)

    # ✅ Kill Kimera process
    kimera.shutdown()
    kill_ros_node("/kimera_semantics_node")  # Replace with actual node name

    rospy.loginfo(f"Kimera-Semantics stopped for {scene}.")

  rospy.loginfo("All scenes processed!")

if __name__ == "__main__":
  main()
