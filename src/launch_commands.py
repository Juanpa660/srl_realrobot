#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import LaserScan, JointState, Imu
import tf.transformations

rospy.init_node('tbot', anonymous=True)
pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
initial_pose = PoseWithCovarianceStamped()
initial_pose.header.frame_id = 'map'  # the frame of reference, usually 'map'
initial_pose.pose.pose.position.x = 0
initial_pose.pose.pose.position.y = 0
theta = 1.2
q = tf.transformations.quaternion_from_euler(0, 0, theta)
initial_pose.pose.pose.orientation = Quaternion(*q)


lidar_msg = LaserScan()

def callback_lidar(scan):
    global lidar_msg
    lidar_msg = scan

lidar_sub = rospy.Subscriber('/scan', LaserScan, callback_lidar)

rate = rospy.Rate(1)

while not rospy.is_shutdown():
    print(lidar_msg)
    rate.sleep()


# Lidar launch
# roslaunch sick_tim sick_tim551_2050001.launch

# Robot launch
# roslaunch kobuki_node robot_with_tf.launch

# Localize launch
# roslaunch tbot localize.launch
# rostopic echo -n1 /amcl_pose

# roslaunch tbot init_particles.launch

# Control robot keyop launch
# roslaunch kobuki_keyop safe_keyop.launch



# To generate and save map
# Gmapping launch
# roslaunch tbot slam_gmapping.launch
# rosrun map_server map_saver -f <map_name>
