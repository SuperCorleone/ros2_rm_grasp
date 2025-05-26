# ros2_rm_grasp

colcon build --symlink-install

source install/setup.zsh

1 terminal
ros2 launch robomaster_ros main.launch device:="tcp:host=localhost;port=33333" simulation:=True

1 terminal
ros2 run rm_grasp aruco_detector 

1 terminal
ros2 run rm_grasp main_controller
