import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import AnyLaunchDescriptionSource # 更通用，可以处理 .launch, .launch.xml, .launch.py
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 获取 robomaster_ros 包的共享目录路径
    robomaster_ros_pkg_share = get_package_share_directory('robomaster_ros')
    
    # 1. 包含 robomaster_ros 的 main.launch 文件
    # 根据你的要求，我们不传递或将相机/TOF相关的参数设置为 'false'
    # 假设 'false' 会让 main.launch 禁用它们，或者 main.launch 默认不启动它们除非显式设置为 'true'
    
    # 这些是你想保留的非传感器参数
    main_launch_args_to_pass = {
        'model': 'ep',
        'enable_gripper': 'true',
        'chassis_odom_twist_in_odom': 'true',
        # 你注释掉的传感器参数，我们显式设为 'false' 或不传递
        # 'camera_0': 'false',
        # 'camera_1': 'false',
        # 'tof_0': 'false',
        # 'tof_1': 'false',
        # 'tof_2': 'false',
        # 'tof_3': 'false',
        # # 相机尺寸和帧率参数，如果相机被禁用，它们可能无意义，但为完整起见可以传递
        # # 或者如果 main.launch 在这些参数未提供时会报错，则需要传递
        # 'camera_width': '640', # 保持你原有的值
        # 'camera_height': '360',
        # 'camera_fps': '30',
    }

    robomaster_main_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource( # 使用AnyLaunchDescriptionSource更安全
            os.path.join(robomaster_ros_pkg_share, 'launch', 'main.launch') # 指向实际的launch文件
        ),
        launch_arguments=main_launch_args_to_pass.items()
    )

    # 2. 启动 aruco_detector 节点
    aruco_detector_node = Node(
        package='rm_grasp',
        executable='aruco_detector', # 确保这是你在CMakeLists.txt或setup.py中定义的可执行文件名
        name='aruco_detector_node',   # 与你日志中一致的节点名
        output='screen'
    )

    delay_for_aruco_detector_start = TimerAction(
        period=2.0, # 延时5秒 (根据你的系统和 robomaster_ros 的启动速度调整)
        actions=[
            LogInfo(msg="Initial 5-second delay for robomaster_ros setup complete. Launching aruco_detector."),
            aruco_detector_node
        ]
    )

    # 3. 定义 main_controller 节点的启动行为 (包含gdb前缀)
    main_controller_node_action = Node(
        package='rm_grasp',
        executable='main_controller', # 确保可执行文件名正确
        name='main_controller_node',  # 与你日志中一致的节点名
        output='screen'
    )

    # 4. 创建一个事件处理器，当aruco_detector_node启动后，触发一个3秒的定时器，定时器结束后再启动main_controller_node
    delayed_main_controller_launch = RegisterEventHandler(
        OnProcessStart(
            target_action=aruco_detector_node, # 监听aruco_detector_node的启动事件
            on_start=[
                LogInfo(msg="aruco_detector_node has started. Initiating 3-second delay for main_controller_node."),
                TimerAction(
                    period=2.0, # 延时3秒
                    actions=[
                        LogInfo(msg="3-second delay complete. Launching main_controller_node."),
                        main_controller_node_action # 延时结束后执行的动作：启动main_controller
                    ]
                )
            ]
        )
    )

    return LaunchDescription([
        #robomaster_main_launch,
        aruco_detector_node,
        delayed_main_controller_launch
    ])