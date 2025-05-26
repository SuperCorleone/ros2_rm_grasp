# 在你的 rm_grasp/launch 目录下，例如 my_final_robot.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    OpaqueFunction,
    TimerAction,
    LogInfo # 确保 LogInfo 已导入
)
from launch.substitutions import LaunchConfiguration, Command, TextSubstitution, PythonExpression
from ament_index_python.packages import get_package_share_directory
from launch.launch_context import LaunchContext
import xacro # 确保 xacro 已安装 (pip install xacro)
from launch.utilities import perform_substitutions

# 将 model.launch.py 中的 urdf 函数逻辑提取或适配过来
def get_robot_description_string(context: LaunchContext, robot_name_lc: LaunchConfiguration, model_lc: LaunchConfiguration) -> str:
    # 将 LaunchConfiguration 转换为实际字符串值
    robot_name_str = perform_substitutions(context, [robot_name_lc])
    model_str = perform_substitutions(context, [model_lc])

    urdf_xacro_path = os.path.join(
        get_package_share_directory('robomaster_description'), # 确保这个包名是正确的
        'urdf',
        f'robomaster_{model_str}.urdf.xacro'
    )
    
    print(f"[DEBUG Launch] XACRO path: {urdf_xacro_path}") # 调试路径
    if not os.path.exists(urdf_xacro_path):
        print(f"CRITICAL ERROR: XACRO file not found at {urdf_xacro_path}")
        return ""

    try:
        # xacro.process_file 需要一个文件名。
        # mappings 参数用于传递给 xacro 的内部参数 (相当于 xacro urdf_xacro_path name:='' model:='ep')
        # 确保你的 xacro 文件 robomaster_ep.urdf.xacro 能够接受一个空的 'name' 参数
        # 或者如果它不需要 'name' 参数，则 mappings 中不应包含它或为空
        mappings_for_xacro = {'model': model_str} # 'model' 通常是xacro需要的
        if robot_name_str: # 仅当 robot_name_str 非空时才传递 name 映射
            mappings_for_xacro['name'] = robot_name_str
        
        print(f"[DEBUG Launch] XACRO mappings: {mappings_for_xacro}")

        doc = xacro.process_file(urdf_xacro_path, mappings=mappings_for_xacro)
        return doc.toprettyxml(indent='  ')
    except Exception as e:
        print(f"Error processing XACRO for robot_description: {e}")
        return ''

# OpaqueFunction 需要一个接受 context 和其他通过kwargs传递的LaunchConfiguration的函数
def launch_rsp_and_jsp(context: LaunchContext, **kwargs):
    robot_name_lc = kwargs['robot_name_config'] # 从kwargs获取LaunchConfiguration对象
    model_lc = kwargs['model_config']

    robot_description_str = get_robot_description_string(context, robot_name_lc, model_lc)
    
    if not robot_description_str:
        # 如果 URDF 为空，打印更严重的错误，因为RSP会失败
        error_msg = "CRITICAL: robot_description string is empty. Robot State Publisher will fail. Check XACRO processing and paths."
        print(error_msg)
        # 可以在这里决定是否让launch失败，例如 raise Exception(error_msg)
        # 为保持原样，我们让它继续，RSP会报错

    # 获取解析后的 robot_name_str 用于命名空间和话题名
    robot_name_str_resolved = perform_substitutions(context, [robot_name_lc])

    # Robot State Publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        namespace=robot_name_str_resolved if robot_name_str_resolved else '',
        output='screen',
        parameters=[{
            'robot_description': robot_description_str,
            'use_sim_time': False # 根据你是否使用仿真时间调整
        }],
        arguments=["--ros-args", "--log-level", "warn"]
    )

    # Joint State Publisher
    # 构建 source_list 中的话题名
    jsp_source_topic = PythonExpression(["'/', '", robot_name_str_resolved, "/joint_states_p' if '", robot_name_str_resolved, "' else '/joint_states_p'"])


    jsp_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher', # 如果需要GUI版本，这里是 'joint_state_publisher_gui'
        name='joint_state_publisher',
        namespace=robot_name_str_resolved if robot_name_str_resolved else '',
        output='screen',
        parameters=[{
            # 'source_list': [TextSubstitution(text=f'{ "/" + robot_name_str_resolved if robot_name_str_resolved else "" }/joint_states_p')], # 这种方式更直接
            'source_list': [jsp_source_topic], # 使用 PythonExpression 动态构建话题名
            'rate': 10.0,
            'use_sim_time': False
        }],
        arguments=["--ros-args", "--log-level", "warn"]
    )
    return [rsp_node, jsp_node]


def generate_launch_description() -> LaunchDescription: # 修正返回类型注解
    # --- 0. 定义启动参数 ---
    model_arg = DeclareLaunchArgument(
        'model', default_value='ep', 
        description='Robot model (ep or s1)')
    
    # robot_name 参数不再由这个文件声明，但我们仍然创建一个 LaunchConfiguration
    # 对象，它会从命令行（如果提供）或父launch文件获取值，如果都没有则使用默认空字符串。
    robot_name_lc = LaunchConfiguration('robot_name', default='') 

    enable_gripper_arg = DeclareLaunchArgument(
        'enable_gripper', default_value='true', 
        description='Enable gripper module')
    
    # 获取其他 LaunchConfiguration 对象
    model_lc = LaunchConfiguration('model')
    enable_gripper_lc = LaunchConfiguration('enable_gripper')

    # --- 1. 启动 robomaster_driver ---
    robomaster_driver_node = Node(
        package='robomaster_ros',
        executable='robomaster_driver',
        name='robomaster', # 节点名保持 'robomaster'
        namespace=robot_name_lc, # 将解析为空字符串，即全局命名空间
        output='screen',
        parameters=[
            {'model': model_lc},
            {'conn_type': 'sta'}, 
            {'gripper/enabled': enable_gripper_lc},
            {'chassis_odom_twist_in_odom': True},
            # 如果驱动需要 tf_prefix 且希望它为空（因为没有命名空间了）
            # {'tf_prefix': ''} # 或者直接不设置，看驱动的默认行为
        ],
        # arguments=['--ros-args', '--log-level', 'debug']
    )
    
    # --- 2. & 3. 启动 Robot State Publisher 和 Joint State Publisher ---
    rsp_jsp_group = OpaqueFunction(
        function=launch_rsp_and_jsp,
        # 将 LaunchConfiguration 对象传递给 OpaqueFunction 的 kwargs
        kwargs={'robot_name_config': robot_name_lc, 'model_config': model_lc}
    )

    # --- 4. 启动你的 aruco_detector 节点 ---
    aruco_detector_node = Node(
        package='rm_grasp', 
        executable='aruco_detector',
        name='aruco_detector_node', 
        output='screen',
        # 由于没有 robot_name 命名空间了，如果 aruco_detector 或 main_controller
        # 内部硬编码了需要 /rm0/base_link 这样的frame，你需要将它们修改为使用
        # 全局的 'base_link'，或者通过参数传递 base_frame_id = ''
        # parameters=[{'robot_base_frame': ''}] # 示例
    )

    # --- 5. 延迟启动你的 main_controller 节点 ---
    main_controller_node = Node(
        package='rm_grasp',
        executable='main_controller',
        name='main_controller_node',
        output='screen',
        # parameters=[{'robot_base_frame': ''}] # 示例
    )
    
    delayed_main_controller = GroupAction(actions=[
        LogInfo(msg=["Waiting 5s for other nodes to stabilize before launching main_controller..."]), # 直接使用 LogInfo
        TimerAction(
            period=5.0,
            actions=[main_controller_node]
        )
    ])

    return LaunchDescription([
        model_arg,
        # robot_name_arg, # 已移除
        enable_gripper_arg,
        
        robomaster_driver_node, 
        rsp_jsp_group,          
        
        aruco_detector_node,
        delayed_main_controller
    ])