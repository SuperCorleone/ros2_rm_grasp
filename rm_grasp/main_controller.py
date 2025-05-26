import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.logging import LoggingSeverity

import numpy as np
from geometry_msgs.msg import Twist, Vector3, PoseStamped, Transform
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from robomaster_msgs.action import GripperControl # 确保这个action定义是正确的
from action_msgs.msg import GoalStatus as ActionMsgGoalStatus

from enum import Enum
import math
import traceback
import time 

class TaskState(Enum):
    SCAN = 1
    APPROACH = 2
    ALIGN = 3
    # GRASP = 4 # 旧的GRASP状态
    TRANSPORT = 5
    RELEASE = 6
    DONE = 7
    ERROR = 8
    # 移除旧的GRASP_INITIATE状态，用一个新状态替代
    # GRASP_INITIATE_OPEN = 9
    # GRASP_INITIATE_CLOSE = 10
    # ARM_LIFT_AFTER_GRASP = 11
    PERFORM_GRASP_SEQUENCE = 12 # 新的状态，用于执行完整的抓取序列
    # SIMPLIFIED_GRASP_AND_LIFT = 13 # 如果之前有这个，顺延或移除


def quaternion_to_euler_yaw(q_x, q_y, q_z, q_w):
    siny_cosp = 2 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller_node')
        self.get_logger().set_level(LoggingSeverity.DEBUG)

        self.state = TaskState.SCAN
        self.get_logger().info(f"Initializing MainController. Initial state: {self.state.name}")

        self.latched_target_transform: Transform = None

        self.tf_buffer_cache_time_sec = 20.0
        self.tf_buffer = Buffer(cache_time=Duration(seconds=self.tf_buffer_cache_time_sec))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_lookup_timeout_sec = 0.05

        # *** 修改: 确保Action Server名称与第一个脚本或CoppeliaSim设置一致 ***
        # 例如，如果第一个脚本用 '/RoboMaster/gripper' 并且能工作
        self.gripper_client = ActionClient(self, GripperControl, '/gripper')
        self.gripper_power = 0.8
        self.gripper_target_state_for_feedback = None # 用于在feedback_callback中判断
        self.gripper_action_in_progress = False # 标记夹爪动作是否正在进行
        self.next_step_after_gripper = None # 记录夹爪动作完成后要执行的函数

        self.current_pose = None
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.tof_distance = None
        self.tof_sub = self.create_subscription(Range, '/tof', self.tof_cb, 10)

        # *** 修改: 确保机械臂话题名称与第一个脚本或CoppeliaSim设置一致 ***
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(Vector3, '/cmd_arm', 10) # 或者 '/arm_cmd'

        # --- 控制参数 (保持不变，除非特别说明) ---
        self.scan_angular_speed = 0.5
        self.scan_target_max_dist_m = 5.0
        self.grasp_dist_m = 0.05
        self.approach_align_dx_threshold_m = 0.05
        self.approach_align_dy_threshold_m = 0.03
        self.approach_far_linear_gain = 0.8
        self.approach_far_angular_gain = 0.4
        self.approach_far_max_linear_vel_ms = 0.3
        self.approach_near_threshold_add_m = 0.05
        self.approach_near_linear_gain = 0.3
        self.approach_near_angular_gain = 0.5
        self.approach_near_max_linear_vel_ms = 0.1
        self.approach_max_angular_vel_rads = self.scan_angular_speed
        self.approach_motion_deadband_dx_m = 0.015
        self.approach_motion_deadband_dy_m = 0.015
        self.align_arm_z_pos = 0.12 # 手臂预抓取高度
        self.gripper_open_delay_sec = 1.0 # 夹爪打开后的延时
        self.gripper_close_delay_sec = 1.0 # 夹爪关闭后的延时 (如果需要)
        self.arm_lift_z_pos_after_grasp = 0.18
        self.arm_lift_duration_sec = 1.5 # 手臂抬升后的延时

        self.transport_target_pos_xyz = [1.0, 1.0, 0.0]
        self.transport_target_reached_threshold_m = 0.15
        self.transport_turn_angle_threshold_rad = math.radians(10.0)
        self.transport_creep_linear_vel_ms = 0.05
        self.transport_linear_gain = 0.4
        self.transport_angular_gain = 0.7
        self.transport_max_linear_vel_ms = 0.3
        self.transport_max_angular_vel_rads = 0.4
        self.transport_fine_tune_angular_gain_factor = 0.5

        self.control_cycle_period_sec = 0.1
        self.timer = self.create_timer(self.control_cycle_period_sec, self.control_cycle_wrapper) # 修改为wrapper

        self.get_logger().info("MainController initialized with parameters.")

    def odom_cb(self, msg: Odometry):
        self.current_pose = msg.pose.pose

    def tof_cb(self, msg: Range):
        self.tof_distance = msg.range

    # 修改: _set_state 不再是 async
    def _set_state(self, new_state: TaskState):
        if self.state != new_state:
            self.get_logger().info(f"State transition: {self.state.name} -> {new_state.name}")
            self.state = new_state

    # 修改: control_cycle_wrapper 来处理 async control_cycle
    def control_cycle_wrapper(self):
        # 这个包装器允许我们从同步的定时器回调中调用异步的 control_cycle
        # 但由于我们正在移除 control_cycle 中的 async/await，这个包装器可能不再需要
        # 暂时保留，如果 control_cycle 完全同步了，可以直接调用
        # asyncio.run(self.control_cycle()) # 这在ROS 2节点中通常不这么用
        # 更简单的方式是让 control_cycle 本身不再是 async
        self.control_cycle()


    # 修改: control_cycle 不再是 async，内部调用的状态处理函数也不是 async
    def control_cycle(self):
        if self.gripper_action_in_progress:
            self.get_logger().debug(f"Gripper action in progress, skipping control cycle for state {self.state.name}")
            return

        self.get_logger().debug(f"Current state: {self.state.name}")
        try:
            handler_name = self.state.name.lower() + "_state_handler"
            current_state_handler = getattr(self, handler_name, None)
            if current_state_handler and callable(current_state_handler):
                current_state_handler() # 直接调用，不再 await
            else:
                self.get_logger().error(f"No valid handler for state: {self.state.name} (tried: {handler_name})")
                self._set_state(TaskState.ERROR)
        except Exception as e:
            self.get_logger().error(f"Error in control cycle for state {self.state.name}: {str(e)}\n{traceback.format_exc()}")
            self._set_state(TaskState.ERROR)
            self.cmd_vel_pub.publish(Twist())

    # 状态处理函数不再是 async
    def scan_state_handler(self):
        self.get_logger().debug("[SCAN] Executing.")
        # ... (SCAN 逻辑保持不变, 但移除 await)
        if self.latched_target_transform is not None:
            self.get_logger().info("[SCAN] Clearing previously latched target transform.")
            self.latched_target_transform = None
        twist = Twist()
        twist.angular.z = self.scan_angular_speed
        self.cmd_vel_pub.publish(twist)
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'target_marker', rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec))
            trans_x = transform.transform.translation.x
            self.get_logger().info(f"[SCAN] TF LOOKUP SUCCESS! Target at x: {trans_x:.3f} m.")
            if 0 < trans_x < self.scan_target_max_dist_m:
                self.cmd_vel_pub.publish(Twist())
                self._set_state(TaskState.APPROACH) # 不再 await
            else:
                self.get_logger().debug(f"[SCAN] Target at {trans_x:.3f}m. Condition not met. Continuing SCAN.")
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"[SCAN] TF lookup failed: {type(e).__name__} - {str(e)}. Continuing SCAN.")
        except Exception as e:
            self.get_logger().error(f"[SCAN] Unexpected error: {str(e)}\n{traceback.format_exc()}")
            self._set_state(TaskState.ERROR) # 不再 await

    def approach_state_handler(self):
        self.get_logger().debug("[APPROACH] Executing.")
        # ... (APPROACH 逻辑保持不变, 但移除 await)
        try:
            current_tf_lookup = self.tf_buffer.lookup_transform(
                'base_link', 'target_marker', rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_lookup_timeout_sec))
            trans_x = current_tf_lookup.transform.translation.x
            trans_y = current_tf_lookup.transform.translation.y
            # ... (其余计算和发布逻辑)
            dx = trans_x - self.grasp_dist_m
            dy = trans_y
            # ... (PID和速度裁剪逻辑) ...
            twist = Twist() # 假设已经计算好
            # ... (计算 final_linear_x, final_angular_z) ...
            final_linear_x = 0.0 # 占位
            final_angular_z = 0.0 # 占位
            # (此处应有实际的速度计算逻辑，从原代码复制)
            if abs(trans_x) < (self.grasp_dist_m + self.approach_near_threshold_add_m):
                current_linear_gain = self.approach_near_linear_gain
                current_angular_gain = self.approach_near_angular_gain
                current_max_linear_vel = self.approach_near_max_linear_vel_ms
            else:
                current_linear_gain = self.approach_far_linear_gain
                current_angular_gain = self.approach_far_angular_gain
                current_max_linear_vel = self.approach_far_max_linear_vel_ms
            current_max_angular_vel = self.approach_max_angular_vel_rads
            desired_linear_x = current_linear_gain * dx
            desired_angular_z = -current_angular_gain * dy
            final_linear_x = np.clip(desired_linear_x, -current_max_linear_vel, current_max_linear_vel)
            final_angular_z = np.clip(desired_angular_z, -current_max_angular_vel, current_max_angular_vel)
            if abs(dx) < self.approach_motion_deadband_dx_m: final_linear_x = 0.0
            if abs(dy) < self.approach_motion_deadband_dy_m: final_angular_z = 0.0
            twist.linear.x = final_linear_x
            twist.angular.z = final_angular_z


            if abs(dx) < self.approach_align_dx_threshold_m and abs(dy) < self.approach_align_dy_threshold_m:
                self.get_logger().info(f"[APPROACH] Target ALIGNED. Latching target pose.")
                self.cmd_vel_pub.publish(Twist())
                try:
                    self.latched_target_transform = current_tf_lookup.transform
                    self._set_state(TaskState.ALIGN) # 不再 await
                except AttributeError:
                    self.get_logger().error(f"[APPROACH] Failed to latch target pose. Returning to SCAN.")
                    self.latched_target_transform = None
                    self._set_state(TaskState.SCAN) # 不再 await
                return
            self.cmd_vel_pub.publish(twist)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"[APPROACH] TF lookup failed. Switching to SCAN.")
            self.latched_target_transform = None
            self._set_state(TaskState.SCAN) # 不再 await
        except Exception as e:
            self.get_logger().error(f"[APPROACH] Unexpected error: {str(e)}\n{traceback.format_exc()}")
            self.latched_target_transform = None
            self._set_state(TaskState.ERROR) # 不再 await


    def align_state_handler(self):
        if self.latched_target_transform is None:
            self.get_logger().error("[ALIGN] No latched target pose. Returning to SCAN.")
            self._set_state(TaskState.SCAN)
            return

        self.get_logger().info(f"[ALIGN] Lowering arm to Z={self.align_arm_z_pos}. Switching to PERFORM_GRASP_SEQUENCE.")
        # 假设手臂控制器能处理目标Z值
        self.arm_pub.publish(Vector3(x=0.0, y=0.0, z=self.align_arm_z_pos)) # 明确x,y
        # 增加一个小延时确保手臂指令发出，但理想情况下应有手臂动作的反馈
        time.sleep(0.5) # 短暂延时，注意这会阻塞，更好的方式是手臂也有动作状态
        self._set_state(TaskState.PERFORM_GRASP_SEQUENCE)


    # 新的夹爪动作发送函数 (模仿第一个脚本的 send_gripper_command)
    def _send_gripper_goal(self, target_state_value, next_step_callable=None):
        if self.gripper_action_in_progress:
            self.get_logger().warn("Gripper action already in progress, new goal ignored.")
            return

        if not self.gripper_client.server_is_ready():
            self.get_logger().error("Gripper action server not ready!") # <--- 修正后
            self._set_state(TaskState.ERROR) # 或者尝试重新连接/等待
            return

        self.gripper_target_state_for_feedback = target_state_value
        self.next_step_after_gripper = next_step_callable
        self.gripper_action_in_progress = True

        goal_msg = GripperControl.Goal()
        goal_msg.target_state = target_state_value
        goal_msg.power = self.gripper_power

        self.get_logger().info(f"Sending gripper goal: {'OPEN' if target_state_value == GripperControl.Goal.OPEN else 'CLOSE'} with power {goal_msg.power}")
        send_goal_future = self.gripper_client.send_goal_async(
            goal_msg,
            feedback_callback=self._gripper_feedback_callback  # 使用新的反馈回调
        )
        send_goal_future.add_done_callback(self._gripper_goal_response_callback)

    def _gripper_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle:
            self.get_logger().error('Gripper goal future failed to return a handle.')
            self.gripper_action_in_progress = False
            self._set_state(TaskState.ERROR)
            return

        if not goal_handle.accepted:
            self.get_logger().info('Gripper command rejected')
            self.gripper_action_in_progress = False
            # 决定下一步，例如回到某个状态或报错
            self._set_state(TaskState.ALIGN) # 例如，回到对齐状态重试
            return

        self.get_logger().info('Gripper command accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._gripper_get_result_callback)

    def _gripper_feedback_callback(self, feedback_msg):
        # 假设 feedback_msg.feedback 包含 GripperControl.Feedback 的内容
        # 并且 GripperControl.Feedback 有一个字段叫 current_state (需要核实 .action 文件)
        feedback = feedback_msg.feedback
        if hasattr(feedback, 'current_state'): # 确保 feedback 对象有 current_state 属性
            self.get_logger().info(f'Gripper Feedback: Current state - {feedback.current_state}')
            # 如果达到了我们发送的目标状态
            if feedback.current_state == self.gripper_target_state_for_feedback:
                self.get_logger().info(f"Gripper reached target state: {self.gripper_target_state_for_feedback} via feedback.")
                # self.gripper_action_in_progress = False # 不在此处标记完成，等待 result
                # 如果有下一步，则执行 (模仿第一个脚本的行为)
                if self.next_step_after_gripper and callable(self.next_step_after_gripper):
                    # 防止重复调用，清除
                    step_to_call = self.next_step_after_gripper
                    self.next_step_after_gripper = None
                    self.get_logger().info("Executing next step after gripper feedback.")
                    step_to_call()
        else:
            self.get_logger().debug(f'Gripper Feedback received (current_state field missing or structure unknown).')


    def _gripper_get_result_callback(self, future):
        result_wrapper = future.result()
        if not result_wrapper:
            self.get_logger().error('Gripper result future failed.')
            self.gripper_action_in_progress = False
            self._set_state(TaskState.ERROR)
            return

        status = result_wrapper.status
        result = result_wrapper.result # GripperControl.Result

        if status == ActionMsgGoalStatus.STATUS_SUCCEEDED:
            # 假设 result 包含一个名为 'success' 或 'achieved_state' 的字段 (需核实.action)
            log_msg = f"Gripper action SUCCEEDED."
            if hasattr(result, 'success') and result.success:
                log_msg += " Result: success=true."
            elif hasattr(result, 'achieved_state'):
                 log_msg += f" Result: achieved_state={result.achieved_state}."
            self.get_logger().info(log_msg)
        else:
            self.get_logger().error(f'Gripper action failed with status: {status}')
            # 决定下一步

        self.gripper_action_in_progress = False # 动作真正结束
        # 如果没有通过feedback触发下一步，可以在这里根据最终结果触发
        if self.next_step_after_gripper and callable(self.next_step_after_gripper) and status == ActionMsgGoalStatus.STATUS_SUCCEEDED:
            step_to_call = self.next_step_after_gripper
            self.next_step_after_gripper = None
            self.get_logger().info("Executing next step after gripper result SUCCEEDED.")
            step_to_call()
        elif status != ActionMsgGoalStatus.STATUS_SUCCEEDED:
             self.get_logger().warn("Gripper action did not succeed, not calling next_step_after_gripper if any.")
             self._set_state(TaskState.ALIGN) # 例如，失败了回到对齐状态

    # 新的状态处理函数
    def perform_grasp_sequence_state_handler(self):
        self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Initiating grasp sequence.")
        self.get_logger().info("Step 1: Opening gripper.")
        # 传递下一步操作 (关闭夹爪)
        self._send_gripper_goal(GripperControl.Goal.OPEN, next_step_callable=self._initiate_close_gripper_after_open)

    def _initiate_close_gripper_after_open(self):
        self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Gripper presumably open (triggered by feedback/result).")
        self.get_logger().info(f"Delaying for {self.gripper_open_delay_sec}s before closing...")
        # 注意: time.sleep() 会阻塞当前线程 (定时器回调)。
        # 在生产代码中，应使用ROS定时器或更复杂的异步机制来实现非阻塞延时。
        time.sleep(self.gripper_open_delay_sec)

        self.get_logger().info("Step 2: Closing gripper.")
        # 传递下一步操作 (抬升手臂)
        self._send_gripper_goal(GripperControl.Goal.CLOSE, next_step_callable=self._initiate_lift_arm_after_close)

    def _initiate_lift_arm_after_close(self):
        self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Gripper presumably closed (triggered by feedback/result).")
        self.get_logger().info(f"Step 3: Lifting arm to Z={self.arm_lift_z_pos_after_grasp}.")
        self.arm_pub.publish(Vector3(x=0.0, y=0.0, z=self.arm_lift_z_pos_after_grasp))

        self.get_logger().info(f"Delaying for {self.arm_lift_duration_sec}s for arm to lift...")
        time.sleep(self.arm_lift_duration_sec) # 同样注意阻塞问题

        self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Grasp sequence complete. Proceeding to TRANSPORT.")
        self._set_state(TaskState.TRANSPORT)


    # 移除旧的 grasp_initiate_open/close 和 arm_lift_after_grasp 状态处理函数
    # async def grasp_initiate_open_state_handler(self): ...
    # async def grasp_initiate_close_state_handler(self): ...
    # async def arm_lift_after_grasp_state_handler(self): ...


    def transport_state_handler(self):
        self.get_logger().debug("[TRANSPORT] Executing.")
        # ... (TRANSPORT 逻辑保持不变, 但移除 await)
        if self.current_pose is None:
            self.get_logger().warn("[TRANSPORT] Current pose not available."); self.cmd_vel_pub.publish(Twist()); return
        # ... (计算和发布逻辑) ...
        current_x = self.current_pose.position.x; current_y = self.current_pose.position.y
        # ... (与之前相同)
        dist_to_target = math.hypot(self.transport_target_pos_xyz[0] - current_x, self.transport_target_pos_xyz[1] - current_y)
        if dist_to_target < self.transport_target_reached_threshold_m:
            self.get_logger().info("[TRANSPORT] Reached target destination."); self.cmd_vel_pub.publish(Twist())
            self._set_state(TaskState.RELEASE); return # 不再 await
        # ... (发布速度指令)

    def release_state_handler(self):
        self.get_logger().info("[RELEASE] Commanding gripper to OPEN.")
        # 在释放后，我们认为任务完成
        self._send_gripper_goal(GripperControl.Goal.OPEN, next_step_callable=lambda: self._set_state(TaskState.DONE))


    def done_state_handler(self):
        self.get_logger().info("********* TASK COMPLETE *********")
        self.cmd_vel_pub.publish(Twist())
        if self.timer is not None and not self.timer.is_canceled():
             self.timer.cancel(); self.get_logger().info("Control cycle timer cancelled (TASK DONE).")

    def error_state_handler(self):
        self.get_logger().error("********* TASK IN ERROR STATE *********")
        self.cmd_vel_pub.publish(Twist())
        if self.timer is not None and not self.timer.is_canceled():
             self.timer.cancel(); self.get_logger().info("Control cycle timer cancelled (TASK ERROR).")


def main(args=None):
    rclpy.init(args=args)
    main_controller_node = MainController()
    try:
        rclpy.spin(main_controller_node)
    except KeyboardInterrupt: main_controller_node.get_logger().info("KeyboardInterrupt received, shutting down node.")
    except Exception as e: main_controller_node.get_logger().fatal(f"Unhandled exception in rclpy.spin(): {str(e)}\n{traceback.format_exc()}")
    finally:
        if rclpy.ok(): # Check if context is still valid
            if hasattr(main_controller_node, 'timer') and main_controller_node.timer is not None and not main_controller_node.timer.is_canceled():
                main_controller_node.timer.cancel()
                main_controller_node.get_logger().info("Timer cancelled in main finally block.")
            main_controller_node.get_logger().info("Destroying MainController node...")
            main_controller_node.destroy_node()
            if rclpy.ok(): # Check again before shutdown, as destroy_node might take time or context might change
                 rclpy.shutdown()
                 # Logger might not work after shutdown
                 print("rclpy.shutdown() called.") # Use print as logger might be gone
        else:
            print("rclpy context already invalidated during shutdown.") # Use print
        print("MainController node shutdown process complete.")

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from rclpy.duration import Duration
# from rclpy.logging import LoggingSeverity

# import numpy as np
# from geometry_msgs.msg import Twist, Vector3, PoseStamped, Transform
# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Range
# from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
# from robomaster_msgs.action import GripperControl # 确保这个action定义是正确的
# from action_msgs.msg import GoalStatus as ActionMsgGoalStatus

# from enum import Enum
# import math
# import traceback
# import time # 引入 time 模块

# class TaskState(Enum):
#     SCAN = 1
#     APPROACH = 2
#     ALIGN = 3
#     TRANSPORT = 5
#     RELEASE = 6
#     DONE = 7
#     ERROR = 8
#     PERFORM_GRASP_SEQUENCE = 12

# def quaternion_to_euler_yaw(q_x, q_y, q_z, q_w):
#     siny_cosp = 2 * (q_w * q_z + q_x * q_y)
#     cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
#     yaw = math.atan2(siny_cosp, cosy_cosp)
#     return yaw

# class MainController(Node):
#     def __init__(self):
#         super().__init__('main_controller_node')
#         self.get_logger().set_level(LoggingSeverity.DEBUG)

#         self.state = TaskState.SCAN
#         self.get_logger().info(f"Initializing MainController. Initial state: {self.state.name}")

#         self.latched_target_transform: Transform = None

#         self.tf_buffer_cache_time_sec = 20.0
#         self.tf_buffer = Buffer(cache_time=Duration(seconds=self.tf_buffer_cache_time_sec))
#         self.tf_listener = TransformListener(self.tf_buffer, self)
#         self.tf_lookup_timeout_sec = 0.05

#         # !!! 重要: 请根据 'ros2 action list' 和 'ros2 action info' 的结果修改这里的名称 !!!
#         self.gripper_action_server_name = '/gripper' # 或 '/RoboMaster/gripper' 或其他正确的名称
#         self.get_logger().info(f"Attempting to connect to gripper action server: {self.gripper_action_server_name}")
#         self.gripper_client = ActionClient(self, GripperControl, self.gripper_action_server_name)
#         self.gripper_power = 0.8
#         self.gripper_target_state_for_feedback = None
#         self.gripper_action_in_progress = False
#         self.next_step_after_gripper_success = None # 修改变量名以明确是成功后执行

#         self.current_pose = None
#         self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
#         self.tof_distance = None
#         self.tof_sub = self.create_subscription(Range, '/tof', self.tof_cb, 10)

#         # !!! 重要: 请根据CoppeliaSim中机械臂监听的话题名称修改 !!!
#         self.arm_command_topic = '/cmd_arm' # 或 '/RoboMaster/cmd_arm'
#         self.get_logger().info(f"Publishing arm commands to topic: {self.arm_command_topic}")
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.arm_pub = self.create_publisher(Vector3, self.arm_command_topic, 10)

#         # --- 控制参数 ---
#         self.scan_angular_speed = 0.5
#         self.scan_target_max_dist_m = 5.0
#         self.grasp_dist_m = 0.05
#         self.approach_align_dx_threshold_m = 0.05
#         self.approach_align_dy_threshold_m = 0.03
#         self.approach_far_linear_gain = 0.8
#         self.approach_far_angular_gain = 0.4
#         self.approach_far_max_linear_vel_ms = 0.3
#         self.approach_near_threshold_add_m = 0.05
#         self.approach_near_linear_gain = 0.3
#         self.approach_near_angular_gain = 0.5
#         self.approach_near_max_linear_vel_ms = 0.1
#         self.approach_max_angular_vel_rads = self.scan_angular_speed
#         self.approach_motion_deadband_dx_m = 0.015
#         self.approach_motion_deadband_dy_m = 0.015
#         self.align_arm_z_pos = 0.12
#         self.gripper_open_delay_sec = 1.5 # 稍增加延时，确保物理动作有时间
#         self.gripper_close_delay_sec = 1.5 # 增加关闭延时
#         self.arm_lift_z_pos_after_grasp = 0.18
#         self.arm_lift_duration_sec = 2.0 # 稍增加延时

#         self.transport_target_pos_xyz = [1.0, 1.0, 0.0]
#         self.transport_target_reached_threshold_m = 0.15
#         self.transport_turn_angle_threshold_rad = math.radians(10.0)
#         self.transport_creep_linear_vel_ms = 0.05
#         self.transport_linear_gain = 0.4
#         self.transport_angular_gain = 0.7
#         self.transport_max_linear_vel_ms = 0.3
#         self.transport_max_angular_vel_rads = 0.4
#         self.transport_fine_tune_angular_gain_factor = 0.5

#         self.control_cycle_period_sec = 0.1
#         self.timer = self.create_timer(self.control_cycle_period_sec, self.control_cycle)

#         self.get_logger().info("MainController initialized with parameters.")

#     def odom_cb(self, msg: Odometry):
#         self.current_pose = msg.pose.pose

#     def tof_cb(self, msg: Range):
#         self.tof_distance = msg.range

#     def _set_state(self, new_state: TaskState):
#         if self.state != new_state:
#             self.get_logger().info(f"State transition: {self.state.name} -> {new_state.name}")
#             self.state = new_state
#             # 当状态改变时，如果 gripper_action_in_progress 为 true 但我们切换到了一个不应等待夹爪的动作，
#             # 可能需要重置它，但这需要小心处理，避免中断正在进行的有效动作。
#             # 通常，状态转换应由动作完成或明确的逻辑触发。

#     def control_cycle(self):
#         if self.gripper_action_in_progress:
#             self.get_logger().debug(f"Gripper action in progress (target: {self.gripper_target_state_for_feedback}), skipping control cycle for state {self.state.name}")
#             return

#         self.get_logger().debug(f"Current state: {self.state.name}")
#         try:
#             handler_name = self.state.name.lower() + "_state_handler"
#             current_state_handler = getattr(self, handler_name, None)
#             if current_state_handler and callable(current_state_handler):
#                 current_state_handler()
#             else:
#                 self.get_logger().error(f"No valid handler for state: {self.state.name} (tried: {handler_name})")
#                 self._set_state(TaskState.ERROR)
#         except Exception as e:
#             self.get_logger().error(f"Error in control cycle for state {self.state.name}: {str(e)}\n{traceback.format_exc()}")
#             self._set_state(TaskState.ERROR)
#             self.cmd_vel_pub.publish(Twist())

#     def scan_state_handler(self):
#         self.get_logger().debug("[SCAN] Executing.")
#         if self.latched_target_transform is not None:
#             self.get_logger().info("[SCAN] Clearing previously latched target transform.")
#             self.latched_target_transform = None
#         twist = Twist()
#         twist.angular.z = self.scan_angular_speed
#         self.cmd_vel_pub.publish(twist)
#         try:
#             transform = self.tf_buffer.lookup_transform(
#                 'base_link', 'target_marker', rclpy.time.Time(),
#                 timeout=Duration(seconds=self.tf_lookup_timeout_sec))
#             trans_x = transform.transform.translation.x
#             self.get_logger().info(f"[SCAN] TF LOOKUP SUCCESS! Target at x: {trans_x:.3f} m.")
#             if 0 < trans_x < self.scan_target_max_dist_m:
#                 self.cmd_vel_pub.publish(Twist())
#                 self._set_state(TaskState.APPROACH)
#             else:
#                 self.get_logger().debug(f"[SCAN] Target at {trans_x:.3f}m. Condition not met. Continuing SCAN.")
#         except (LookupException, ConnectivityException, ExtrapolationException) as e:
#             self.get_logger().warn(f"[SCAN] TF lookup failed: {type(e).__name__} - {str(e)}. Continuing SCAN.")
#         except Exception as e:
#             self.get_logger().error(f"[SCAN] Unexpected error: {str(e)}\n{traceback.format_exc()}")
#             self._set_state(TaskState.ERROR)

#     def approach_state_handler(self):
#         self.get_logger().debug("[APPROACH] Executing.")
#         try:
#             current_tf_lookup = self.tf_buffer.lookup_transform(
#                 'base_link', 'target_marker', rclpy.time.Time(),
#                 timeout=Duration(seconds=self.tf_lookup_timeout_sec))
#             trans_x = current_tf_lookup.transform.translation.x
#             trans_y = current_tf_lookup.transform.translation.y
#             dx = trans_x - self.grasp_dist_m
#             dy = trans_y
            
#             twist = Twist()
#             if abs(trans_x) < (self.grasp_dist_m + self.approach_near_threshold_add_m):
#                 current_linear_gain = self.approach_near_linear_gain
#                 current_angular_gain = self.approach_near_angular_gain
#                 current_max_linear_vel = self.approach_near_max_linear_vel_ms
#             else:
#                 current_linear_gain = self.approach_far_linear_gain
#                 current_angular_gain = self.approach_far_angular_gain
#                 current_max_linear_vel = self.approach_far_max_linear_vel_ms
#             current_max_angular_vel = self.approach_max_angular_vel_rads
            
#             desired_linear_x = current_linear_gain * dx
#             desired_angular_z = -current_angular_gain * dy # Negative to correct based on dy
            
#             final_linear_x = np.clip(desired_linear_x, -current_max_linear_vel, current_max_linear_vel)
#             final_angular_z = np.clip(desired_angular_z, -current_max_angular_vel, current_max_angular_vel)
            
#             if abs(dx) < self.approach_motion_deadband_dx_m: final_linear_x = 0.0
#             if abs(dy) < self.approach_motion_deadband_dy_m: final_angular_z = 0.0
            
#             twist.linear.x = final_linear_x
#             twist.angular.z = final_angular_z

#             self.get_logger().info(f"[APPROACH] Target TF: x_base={trans_x:.3f}, y_base={trans_y:.3f}, Errors: dx={dx:.3f}, dy={dy:.3f}, CmdVel: lin_x={final_linear_x:.3f}, ang_z={final_angular_z:.3f}")

#             if abs(dx) < self.approach_align_dx_threshold_m and abs(dy) < self.approach_align_dy_threshold_m:
#                 self.get_logger().info(f"[APPROACH] Target ALIGNED. Latching target pose.")
#                 self.cmd_vel_pub.publish(Twist())
#                 try:
#                     self.latched_target_transform = current_tf_lookup.transform
#                     self._set_state(TaskState.ALIGN)
#                 except AttributeError:
#                     self.get_logger().error(f"[APPROACH] Failed to latch target pose. Returning to SCAN.")
#                     self.latched_target_transform = None
#                     self._set_state(TaskState.SCAN)
#                 return
#             self.cmd_vel_pub.publish(twist)
#         except (LookupException, ConnectivityException, ExtrapolationException) as e:
#             self.get_logger().warn(f"[APPROACH] TF lookup failed. Switching to SCAN.")
#             self.latched_target_transform = None
#             self._set_state(TaskState.SCAN)
#         except Exception as e:
#             self.get_logger().error(f"[APPROACH] Unexpected error: {str(e)}\n{traceback.format_exc()}")
#             self.latched_target_transform = None
#             self._set_state(TaskState.ERROR)

#     def align_state_handler(self):
#         if self.latched_target_transform is None:
#             self.get_logger().error("[ALIGN] No latched target pose. Returning to SCAN.")
#             self._set_state(TaskState.SCAN)
#             return

#         self.get_logger().info(f"[ALIGN] Lowering arm to Z={self.align_arm_z_pos}. Switching to PERFORM_GRASP_SEQUENCE.")
#         self.arm_pub.publish(Vector3(x=0.0, y=0.0, z=self.align_arm_z_pos))
#         # WARNING: time.sleep() blocks the callback. For longer delays, use a ROS timer.
#         time.sleep(0.5) # Short delay for command to be sent and possibly arm to start moving
#         self._set_state(TaskState.PERFORM_GRASP_SEQUENCE)

#     def _send_gripper_goal(self, target_state_value, next_step_on_success_callable=None):
#         if self.gripper_action_in_progress:
#             self.get_logger().warn(f"Gripper action already in progress (current target: {self.gripper_target_state_for_feedback}), new goal for {target_state_value} ignored.")
#             return

#         if not self.gripper_client.wait_for_server(timeout_sec=1.0): # Add a timeout
#             self.get_logger().error(f"Gripper action server '{self.gripper_action_server_name}' not available!")
#             self._set_state(TaskState.ERROR)
#             return

#         self.gripper_target_state_for_feedback = target_state_value
#         self.next_step_after_gripper_success = next_step_on_success_callable
#         self.gripper_action_in_progress = True

#         goal_msg = GripperControl.Goal()
#         # Ensure GripperControl.Goal.OPEN/CLOSE are the correct integer values if not enum members
#         # For Robomaster, typically OPEN=0, CLOSE=1 (or 1 and 0 depending on SDK/firmware)
#         # Let's assume GripperControl.Goal.OPEN and GripperControl.Goal.CLOSE are defined as expected
#         # (e.g. in the .action file or the generated Python code for the action)
#         if target_state_value == "OPEN_STATE": # Placeholder if direct enum/int not used
#             goal_msg.target_state = GripperControl.Goal.OPEN # Or the correct int value
#             log_target_str = "OPEN"
#         elif target_state_value == "CLOSE_STATE": # Placeholder
#             goal_msg.target_state = GripperControl.Goal.CLOSE # Or the correct int value
#             log_target_str = "CLOSE"
#         else: # Assuming target_state_value is already the correct enum/int
#             goal_msg.target_state = target_state_value
#             log_target_str = str(target_state_value)


#         goal_msg.power = self.gripper_power

#         self.get_logger().info(f"Sending gripper goal: {log_target_str} with power {goal_msg.power}")
#         send_goal_future = self.gripper_client.send_goal_async(
#             goal_msg,
#             feedback_callback=self._gripper_feedback_callback
#         )
#         send_goal_future.add_done_callback(self._gripper_goal_response_callback)

#     def _gripper_goal_response_callback(self, future):
#         goal_handle = None
#         try:
#             goal_handle = future.result()
#         except Exception as e:
#             self.get_logger().error(f'Exception while getting goal handle: {e}')
#             self.gripper_action_in_progress = False
#             self._set_state(TaskState.ERROR)
#             return

#         if not goal_handle or not goal_handle.accepted:
#             self.get_logger().error(f'Gripper command {"rejected" if goal_handle else "failed (no handle)"}.')
#             self.gripper_action_in_progress = False
#             self._set_state(TaskState.ALIGN) # Or ERROR
#             return

#         self.get_logger().info('Gripper command accepted.')
#         self._get_result_future = goal_handle.get_result_async()
#         self._get_result_future.add_done_callback(self._gripper_get_result_callback)

#     def _gripper_feedback_callback(self, feedback_msg):
#         feedback = feedback_msg.feedback
#         # !!! IMPORTANT: Verify 'current_state' exists in your GripperControl.action feedback definition !!!
#         if hasattr(feedback, 'current_state'):
#             self.get_logger().info(f'Gripper Feedback: Current state - {feedback.current_state} (Target was: {self.gripper_target_state_for_feedback})')
#             # Note: Original logic triggered next step on feedback match.
#             # This is often too early as feedback might report state before action is truly "done".
#             # We will primarily rely on the final result now.
#             # However, feedback can be useful for monitoring or more complex logic if needed.
#             if feedback.current_state == self.gripper_target_state_for_feedback:
#                  self.get_logger().debug(f"Feedback indicates gripper reached target state {self.gripper_target_state_for_feedback}.")
#         else:
#             self.get_logger().debug(f'Gripper Feedback received (but "current_state" field is missing or structure is unknown). Feedback: {feedback}')


#     def _gripper_get_result_callback(self, future):
#         result_wrapper = None
#         try:
#             result_wrapper = future.result()
#         except Exception as e:
#             self.get_logger().error(f'Exception while getting result wrapper: {e}')
#             self.gripper_action_in_progress = False
#             self._set_state(TaskState.ERROR)
#             return

#         # Regardless of success/failure, the action has concluded from the client's perspective of waiting for a result.
#         self.gripper_action_in_progress = False
#         self.get_logger().info(f"Gripper action concluded (result received). Resetting gripper_action_in_progress=False.")


#         if not result_wrapper:
#             self.get_logger().error('Gripper result future failed to return a wrapper.')
#             self._set_state(TaskState.ERROR)
#             return

#         status = result_wrapper.status
#         result = result_wrapper.result # GripperControl.Result

#         if status == ActionMsgGoalStatus.STATUS_SUCCEEDED:
#             log_msg = "Gripper action SUCCEEDED."
#             # !!! IMPORTANT: Verify 'success' or 'achieved_state' exists in your GripperControl.action result definition !!!
#             if hasattr(result, 'success') and result.success:
#                 log_msg += " Result field 'success'=true."
#             elif hasattr(result, 'achieved_state'):
#                  log_msg += f" Result field 'achieved_state'={result.achieved_state}."
#             else:
#                 log_msg += " (No specific success field in result or not checked)."
#             self.get_logger().info(log_msg)

#             if self.next_step_after_gripper_success and callable(self.next_step_after_gripper_success):
#                 self.get_logger().info("Executing next step after gripper result SUCCEEDED.")
#                 # Clear before calling to prevent re-entrancy issues if next step also uses gripper
#                 step_to_call = self.next_step_after_gripper_success
#                 self.next_step_after_gripper_success = None
#                 step_to_call()
#             else:
#                 self.get_logger().debug("No next_step_after_gripper_success defined or not callable after SUCCEEDED.")

#         else: # FAILED, ABORTED, CANCELED etc.
#             self.get_logger().error(f'Gripper action did NOT SUCCEED. Status: {status} (see action_msgs.msg.GoalStatus for details)')
#             if result:
#                  self.get_logger().error(f'Result content on failure: {result}')

#             # If action failed, do not proceed with next_step_after_gripper_success
#             self.next_step_after_gripper_success = None # Clear any pending next step
#             self.get_logger().warn("Gripper action did not succeed, returning to ALIGN state for potential retry.")
#             self._set_state(TaskState.ALIGN)


#     def perform_grasp_sequence_state_handler(self):
#         self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Initiating grasp sequence.")
#         self.get_logger().info("Step 1: Opening gripper.")
#         self._send_gripper_goal(GripperControl.Goal.OPEN, next_step_on_success_callable=self._initiate_close_gripper_after_open)

#     def _initiate_close_gripper_after_open(self):
#         self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Gripper OPEN action SUCCEEDED (or presumed done).")
#         self.get_logger().info(f"Delaying for {self.gripper_open_delay_sec}s before closing...")
#         time.sleep(self.gripper_open_delay_sec) # WARNING: Blocking call

#         self.get_logger().info("Step 2: Closing gripper.")
#         self._send_gripper_goal(GripperControl.Goal.CLOSE, next_step_on_success_callable=self._initiate_lift_arm_after_close)

#     def _initiate_lift_arm_after_close(self):
#         self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Gripper CLOSE action SUCCEEDED (or presumed done).")
#         self.get_logger().info(f"Step 3: Lifting arm to Z={self.arm_lift_z_pos_after_grasp}.")
#         self.arm_pub.publish(Vector3(x=0.0, y=0.0, z=self.arm_lift_z_pos_after_grasp))

#         self.get_logger().info(f"Delaying for {self.arm_lift_duration_sec}s for arm to lift...")
#         time.sleep(self.arm_lift_duration_sec) # WARNING: Blocking call

#         self.get_logger().info("[PERFORM_GRASP_SEQUENCE] Grasp sequence complete. Proceeding to TRANSPORT.")
#         self._set_state(TaskState.TRANSPORT)

#     def transport_state_handler(self):
#         self.get_logger().debug("[TRANSPORT] Executing.")
#         if self.current_pose is None:
#             self.get_logger().warn("[TRANSPORT] Current pose not available."); self.cmd_vel_pub.publish(Twist()); return
        
#         current_x = self.current_pose.position.x
#         current_y = self.current_pose.position.y
#         current_q = self.current_pose.orientation
#         current_yaw = quaternion_to_euler_yaw(current_q.x, current_q.y, current_q.z, current_q.w)
        
#         target_x = self.transport_target_pos_xyz[0]
#         target_y = self.transport_target_pos_xyz[1]
        
#         target_angle = math.atan2(target_y - current_y, target_x - current_x)
#         angle_error = target_angle - current_yaw
#         angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi # Normalize to -pi to pi
        
#         dist_to_target = math.hypot(target_x - current_x, target_y - current_y)
        
#         self.get_logger().info(f"[TRANSPORT] To:({target_x:.2f},{target_y:.2f}), From:({current_x:.2f},{current_y:.2f}), Dist:{dist_to_target:.2f}, Yaw:{math.degrees(current_yaw):.1f}deg, TargetAng:{math.degrees(target_angle):.1f}deg, AngErr:{math.degrees(angle_error):.1f}deg")
        
#         twist = Twist()
#         if dist_to_target < self.transport_target_reached_threshold_m:
#             self.get_logger().info("[TRANSPORT] Reached target destination.")
#             self.cmd_vel_pub.publish(Twist())
#             self._set_state(TaskState.RELEASE)
#             return
            
#         if abs(angle_error) > self.transport_turn_angle_threshold_rad:
#             twist.linear.x = self.transport_creep_linear_vel_ms # Creep forward while turning
#             twist.angular.z = np.clip(self.transport_angular_gain * angle_error, 
#                                       -self.transport_max_angular_vel_rads, 
#                                       self.transport_max_angular_vel_rads)
#         else:
#             twist.linear.x = np.clip(self.transport_linear_gain * dist_to_target, 
#                                      0.05, # Min speed to ensure movement
#                                      self.transport_max_linear_vel_ms)
#             # Fine tune angular speed when mostly aligned
#             fine_tune_angular_gain = self.transport_angular_gain * self.transport_fine_tune_angular_gain_factor
#             twist.angular.z = np.clip(fine_tune_angular_gain * angle_error, 
#                                       -self.transport_max_angular_vel_rads * 0.5, # Reduced max angular when aligned 
#                                       self.transport_max_angular_vel_rads * 0.5)
                                      
#         self.cmd_vel_pub.publish(twist)
#         self.get_logger().debug(f"[TRANSPORT] cmd_vel: lin_x={twist.linear.x:.3f}, ang_z={twist.angular.z:.3f}")


#     def release_state_handler(self):
#         self.get_logger().info("[RELEASE] Commanding gripper to OPEN.")
#         self._send_gripper_goal(GripperControl.Goal.OPEN, next_step_on_success_callable=lambda: self._set_state(TaskState.DONE))

#     def done_state_handler(self):
#         self.get_logger().info("********* TASK COMPLETE *********")
#         self.cmd_vel_pub.publish(Twist())
#         if self.timer is not None and not self.timer.is_canceled():
#              self.timer.cancel()
#              self.get_logger().info("Control cycle timer cancelled (TASK DONE).")

#     def error_state_handler(self):
#         self.get_logger().error("********* TASK IN ERROR STATE *********")
#         self.cmd_vel_pub.publish(Twist())
#         if self.timer is not None and not self.timer.is_canceled():
#              self.timer.cancel()
#              self.get_logger().info("Control cycle timer cancelled (TASK ERROR).")

# def main(args=None):
#     rclpy.init(args=args)
#     main_controller_node = MainController()
#     try:
#         rclpy.spin(main_controller_node)
#     except KeyboardInterrupt:
#         main_controller_node.get_logger().info("KeyboardInterrupt received, shutting down node.")
#     except Exception as e:
#         main_controller_node.get_logger().fatal(f"Unhandled exception in rclpy.spin(): {str(e)}\n{traceback.format_exc()}")
#     finally:
#         # Ensure cleanup happens
#         main_controller_node.get_logger().info("Initiating shutdown sequence...")
#         if hasattr(main_controller_node, 'timer') and main_controller_node.timer is not None and not main_controller_node.timer.is_canceled():
#             main_controller_node.timer.cancel()
#             main_controller_node.get_logger().info("Timer cancelled in main finally block.")
        
#         # It's good practice to destroy the node explicitly before shutdown if rclpy.ok()
#         if rclpy.ok():
#             main_controller_node.get_logger().info("Destroying MainController node...")
#             main_controller_node.destroy_node()
        
#         if rclpy.ok():
#             main_controller_node.get_logger().info("Calling rclpy.shutdown()...") # Logger might not work after this
#             rclpy.shutdown()
#             print("rclpy.shutdown() called.") # Use print as logger might be gone
#         else:
#             print("rclpy context already invalidated or shutdown called elsewhere.")
#         print("MainController node shutdown process complete.")

# if __name__ == '__main__':
#     main()