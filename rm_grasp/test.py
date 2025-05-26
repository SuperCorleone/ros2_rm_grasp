import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

# 假设 action 文件 GripperControl.action 定义在 robomaster_msgs 包中
# 如果没有明确的常量如 GripperControl.Goal.OPEN，您可能需要直接使用整数值
# e.g., OPEN=0, CLOSE=1, PAUSE=2 (需要与CoppeliaSim中服务器的实现对应)
from robomaster_msgs.action import GripperControl # 确保这个导入是正确的

class GripperControlClient(Node):
    """
    ROS 2 动作客户端，用于测试 Robomaster EP 夹爪。
    """
    # 假设的状态映射 (这些值需要与 action 定义或服务器实现一致)
    # 如果 GripperControl.Goal.OPEN 等常量存在，使用它们会更好
    # 例如: STATE_MAP = {
    #    "OPEN": GripperControl.Goal.OPEN,
    #    "CLOSE": GripperControl.Goal.CLOSE,
    #    "PAUSE": GripperControl.Goal.PAUSE
    # }
    # 如果没有，则使用预期的整数值:
    STATE_MAP = {
        "OPEN": 0,  # 示例值, 请核实
        "CLOSE": 1, # 示例值, 请核实
        "PAUSE": 2  # 示例值, 请核实
    }
    # 默认力度值
    DEFAULT_POWER = 0.5

    def __init__(self, action_server_name='gripper'):
        super().__init__('gripper_test_client')
        self._action_client = ActionClient(self, GripperControl, action_server_name)
        self.get_logger().info(f"GripperControlClient node initialized for action server: '{action_server_name}'")
        self.goal_done_ = False

    def send_goal(self, command_str, power=None):
        """
        发送夹爪控制目标。
        command_str: "OPEN", "CLOSE", or "PAUSE"
        power: (可选) 夹爪力度，浮点数
        """
        self.goal_done_ = False
        command_str_upper = command_str.upper()

        if command_str_upper not in self.STATE_MAP:
            self.get_logger().error(f"Invalid command: {command_str}. Valid commands are OPEN, CLOSE, PAUSE.")
            return False

        goal_msg = GripperControl.Goal()
        goal_msg.target_state = self.STATE_MAP[command_str_upper]
        goal_msg.power = float(power) if power is not None else self.DEFAULT_POWER

        self.get_logger().info(f"Waiting for action server '{self._action_client._action_name}'...")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server '{self._action_client._action_name}' not available after waiting.")
            return False

        self.get_logger().info(f"Sending goal: {command_str_upper} with power {goal_msg.power}")

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def goal_response_callback(self, future):
        """处理接受或拒绝目标的响应。"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.goal_done_ = True
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """处理动作完成后的结果。"""
        result = future.result().result
        # 假设 GripperControl_Result 有一个布尔字段 'success'
        # 您需要根据实际的 .action 文件调整
        if hasattr(result, 'success'):
             self.get_logger().info(f'Result: {{success: {result.success}}}')
        else:
             self.get_logger().info('Result received (no specific fields to display or fields unknown).')
        self.goal_done_ = True

    def feedback_callback(self, feedback_msg):
        """处理来自服务器的反馈。"""
        feedback = feedback_msg.feedback
        # 假设 GripperControl_Feedback 有一个 'current_state' 或 'progress' 字段
        # 您需要根据实际的 .action 文件调整
        if hasattr(feedback, 'status_string'): # 这是一个假设的反馈字段
            self.get_logger().info(f"Received feedback: {{status: {feedback.status_string}}}")
        elif hasattr(feedback, 'progress'): # 另一个假设的反馈字段
            self.get_logger().info(f"Received feedback: {{progress: {feedback.progress}}}")
        else:
            self.get_logger().info("Received feedback (no specific fields to display or fields unknown).")

    def is_goal_done(self):
        return self.goal_done_

def main(args=None):
    rclpy.init(args=args)
    action_client_node = GripperControlClient(action_server_name='gripper') # 或者您的 action server 名称

    print(f"RoboMaster EP Gripper Test Client")
    print(f"Commands will be sent to action server: '{action_client_node._action_client._action_name}'")
    print("Enter gripper command (OPEN, CLOSE, PAUSE), optionally followed by power (e.g., OPEN 0.7).")
    print("Or type EXIT to quit.")

    try:
        while rclpy.ok():
            user_input = input("> ").strip()
            if not user_input:
                continue

            parts = user_input.split()
            command = parts[0].upper()
            power_val = None
            if len(parts) > 1:
                try:
                    power_val = float(parts[1])
                except ValueError:
                    print("Invalid power value. It must be a number.")
                    continue

            if command == "EXIT":
                break

            if command in GripperControlClient.STATE_MAP:
                action_client_node.send_goal(command, power=power_val)
                # 等待动作完成，同时允许rclpy处理事件
                while not action_client_node.is_goal_done():
                    rclpy.spin_once(action_client_node, timeout_sec=0.1)
            else:
                print(f"Unknown command: {command}. Valid commands: OPEN, CLOSE, PAUSE, EXIT.")

    except KeyboardInterrupt:
        pass
    finally:
        action_client_node.get_logger().info('Shutting down GripperControlClient node.')
        action_client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()