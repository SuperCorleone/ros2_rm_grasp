import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import traceback
from typing import List, Tuple # 从你的代码中引入

# TF 和 Pose
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


# --- 你提供的自定义数据结构和类 ---
class ArucoResult:
    def __init__(self, corners: np.ndarray, index: int = -1):
        self.corners: np.ndarray = corners
        self.index: int = index

class ArucoDict:
    def __init__(self):
        self.sigs: List[List[int]] = []
        self.worldLoc: List[np.ndarray] = [] # 这个worldLoc可能需要调整或不直接使用

class ArucoDetector_Python: # 重命名以避免与ROS节点类冲突
    def __init__(self, marker_image: np.ndarray, bits: int):
        if marker_image is None:
            raise ValueError("标记图像 'marker_image' 不能为空")
        if len(marker_image.shape) == 3 and marker_image.shape[2] == 3:
            # self.get_logger().info("构造函数：提供的标记图像是彩色的，将转换为灰度图。") # ROS节点中用self.get_logger()
            print("ArucoDetector_Python: 提供的标记图像是彩色的，将转换为灰度图。")
            marker_image = cv2.cvtColor(marker_image, cv2.COLOR_BGR2GRAY)
        elif len(marker_image.shape) == 3 and marker_image.shape[2] == 4:
            print("ArucoDetector_Python: 提供的标记图像是BGRA，将转换为灰度图。")
            marker_image = cv2.cvtColor(marker_image, cv2.COLOR_BGRA2GRAY)
        
        self.m_dict: ArucoDict = self._load_marker_dictionary(marker_image, bits)
        self.bits: int = bits
        self.pixel_len: int = int(np.sqrt(bits))
        if self.pixel_len * self.pixel_len != bits:
            raise ValueError(f"比特数 {bits} 必须是平方数 (例如 16, 25, 36)。")
        self.term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    def _get_contours_bits(self, image: np.ndarray, cnt: np.ndarray, bits_param: int) -> List[int]:
        pixel_len_local = int(np.sqrt(bits_param))
        if pixel_len_local * pixel_len_local != bits_param:
             raise ValueError(f"_get_contours_bits: 比特数 {bits_param} 必须是平方数。")
        dst_corners = np.array([[0,0],[bits_param-1,0],[bits_param-1,bits_param-1],[0,bits_param-1]], dtype=np.float32)
        if not isinstance(cnt, np.ndarray) or cnt.shape != (4,2) or cnt.dtype != np.float32:
            if isinstance(cnt, np.ndarray) and cnt.shape == (4,1,2):
                cnt = cnt.reshape((4,2)).astype(np.float32)
            else: raise ValueError("角点 'cnt' 必须是 4x2 float32 Numpy 数组。")
        M = cv2.getPerspectiveTransform(cnt, dst_corners)
        warped = cv2.warpPerspective(image, M, (bits_param, bits_param))
        _, binary_warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        binary_warped = cv2.erode(binary_warped, element)
        res_bits = []
        for r in range(pixel_len_local):
            for c in range(pixel_len_local):
                y = int(r * pixel_len_local + pixel_len_local / 2)
                x = int(c * pixel_len_local + pixel_len_local / 2)
                if y < bits_param and x < bits_param:
                    res_bits.append(1 if binary_warped[y,x] >= 128 else 0)
                else:
                    # print(f"警告：采样点 ({x},{y}) 超出warped图像边界 ({bits_param},{bits_param})") # Use logger in ROS node
                    res_bits.append(0)
        return res_bits

    def _equal_sig(self, sig1: List[int], sig2: List[int], allowed_misses: int = 0) -> bool:
        if len(sig1) != len(sig2): return False
        misses = sum(1 for i in range(len(sig1)) if sig1[i] != sig2[i])
        return misses <= allowed_misses

    def _order_contour_clockwise_top_left(self, cnt: np.ndarray) -> np.ndarray:
        points = cnt.reshape((4, 2)).astype(np.float32)
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

    def _find_squares(self, img: np.ndarray) -> List[np.ndarray]:
        cand_squares = []
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
        thresh = ~thresh
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) >= 200 and cv2.isContourConvex(approx): # Min area 200
                approx_float = approx.reshape(-1, 2).astype(np.float32)
                cv2.cornerSubPix(img, approx_float, (5,5), (-1,-1), self.term_crit)
                ordered_approx = self._order_contour_clockwise_top_left(approx_float)
                cand_squares.append(ordered_approx)
        return cand_squares

    def _load_marker_dictionary(self, marker_img: np.ndarray, bits_param: int) -> ArucoDict:
        loaded_dict = ArucoDict()
        h, w = marker_img.shape[:2]
        cnt_marker = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
        # These world points are specific to the custom dict and may need scaling/centering for solvePnP
        # The size (25x25) needs to relate to your self.marker_size_meters
        world_points_orig = np.array([[0,0,0],[25,0,0],[25,25,0],[0,25,0]], dtype=np.float32)
        current_marker_img = marker_img.copy()
        current_world_points = world_points_orig.copy()
        for i in range(4):
            sig = self._get_contours_bits(current_marker_img, cnt_marker, bits_param)
            loaded_dict.sigs.append(sig)
            loaded_dict.worldLoc.append(current_world_points.copy())
            if i < 3:
                current_marker_img = cv2.rotate(current_marker_img, cv2.ROTATE_90_CLOCKWISE)
                current_world_points = np.roll(current_world_points, shift=1, axis=0)
        return loaded_dict

    def detect_arucos(self, frame: np.ndarray, allowed_misses: int = 0) -> List[ArucoResult]:
        if frame is None: return []
        if len(frame.shape) == 3: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results: List[ArucoResult] = []
        candidate_squares = self._find_squares(frame)
        max_detections = 3 # As per C++
        for cnt in candidate_squares:
            if len(results) >= max_detections: break
            sig = self._get_contours_bits(frame, cnt, self.bits)
            for j in range(len(self.m_dict.sigs)):
                if self._equal_sig(sig, self.m_dict.sigs[j], allowed_misses):
                    results.append(ArucoResult(corners=cnt, index=j))
                    break 
        return results

# --- END: 你提供的自定义类 ---
# ▼▼▼ ROS 2 节点 ArucoDetector ▼▼▼
class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector_node') # 节点名与日志一致
        self.get_logger().set_level(LoggingSeverity.DEBUG)
        self.get_logger().info('Custom ArucoDetector (ROS Node) initializing...')

        # --- 常规参数 ---
        self.target_id_for_template = 0 # 用于生成模板和日志中概念上的目标ID
        self.marker_size_meters = 0.05  # ArUco标记的物理边长 (米)
        
        # --- 相机参数 (用于计算内参矩阵) ---
        self.camera_image_width = 640.0
        self.camera_image_height = 360.0
        self.camera_fov_y_degrees = 60.00 # 你更新后的垂直视场角 (度)

        # --- 自定义ArUco检测器相关参数 ---
        self.custom_detector_bits = 16       # ArUco码内部数据比特数 (例如 4x4 -> 16)
        self.template_marker_generation_size_pixels = 60 # 生成字典模板图的边长 (像素)
        self.template_marker_dictionary_type = cv2.aruco.DICT_4X4_50 # 生成模板用的字典
        
        # --- image_callback 中调用 custom_detector.detect_arucos 时的参数 ---
        self.detection_allowed_bit_errors = 1 # 允许的比特匹配误差

        # --- ROS Topic 和 Frame ID 参数 ---
        self.image_topic_name = '/camera/image_color' # 订阅的图像话题
        self.pose_topic_name = '/target_pose'       # 发布的PoseStamped话题
        self.tf_parent_frame_id = 'base_link'       # 发布的TF的父框架
        self.tf_child_frame_id = 'target_marker'    # 发布的TF的子框架 (目标)


        # --- 初始化 CvBridge ---
        self.bridge = CvBridge()
        self.get_logger().info("CvBridge initialized.")

        # --- 计算并设置相机内参和畸变系数 ---
        W = self.camera_image_width
        H = self.camera_image_height
        cx = W / 2.0
        cy = H / 2.0
        fov_y_radians = np.deg2rad(self.camera_fov_y_degrees)
        fy = (H / 2.0) / np.tan(fov_y_radians / 2.0)
        fx = fy  # 假设方形像素
        self.camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1.0]], dtype=np.float32)
        self.get_logger().info(f"Camera matrix: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        self.dist_coeffs = np.zeros(5, dtype=np.float32)
        self.get_logger().info(f"Distortion coeffs set to zeros.")

        # --- 初始化自定义ArUco检测器 (ArucoDetector_Python) ---
        self.get_logger().info(f"Generating template marker for custom dictionary: Type {self.template_marker_dictionary_type}, ID {self.target_id_for_template}, size {self.template_marker_generation_size_pixels}px")
        
        official_aruco_dict_for_template = cv2.aruco.getPredefinedDictionary(self.template_marker_dictionary_type)
        template_marker_img = np.zeros((self.template_marker_generation_size_pixels, self.template_marker_generation_size_pixels), dtype=np.uint8)
        
        # 使用 OpenCV 生成标准的 ArUco 标记图像作为模板
        if hasattr(cv2.aruco, 'generateImageMarker'): # OpenCV 4.7+
             template_marker_img = cv2.aruco.generateImageMarker(
                official_aruco_dict_for_template, self.target_id_for_template, 
                self.template_marker_generation_size_pixels, template_marker_img, 1)
        elif hasattr(cv2.aruco, 'drawMarker'): # OpenCV 4.6.0 (你的版本)
             cv2.aruco.drawMarker(official_aruco_dict_for_template, self.target_id_for_template, 
                                  self.template_marker_generation_size_pixels, template_marker_img, 1)
        else:
            self.get_logger().error("Neither generateImageMarker nor drawMarker found in cv2.aruco. Cannot generate template.")
            template_marker_img = None

        if template_marker_img is None:
            self.get_logger().fatal("Failed to generate template marker image for custom dictionary. Custom detector cannot be initialized.")
            self.custom_detector = None
        else:
            self.get_logger().info("Template marker image generated successfully for custom dictionary.")
            try:
                # 将 ArucoDetector_Python 类定义放在此文件上方或导入它
                self.custom_detector = ArucoDetector_Python(template_marker_img, self.custom_detector_bits)
                self.get_logger().info(f"Custom ArucoDetector_Python initialized with {len(self.custom_detector.m_dict.sigs)} signatures in its dictionary.")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize ArucoDetector_Python: {e}\n{traceback.format_exc()}")
                self.custom_detector = None
        
        # --- ROS2 接口 ---
        self.target_pose_pub = self.create_publisher(PoseStamped, self.pose_topic_name, 10)
        self.get_logger().info(f"Publisher for PoseStamped '{self.pose_topic_name}' created.")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("TransformBroadcaster initialized.")
        self.camera_sub = self.create_subscription(
            Image, self.image_topic_name, self.image_callback, 10)
        self.get_logger().info(f"Subscribing to image topic: '{self.image_topic_name}'")
        
        self.get_logger().info(f"Target conceptually is ArUco ID: {self.target_id_for_template}")
        self.get_logger().info(f"Physical marker size for pose estimation: {self.marker_size_meters} meters.")
        self.get_logger().info("--- Custom ArucoDetector (ROS Node) initialization complete ---")

    def image_callback(self, msg: Image):
        # (你的 image_callback 代码，使用 self.detection_allowed_bit_errors)
        # 例如:
        # detected_results: List[ArucoResult] = self.custom_detector.detect_arucos(gray_image, self.detection_allowed_bit_errors)

        # --- 为了演示，我将粘贴你提供的 image_callback 并修改 allowed_bit_errors ---
        # self.get_logger().debug(f'image_callback: Entered. Timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        if self.custom_detector is None:
            self.get_logger().error("Custom detector not initialized. Skipping detection.")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}') # 修正日志输出变量名
            return
        except Exception as e:
            self.get_logger().error(f'Exception in imgmsg_to_cv2: {e}\n{traceback.format_exc()}')
            return

        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 使用参数化的 allowed_bit_errors
        detected_results: List[ArucoResult] = self.custom_detector.detect_arucos(gray_image, self.detection_allowed_bit_errors)

        found_target_in_this_frame = False
        if detected_results:
            res = detected_results[0] 
            found_target_in_this_frame = True
            marker_half_size = self.marker_size_meters / 2.0
            object_points_3d = np.array([
                [-marker_half_size,  marker_half_size, 0.0],
                [ marker_half_size,  marker_half_size, 0.0],
                [ marker_half_size, -marker_half_size, 0.0],
                [-marker_half_size, -marker_half_size, 0.0]
            ], dtype=np.float32)
            image_points_2d = res.corners
            
            rvec, tvec = None, None
            try:
                success, rvec, tvec = cv2.solvePnP(object_points_3d, image_points_2d,
                                                   self.camera_matrix, self.dist_coeffs)
                if not success:
                    self.get_logger().warn("solvePnP failed.")
                    return
            except Exception as e:
                self.get_logger().error(f"Exception during solvePnP: {e}\n{traceback.format_exc()}")
                return

            if rvec is not None and tvec is not None:
                target_pose_msg = PoseStamped()
                target_pose_msg.header = msg.header
                target_pose_msg.header.stamp = self.get_clock().now().to_msg()
                target_pose_msg.pose.position.x = tvec[0][0]
                target_pose_msg.pose.position.y = tvec[1][0]
                target_pose_msg.pose.position.z = tvec[2][0]
                try:
                    angle = np.linalg.norm(rvec)
                    if angle > 1e-6:
                        axis = rvec.flatten() / angle
                        target_pose_msg.pose.orientation.x = axis[0] * np.sin(angle / 2.0)
                        target_pose_msg.pose.orientation.y = axis[1] * np.sin(angle / 2.0)
                        target_pose_msg.pose.orientation.z = axis[2] * np.sin(angle / 2.0)
                        target_pose_msg.pose.orientation.w = np.cos(angle / 2.0)
                    else: target_pose_msg.pose.orientation.w = 1.0
                    self.target_pose_pub.publish(target_pose_msg)
                except Exception as e:
                    self.get_logger().error(f"Error calculating/publishing PoseStamped: {e}\n{traceback.format_exc()}")
                    # 备份：如果旋转计算失败，发布单位四元数
                    target_pose_msg.pose.orientation.w = 1.0
                    target_pose_msg.pose.orientation.x = 0.0
                    target_pose_msg.pose.orientation.y = 0.0
                    target_pose_msg.pose.orientation.z = 0.0


                try:
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = self.tf_parent_frame_id # 使用参数
                    t.child_frame_id = self.tf_child_frame_id  # 使用参数
                    t.transform.translation.x = tvec[2][0] 
                    t.transform.translation.y = -tvec[0][0] 
                    t.transform.translation.z = tvec[1][0]  
                    t.transform.rotation = target_pose_msg.pose.orientation 
                    self.tf_broadcaster.sendTransform(t)
                except Exception as e:
                    self.get_logger().error(f"Exception during TF broadcast: {e}\n{traceback.format_exc()}")
        
        current_time_ns = self.get_clock().now().nanoseconds
        if found_target_in_this_frame:
            if current_time_ns % (1 * 10**9) < (100 * 10**6): 
                self.get_logger().info(f"STATUS (Custom Detector): Target (ID {self.target_id_for_template}) is being tracked.")
        else:
            if current_time_ns % (3 * 10**9) < (100 * 10**6): 
                self.get_logger().info(f"STATUS (Custom Detector): Target (ID {self.target_id_for_template}) NOT detected.")

# --- main 函数 ---
def main(args=None):
    rclpy.init(args=args)
    aruco_detector_node = ArucoDetector() # 使用新的类名
    try:
        rclpy.spin(aruco_detector_node)
    except KeyboardInterrupt:
        aruco_detector_node.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e: 
        aruco_detector_node.get_logger().error(f'Unhandled exception in spin: {str(e)}\n{traceback.format_exc()}')
    finally:
        if rclpy.ok():
            aruco_detector_node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()