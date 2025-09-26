#!/usr/bin/env python3

# YDLIDAR /scan → 정면/좌/우 최소거리(Float32) 퍼블리시
#
# - TF:   laser_frame -> base_frame 변환 적용 후 base_link 기준 각/거리로 계산
# - 구독 토픽 이름: /scan (sensor_msgs/LaserScan)
# - 발행 토픽 이름:
#     /perception/front_distance : 정면 섹터의 최소거리 [m]
#     /perception/left_distance  : 좌측(+90° 주변) 섹터의 최소거리 [m]
#     /perception/right_distance : 우측(-90° 주변) 섹터의 최소거리 [m]

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener

def yaw_from_quat(x, y, z, w):
    """쿼터니언 → z-yaw (rad)"""
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

class FrontDistanceNode(Node):
    def __init__(self):
        super().__init__('front_distance_node')

        # ---------- 파라미터 ----------
        # 좌표/토픽
        self.base_frame = self.declare_parameter('base_frame', 'base_link') \
            .get_parameter_value().string_value
        self.scan_topic = self.declare_parameter('scan_topic', '/scan') \
            .get_parameter_value().string_value

        # 섹터 폭(도)
        #  - 정면: ±front_deg (예: 5도면 총 10도)
        #  - 좌/우: 각각 중심각이 +90°/-90°이고, 그 주변을 side_deg 폭으로 본다.
        self.front_deg  = float(self.declare_parameter('front_deg', 5.0) # 정면의 각도는 (+5 ~ -5로 총 10도)
                                .get_parameter_value().double_value)
        self.side_deg   = float(self.declare_parameter('side_deg', 10.0) # 좌/우 각도는 (+10 ~ -10로 총 20도)
                                .get_parameter_value().double_value)

        # 퍼블리시 주기 10Hz
        self.period_sec = float(self.declare_parameter('period_sec', 0.10)
                                .get_parameter_value().double_value)

        # 출력 토픽 이름
        self.out_front_topic = self.declare_parameter('front_distance_topic',
                                                      '/perception/front_distance') \
                                .get_parameter_value().string_value                     # 정면 거리 토픽
        self.out_left_topic  = self.declare_parameter('left_distance_topic',
                                                      '/perception/left_distance') \
                                .get_parameter_value().string_value                     # 좌측 거리 토픽
        self.out_right_topic = self.declare_parameter('right_distance_topic',
                                                      '/perception/right_distance') \
                                .get_parameter_value().string_value                     # 우측 거리 토픽    
        # --------------------------------

        # TF 준비
        self.buf = Buffer()
        self.lst = TransformListener(self.buf, self)

        self.last_scan = None
        self.laser_frame = None

        # 구독: /scan (QoS = BestEffort)
        self.sub = self.create_subscription(LaserScan, self.scan_topic,
                                            self.on_scan, qos_profile_sensor_data)

        # 퍼블리셔: 거리 3종
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub_front = self.create_publisher(Float32, self.out_front_topic, qos_out)
        self.pub_left  = self.create_publisher(Float32, self.out_left_topic,  qos_out)
        self.pub_right = self.create_publisher(Float32, self.out_right_topic, qos_out)

        # 타이머로 주기 퍼블리시
        self.timer = self.create_timer(self.period_sec, self.tick)

        self.get_logger().info(
            f"\n[get_distance_scan] scan_topic={self.scan_topic}\n"
            f"  base_frame={self.base_frame}\n"
            f"  sectors: front=±{self.front_deg}°, left=90°±{self.side_deg/2}°, right=-90°±{self.side_deg/2}°\n"
            f"  out: {self.out_front_topic}, {self.out_left_topic}, {self.out_right_topic}\n"
            f"  period={self.period_sec}s"
        )

    # ---------------- 콜백 ---------------- #
    def on_scan(self, msg: LaserScan):
        self.last_scan = msg
        if self.laser_frame is None:
            self.laser_frame = msg.header.frame_id
            self.get_logger().info(f"laser_frame: {self.laser_frame}")

    # ---------------- 메인 처리 ---------------- #
    def tick(self):
        """마지막 스캔과 TF를 이용해 정면/좌/우 최소거리 계산 후 퍼블리시"""
        if self.last_scan is None or self.laser_frame is None:
            return

        scan = self.last_scan

        # 1) 유효 범위 필터
        r = np.asarray(scan.ranges, dtype=float)
        ang = scan.angle_min + np.arange(r.size) * scan.angle_increment
        valid = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
        if not np.any(valid):
            return
        r = r[valid]
        ang = ang[valid]

        # 2) TF: laser_frame -> base_frame
        try:
            # (target_frame, source_frame, time)
            tf = self.buf.lookup_transform(self.base_frame, self.laser_frame, Time())
        except Exception:
            # TF 아직 준비 안됨: 다음 주기 재시도
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # 3) laser 좌표 → base_link 좌표 (2D 회전/병진)
        xl      = r * np.cos(ang)
        yl      = r * np.sin(ang)
        c, s    = math.cos(yaw), math.sin(yaw)
        xb      = c*xl - s*yl + tx
        yb      = s*xl + c*yl + ty

        # 4) base_link에서의 극좌표
        d       = np.hypot(xb, yb)       # 거리 (피타고라스 삼각형의 빗변 구하는 공식)
        phi     = np.arctan2(yb, xb)     # 각도(세타를 구함)  -π ~ +π

        # 5) 섹터 마스크 구성
        #    - 정면: |phi| <= front_fd
        #    - 좌측:  |phi - +pi/2| <= side_hd
        #    - 우측:  |phi - -pi/2| <= side_hd
        front_fd = math.radians(self.front_deg)            # front full half-angle
        side_hd  = math.radians(self.side_deg) * 0.5       # side half-angle
        pi_2     = math.pi * 0.5

        m_front = np.abs(phi) <= front_fd
        m_left  = np.abs(phi - (+pi_2)) <= side_hd
        m_right = np.abs(phi - (-pi_2)) <= side_hd

        # 6) 최소거리 계산 & 퍼블리시
        #    유효 포인트가 없으면 퍼블리시 생략(필요 시 +inf 등 원하는 정책으로 바꿔도 됨)
        if np.any(m_front):
            d_front = float(np.min(d[m_front]))
            self.pub_front.publish(Float32(data=d_front))
        # else: pass

        if np.any(m_left):
            d_left = float(np.min(d[m_left]))
            self.pub_left.publish(Float32(data=d_left))

        if np.any(m_right):
            d_right = float(np.min(d[m_right]))
            self.pub_right.publish(Float32(data=d_right))

        # # 디버그 출력 원하면 주석 해제
        # if np.any(m_front) or np.any(m_left) or np.any(m_right):
        #     self.get_logger().info(
        #         f"front:{d_front if np.any(m_front) else None:.2f}  "
        #         f"left:{d_left if np.any(m_left) else None:.2f}  "
        #         f"right:{d_right if np.any(m_right) else None:.2f}"
        #     )

def main():
    rclpy.init()
    node = FrontDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
