#!/usr/bin/env python3
# get_distance_scan.py
# YDLIDAR /scan → 정면 거리만 발행

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
    # z-yaw만 필요
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

class FrontDistanceNode(Node):
    def __init__(self):
        super().__init__('front_distance_node') # 최소 거리를 발행할 토픽 이름

        # ---------- 파라미터 ----------
        self.base_frame   = self.declare_parameter('base_frame', 'base_link').get_parameter_value().string_value
        self.scan_topic   = self.declare_parameter('scan_topic', '/scan').get_parameter_value().string_value        # 라이다 스캔을 토픽
        self.front_deg    = float(self.declare_parameter('front_deg', 10.0).get_parameter_value().double_value)     # 정면 반각(-10~+10 -> 20도)
        self.period_sec   = float(self.declare_parameter('period_sec', 0.10).get_parameter_value().double_value)    # 퍼블리시 주기
        self.out_topic    = self.declare_parameter('front_distance_topic', '/perception/front_distance').get_parameter_value().string_value
        # --------------------------------

        # TF 준비
        self.buf = Buffer()
        self.lst = TransformListener(self.buf, self)

        self.last_scan = None
        self.laser_frame = None

        # 구독: 라이다 스캔
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)

        # 퍼블리셔: 정면 거리 (Reliable 권장)
        qos_out = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.pub_front = self.create_publisher(Float32, self.out_topic, qos_out)

        # 타이머로 주기 퍼블리시
        self.timer = self.create_timer(self.period_sec, self.tick)

        self.get_logger().info(
            f"\n[FrontDistanceNode] scan_topic={self.scan_topic}, out_topic={self.out_topic}, \n"
            f"base_frame={self.base_frame}, front=±{self.front_deg}°, period={self.period_sec}s"
        )

    def on_scan(self, msg: LaserScan):
        self.last_scan = msg
        if self.laser_frame is None:
            self.laser_frame = msg.header.frame_id
            self.get_logger().info(f"laser_frame: {self.laser_frame}")

    def tick(self):
        if self.last_scan is None or self.laser_frame is None:
            return

        msg = self.last_scan

        # 유효 범위 필터
        r = np.asarray(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(r.size) * msg.angle_increment
        valid = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        if not np.any(valid):
            return
        r = r[valid]
        ang = ang[valid]

        # TF: laser_frame -> base_frame
        try:
            tf = self.buf.lookup_transform(self.base_frame, self.laser_frame, Time()) # (target_frame, source_frame)
        except Exception:
            # TF가 아직 준비 안됐으면 다음 틱에서 재시도
            return

        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # 레이저좌표 → base_link 좌표 (2D)
        xl = r * np.cos(ang)        # 라이다 좌표의 점
        yl = r * np.sin(ang)
        c, s = math.cos(yaw), math.sin(yaw)
        xb = c*xl - s*yl + tx       # base_link 좌표의 점
        yb = s*xl + c*yl + ty

        # 직교 좌표 -> 극좌표로 변환
        d   = np.hypot(xb, yb)      # 극좌표로 변환했을 떄의 거리(직사각형 빗변의 거리를 구하는 공식)
        phi = np.arctan2(yb, xb)    # base_link 기준 각도

        # 정면 섹터 선택
        fd = math.radians(self.front_deg)
        m_front = np.abs(phi) <= fd

        if not np.any(m_front):
            return

        d_front = float(np.min(d[m_front]))

        # 퍼블리시
        msg_out = Float32()
        msg_out.data = d_front
        self.pub_front.publish(msg_out)

        # 디버그 로그 (원하면 주석 해제)
        # self.get_logger().info(f"front(±{self.front_deg:.0f}°) = {d_front:.2f} m")

def main():
    rclpy.init()
    node = FrontDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
