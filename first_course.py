#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
first_course.py (no-hard-stop + turn-arm-delay)
- /object_detection/sos(Empty) 수신 → 주행 시작
- CRUISE(직진): yaw 정렬 유지, (turn_arm_delay_sec 경과 전에는) 벽 감지해도 회전 금지
- turn_arm_delay_sec(기본 5s) 경과 후: 전방 거리 임계 이하면 TURNING_RIGHT 진입
- TURNING_RIGHT: IMU yaw 기준 -90° 우회전(선속도 0, 각속도 제어)
- POST_TURN_ADVANCE: 회전 완료 후 3초 직진
- 그 다음 **정지하지 않고** 다시 CRUISE 복귀(경기 중 정지 금지 규칙 반영)
"""

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32, String


# ================== 사용자가 맨 위에서 바꾸기 쉬운 기본값 ================== #
CRUISE_SPEED = 0.25         # [m/s] 직진 기본 속도
TURN_ARM_DELAY_SEC = 5.0    # [s] SOS 후 이 시간이 지나기 전에는 회전 트리거 비활성
POST_ADVANCE_SEC = 3.0      # [s] 회전 완료 후 전진 유지 시간
FRONT_THRESHOLD_M = 1.5     # [m] 전방 임계(회전 트리거)
RIGHT_TURN_DEG = 90.0       # [deg] 우회전 각도(양수로 표기; 내부에서 -부호로 사용)
# ========================================================================== #


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def wrap_pi(a):
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def ang_err(target, current):
    return wrap_pi(target - current)

def quat_to_yaw(qx, qy, qz, qw):
    s1 = 2.0 * (qw * qz + qx * qy)
    s2 = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s1, s2)


class Phase(Enum):
    IDLE = 0
    CRUISE = 1
    TURNING_RIGHT = 2
    POST_TURN_ADVANCE = 3
    # DONE 상태 제거(경기 중 정지 금지)


class FirstCourseController(Node):
    def __init__(self):
        super().__init__('first_course')

        # ---------------- 파라미터 선언(기본값은 위 상수 사용) ---------------- #
        self.declare_parameter('topic_sos', '/object_detection/sos')
        self.declare_parameter('topic_imu', '/imu/data')
        self.declare_parameter('topic_front_dist', '/perception/front_distance')
        self.declare_parameter('topic_cmd_vel', '/cmd_vel')

        self.declare_parameter('cruise_speed', CRUISE_SPEED)
        self.declare_parameter('yaw_kp', 1.2)                 # 비례이득(Kp): 각도 오차 × Kp = 회전속도
        self.declare_parameter('yaw_deadband_deg', 2.0)       # 작은 오차 무시(deg)
        self.declare_parameter('yaw_max_rate', 1.2)           # 직진 중 최대 회전속도 제한(rad/s)

        self.declare_parameter('front_threshold_m', FRONT_THRESHOLD_M)
        self.declare_parameter('front_hysteresis_m', 0.1)     # 채터 방지
        self.declare_parameter('consecutive_hits', 2)         # 임계 이하 연속 감지 횟수

        self.declare_parameter('turn_kp', 2.0)                # 회전 중 Kp
        self.declare_parameter('turn_max_rate', 1.5)          # 회전 중 최대 회전속도(rad/s)
        self.declare_parameter('turn_tol_deg', 2.0)           # 회전 완료 판정 오차(deg)
        self.declare_parameter('right_turn_deg', RIGHT_TURN_DEG)

        self.declare_parameter('post_advance_sec', POST_ADVANCE_SEC)
        self.declare_parameter('control_rate_hz', 20.0)

        self.declare_parameter('invert_yaw_sign', False)

        # *** 새로 추가: 회전 트리거 지연(arming delay) ***
        self.declare_parameter('turn_arm_delay_sec', TURN_ARM_DELAY_SEC)

        # ---------------- 파라미터 로드 ---------------- #
        p = self.get_parameter
        self.topic_sos = p('topic_sos').get_parameter_value().string_value
        self.topic_imu = p('topic_imu').get_parameter_value().string_value
        self.topic_front_dist = p('topic_front_dist').get_parameter_value().string_value
        self.topic_cmd_vel = p('topic_cmd_vel').get_parameter_value().string_value

        self.cruise_speed = float(p('cruise_speed').value)
        self.yaw_kp = float(p('yaw_kp').value)
        self.yaw_deadband = math.radians(float(p('yaw_deadband_deg').value))
        self.yaw_max_rate = float(p('yaw_max_rate').value)

        self.front_threshold = float(p('front_threshold_m').value)
        self.front_hyst = float(p('front_hysteresis_m').value)
        self.consecutive_hits_req = int(p('consecutive_hits').value)

        self.turn_kp = float(p('turn_kp').value)
        self.turn_max_rate = float(p('turn_max_rate').value)
        self.turn_tol = math.radians(float(p('turn_tol_deg').value))
        self.right_turn_rad = -math.radians(float(p('right_turn_deg').value))  # 우회전은 -부호

        self.post_advance_sec = float(p('post_advance_sec').value)
        self.dt = 1.0 / float(p('control_rate_hz').value)

        self.invert_yaw_sign = bool(p('invert_yaw_sign').value)
        self.turn_arm_delay_sec = float(p('turn_arm_delay_sec').value)

        # ---------------- 토픽/타이머 ---------------- #
        self.state_pub = self.create_publisher(String, '/first_course/state', 10)
        self.cmd_pub = self.create_publisher(Twist, self.topic_cmd_vel, 10)

        self.sub_sos = self.create_subscription(Empty, self.topic_sos, self.cb_sos, 10)
        self.sub_imu = self.create_subscription(Imu, self.topic_imu, self.cb_imu, qos_profile_sensor_data)
        self.sub_front = self.create_subscription(Float32, self.topic_front_dist, self.cb_front_dist, 10)

        self.timer = self.create_timer(self.dt, self.on_timer)

        # ---------------- 내부 상태 ---------------- #
        self.phase = Phase.IDLE
        self.yaw_now = None          # 현재 yaw
        self.yaw_ref = None          # 직진 기준 yaw
        self.target_yaw = None       # 회전 목표 yaw
        self.front_dist = None       # 전방 거리
        self.hit_count = 0           # 임계 이하 연속 카운트

        self.sos_time = None         # SOS 수신 시각
        self.post_end_time = None    # POST_TURN_ADVANCE 종료 시각

        self.get_logger().info('[first_course] Ready. Waiting for /object_detection/sos')

    # ---------------- 콜백 ---------------- #
    def cb_sos(self, _msg: Empty):
        """SOS 수신 → CRUISE 시작(단, IMU yaw가 들어와 있어야 함)."""
        if self.phase != Phase.IDLE:
            return
        if self.yaw_now is None:
            self.get_logger().warn('SOS 수신: 아직 IMU yaw 없음. yaw 수신 후 시작.')
            return
        self.yaw_ref = self.yaw_now
        self.phase = Phase.CRUISE
        self.sos_time = self.get_clock().now()  # turn-arm 딜레이 기준점
        self.get_logger().info('=== START: CRUISE === (turn-arm 딜레이 동작 중)')

    def cb_imu(self, msg: Imu):
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        yaw = quat_to_yaw(qx, qy, qz, qw)
        if self.invert_yaw_sign:
            yaw = -yaw
        self.yaw_now = yaw

    def cb_front_dist(self, msg: Float32):
        self.front_dist = float(msg.data)

    # ---------------- 메인 루프 ---------------- #
    def on_timer(self):
        self.state_pub.publish(String(data=self.phase.name))

        if self.phase in (Phase.IDLE, Phase.CRUISE) and self.yaw_now is None:
            return

        # 회전 트리거 arming 상태(= 지연 시간 경과 여부) 계산
        armed = False
        if self.sos_time is not None:
            elapsed = (self.get_clock().now() - self.sos_time).nanoseconds * 1e-9
            armed = (elapsed >= self.turn_arm_delay_sec)

        if self.phase == Phase.IDLE:
            # 경기 중 정지 금지: 0,0 명령 발행하지 않음(그냥 아무 것도 안 보냄)
            return

        if self.phase == Phase.CRUISE:
            # 1) 직진 + yaw 정렬
            lin = self.cruise_speed
            ang = self.heading_hold(self.yaw_ref, self.yaw_now)

            # 2) (armed가 된 이후에만) 전방 거리로 회전 트리거 판단
            if armed and self.front_dist is not None:
                # 히스테리시스 포함 연속 감지
                thr = self.front_threshold
                if self.front_dist <= thr:
                    self.hit_count += 1
                elif self.front_dist >= thr + self.front_hyst:
                    self.hit_count = 0

                if self.hit_count >= self.consecutive_hits_req:
                    self.target_yaw = wrap_pi(self.yaw_now + self.right_turn_rad)
                    self.phase = Phase.TURNING_RIGHT
                    self.hit_count = 0
                    self.get_logger().info(
                        f'[ARMED] 전방 {self.front_dist:.2f} m → TURNING_RIGHT '
                        f'(target yaw={math.degrees(self.target_yaw):.1f}°)'
                    )
                    # *** 정지 금지: 0,0 발행하지 않고 바로 회전 루프로 이동 ***
                    return
            elif not armed:
                # 아직 회전 트리거 비활성 상태를 가볍게 로깅(과도한 로그 방지 위해 가끔만)
                pass

            self.publish_cmd(lin, ang)
            return

        if self.phase == Phase.TURNING_RIGHT:
            # 선속도 0, 각속도는 yaw 오차 × Kp (제한 포함)
            err = ang_err(self.target_yaw, self.yaw_now)
            ang = clamp(self.turn_kp * err, -self.turn_max_rate, self.turn_max_rate)
            lin = 0.0
            self.publish_cmd(lin, ang)

            # 완료 판정
            if abs(err) <= self.turn_tol and abs(ang) < 0.2 * self.turn_max_rate:
                self.yaw_ref = self.target_yaw
                self.phase = Phase.POST_TURN_ADVANCE
                self.post_end_time = self.get_clock().now() + Duration(seconds=self.post_advance_sec)
                self.get_logger().info('회전 완료 → POST_TURN_ADVANCE (전진 유지)')
            return

        if self.phase == Phase.POST_TURN_ADVANCE:
            now = self.get_clock().now()
            lin = self.cruise_speed
            ang = self.heading_hold(self.yaw_ref, self.yaw_now)
            self.publish_cmd(lin, ang)

            if self.post_end_time is not None and now >= self.post_end_time:
                # *** 정지 금지: DONE으로 가서 멈추지 않고 다시 CRUISE 복귀 ***
                self.phase = Phase.CRUISE
                # CRUISE로 돌아가면 turn-arm은 이미 끝났으니 전방 감지 즉시 활성
                self.get_logger().info('POST 구간 종료 → CRUISE 복귀(주행 지속)')
            return

    # ---------------- 보조 함수 ---------------- #
    def heading_hold(self, yaw_ref, yaw_now):
        """비례제어(P제어): 회전속도 = KP × (목표-현재). 데드밴드/제한 포함."""
        e = ang_err(yaw_ref, yaw_now)
        if abs(e) < self.yaw_deadband:
            return 0.0
        w = self.yaw_kp * e
        return clamp(w, -self.yaw_max_rate, self.yaw_max_rate)

    def publish_cmd(self, lin, ang):
        """/cmd_vel 발행. (경기 규칙상 0,0 ‘정지’ 명령은 의도적으로 만들지 않음)"""
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = FirstCourseController()
    try:
        rclpy.spin(node)
    finally:
        # 경기 규칙: 정지 명령(0,0) 금지 → 종료 시에도 별도 stop 발행하지 않음
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
