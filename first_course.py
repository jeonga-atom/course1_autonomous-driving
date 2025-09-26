#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32, String


# ======= 수정 가능한 기본 값 ========
CRUISE_SPPED        =   0.25          # 차체의 속도 [m/s]
POST_ADVANCE_SEC    =   3.0           # 회전 후 전진 시간 [s]
RIGHT_TURN_DEG      =   90.0          # 우회전 각도 [deg]
YAW_KP              =   1.2           # 보정 민감도
FRONT_THRESHOLD_M   =   0.5           # 전방 임계 거리 [m]
TURN_MAX_RATE       =   1.5           # 회전 중 최대 각속도 [rad/s]
FRONT_IGNORE_SEC    =   5.0           # 시작 후 전방 거리 무시 시간(초), 시간 = 거리/속력 대회장에서 약 19초 일듯?
# =================================

# ------------------ 유틸 ------------------ #
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def wrap_pi(a):
    """[-pi, pi] 범위로 각도 래핑"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def ang_err(target, current):
    """target - current 를 [-pi, pi]로"""
    return wrap_pi(target - current)

def quat_to_yaw(qx, qy, qz, qw):
    """
    ROS ENU 기준 Z-축 yaw 추정.
    yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    """
    s1 = 2.0 * (qw * qz + qx * qy)
    s2 = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(s1, s2)


# ---------------- 상태 정의 ---------------- #
class Phase(Enum):
    IDLE =              0
    CRUISE =            1
    TURNING_RIGHT =     2
    POST_TURN_ADVANCE = 3
    DONE =              4


# --------------- 메인 노드 ----------------- #
class FirstCourseController(Node):
    """
    상태 흐름:
      IDLE:  /object_detection/sos 수신 대기
      CRUISE: 직진 + yaw 정렬(계속) / 전방 <= threshold면 회전으로 전환
      TURNING_RIGHT: IMU yaw 기준 -90° 회전(우회전), 폐루프 제어
      POST_TURN_ADVANCE: 회전 완료 후 3초 전진(새 yaw로 정렬 유지)
      DONE: 정지 유지(불필요한 0 발행 억제)

    토픽:
      Sub:
        - /object_detection/sos (std_msgs/Empty): 시작 트리거
        - /imu/data (sensor_msgs/Imu): yaw 계산
        - /perception/front_distance (std_msgs/Float32): 전방 거리(m)
      Pub:
        - /cmd_vel (geometry_msgs/Twist): 주행/회전 명령
        - /first_course/state (std_msgs/String): 디버그 상태 표시
    """

    def __init__(self):
        super().__init__('first_course')

        # -------- 파라미터 선언 -------- #
        # 토픽
        self.declare_parameter('topic_sos',         '/object_detection/sos')
        self.declare_parameter('topic_imu',         '/imu/data')
        self.declare_parameter('topic_front_dist',  '/perception/front_distance')
        self.declare_parameter('topic_cmd_vel',     '/cmd_vel')

        # 주행/회전 파라미터
        self.declare_parameter('cruise_speed',          CRUISE_SPPED)       # m/s
        self.declare_parameter('yaw_kp',                YAW_KP)             # 직진 중 yaw P이득 (rad/s per rad)
        self.declare_parameter('yaw_deadband_deg',      2.0)                # 직진 deadband (deg)
        self.declare_parameter('yaw_max_rate',          1.2)                # 직진 중 최대 각속도 제한(rad/s)

        self.declare_parameter('front_threshold_m',     FRONT_THRESHOLD_M)  # 회전 트리거 거리(m)
        self.declare_parameter('front_hysteresis_m',    0.1)                # 채터링 방지 히스테리시스
        self.declare_parameter('consecutive_hits',      2)                  # 임계 이하 연속 감지 횟수

        self.declare_parameter('turn_kp',               2.0)                # 회전 중 yaw P이득
        self.declare_parameter('turn_max_rate',         TURN_MAX_RATE)      # 회전 중 최대 각속도(rad/s)
        self.declare_parameter('turn_tol_deg',          2.0)                # 회전 완료 오차(deg)
        self.declare_parameter('right_turn_deg',        RIGHT_TURN_DEG)     # 우회전 각도(+는 좌, -는 우; 아래에서 -로 씀)

        self.declare_parameter('post_advance_sec',      POST_ADVANCE_SEC)   # 회전 후 전진 시간(s)
        self.declare_parameter('control_rate_hz',       20.0)               # 제어 주기(Hz)
        self.declare_parameter('front_ignore_sec',      FRONT_IGNORE_SEC)   # 시작 후 전방 거리 무시 시간(초)


        # IMU 방향성 보정 (센서 축 정의가 다를 경우)
        self.declare_parameter('invert_yaw_sign',       False)

        # 0,0 명령 억제(네트워크 절약용). 정지 전환 시 1회만 0을 발행하고 이후 억제.
        self.declare_parameter('suppress_zero_cmd',     True)

        # -------- 파라미터 로드 -------- #
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
        self.suppress_zero_cmd = bool(p('suppress_zero_cmd').value)
        
        self.front_ignore_sec = float(p('front_ignore_sec').value)


        # -------- 내부 상태 -------- #
        self.phase = Phase.IDLE
        self.state_pub = self.create_publisher(String, '/first_course/state', 10)

        self.cmd_pub = self.create_publisher(Twist, self.topic_cmd_vel, 10)

        self.sub_sos = self.create_subscription(Empty, self.topic_sos, self.cb_sos, 10)
        self.sub_imu = self.create_subscription(Imu, self.topic_imu, self.cb_imu, qos_profile_sensor_data)
        self.sub_front = self.create_subscription(Float32, self.topic_front_dist, self.cb_front_dist, qos_profile_sensor_data)

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.yaw_now            = None        # 현재 yaw(rad)
        self.yaw_ref            = None        # 정렬 기준 yaw(rad)
        self.target_yaw         = None        # 회전 목표 yaw(rad)
        self.front_dist         = None        # 최신 전방 거리(m)
        self.hit_count          = 0           # 임계 이하 연속 카운트

        self.post_end_time      = None        # post-advance 종료 시각
        self.last_zero_sent     = False       # 0,0 명령 1회 발행 여부

        self.front_ignore_until = None        # SOS 이후 front_distance 무시 종료 시각

        self.get_logger().info('[first_course] Ready. Waiting for /object_detection/sos')

    # ------------- 콜백들 ------------- #
    def cb_sos(self, _msg: Empty):
        if self.phase != Phase.IDLE:
            return
        if self.yaw_now is None:
            self.get_logger().warn('============== SOS 수신 대기 ===========')
            # yaw 들어오면 on_timer에서 자동 전이
            self.phase = Phase.IDLE
            return
        self.yaw_ref = self.yaw_now             # sos 받기 직전의 값으로 imu가 정면을 인식
        self.phase = Phase.CRUISE
        self.last_zero_sent = False

        self.front_ignore_until = self.get_clock().now() + Duration(seconds=self.front_ignore_sec)

        self.get_logger().info('============= START:================')

    def cb_imu(self, msg: Imu):
        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        yaw = quat_to_yaw(qx, qy, qz, qw)
        if self.invert_yaw_sign:
            yaw = -yaw
        self.yaw_now = yaw

    def cb_front_dist(self, msg: Float32):
        self.front_dist = float(msg.data)

    # ----------- 상태 머신 주기 처리 ----------- #
    def on_timer(self):
        # 상태 디버그 토픽
        self.state_pub.publish(String(data=self.phase.name))

        # yaw가 아직 없고 시작 요청 받은 상태면 대기
        if self.phase in (Phase.IDLE, Phase.CRUISE) and self.yaw_now is None:
            return

        if self.phase == Phase.IDLE:
            # 불필요한 0 발행 억제
            self.maybe_publish_zero_once()
            return

        if self.phase == Phase.CRUISE:
            lin = self.cruise_speed
            ang = self.heading_hold(self.yaw_ref, self.yaw_now)

            # 무시 타이머가 살아있으면 전방거리 체크 스킵
            now = self.get_clock().now()
            if self.front_ignore_until is not None and now < self.front_ignore_until:
                # 필요하면 남은 시간 디버그 출력(스팸 방지 위해 간헐적으로만 권장)
                # rem = (self.front_ignore_until - now).nanoseconds / 1e9
                # self.get_logger().debug(f'front_distance 무시 중... {rem:.1f}s 남음')
                self.publish_cmd(lin, ang)
                return
            else:
                # 타이머 만료 후에는 None으로 정리(선택)
                if self.front_ignore_until is not None:
                    self.front_ignore_until = None
                    self.get_logger().info('front_distance 무시 기간 종료 → 전방 감지 활성화')

            # 전방거리 체크(연속 카운트로 채터 방지)
            if self.front_dist is not None:
                if self.front_dist <= self.front_threshold:
                    self.hit_count += 1
                elif self.front_dist >= self.front_threshold + self.front_hyst:
                    self.hit_count = 0

                if self.hit_count >= self.consecutive_hits_req:
                    # 우회전 목표 yaw 설정: 현재 yaw 기준 -90°
                    self.target_yaw = wrap_pi(self.yaw_now + self.right_turn_rad)
                    self.phase = Phase.TURNING_RIGHT
                    self.hit_count = 0
                    self.get_logger().info(
                        f'전방 {self.front_dist:.2f} m 감지 → 회전, 목표 yaw={math.degrees(self.target_yaw):.1f}°)'
                    )
                    # 회전 진입 시 즉시 정지(선속도 0 한번 보내줌)
                    self.publish_cmd(0.0, 0.0, force=True)
                    return

            self.publish_cmd(lin, ang)
            return

        if self.phase == Phase.TURNING_RIGHT:
            # 회전은 선속도 0, 각속도 P제어(제한)
            err = ang_err(self.target_yaw, self.yaw_now)
            ang = clamp(self.turn_kp * err, -self.turn_max_rate, self.turn_max_rate)
            lin = 0.0
            self.publish_cmd(lin, ang)

            # 완료 판정
            if abs(err) <= self.turn_tol and abs(ang) < 0.2 * self.turn_max_rate:
                self.yaw_ref = self.target_yaw  # 새 진행 방향을 기준으로 유지
                self.phase = Phase.POST_TURN_ADVANCE
                self.post_end_time = self.get_clock().now() + Duration(seconds=self.post_advance_sec)
                self.get_logger().info('회전 완료 → 3초 전진')
            return

        if self.phase == Phase.POST_TURN_ADVANCE:
            now = self.get_clock().now()
            lin = self.cruise_speed
            ang = self.heading_hold(self.yaw_ref, self.yaw_now)
            self.publish_cmd(lin, ang)

            if self.post_end_time is not None and now >= self.post_end_time:
                self.phase = Phase.DONE
                self.get_logger().info('목표 구간 완료 → 정지!!! ')
                # 정지 명령 1회
                self.publish_cmd(0.0, 0.0, force=True)
            return

        if self.phase == Phase.DONE:
            self.maybe_publish_zero_once()
            return

    # ----------- 제어 보조 함수들 ----------- #
    def heading_hold(self, yaw_ref, yaw_now):
        """직진 중 yaw 정렬용 P제어(데드밴드 + 제한)"""
        e = ang_err(yaw_ref, yaw_now)
        if abs(e) < self.yaw_deadband:
            return 0.0
        w = self.yaw_kp * e
        return clamp(w, -self.yaw_max_rate, self.yaw_max_rate)

    def publish_cmd(self, lin, ang, force=False):
        """
        suppress_zero_cmd=True 인 경우,
        (lin, ang) 둘 다 0이면 네트워크 절약을 위해 반복 발행하지 않음.
        단, force=True면 1회는 반드시 발행.
        """
        if not force and self.suppress_zero_cmd and lin == 0.0 and ang == 0.0:
            # 0명령 억제
            return
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)
        if lin == 0.0 and ang == 0.0:
            self.last_zero_sent = True
        else:
            self.last_zero_sent = False

    def maybe_publish_zero_once(self):
        if self.suppress_zero_cmd:
            if not self.last_zero_sent:
                self.publish_cmd(0.0, 0.0, force=True)
        else:
            self.publish_cmd(0.0, 0.0, force=True)


def main():
    rclpy.init()
    node = FirstCourseController()
    try:
        rclpy.spin(node)
    finally:
        # 종료 시 정지 명령 1회
        try:
            node.publish_cmd(0.0, 0.0, force=True)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
