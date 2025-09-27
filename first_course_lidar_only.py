#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
라이다 좌/우만으로 '가면서 회전'하는 아주 단순한 중앙 유지 컨트롤러
- IMU 사용 안 함
- 정면 라이다 사용 안 함(구독 안함 → 사실상 OFF)
- 시작 후 2초(가변)는 좌/우 라이다를 '무시'하고 직진(복도 진입/초기 안정)
- 그 이후에는 좌/우 임계(기본 0.5 m)만으로 회전 바이어스 부여
  * 좌가 0.5 m 이하 → 오른쪽으로 돌면서 전진(ang<0)
  * 우가 0.5 m 이하 → 왼쪽으로 돌면서 전진(ang>0)
  * 둘 다 여유 → 직진
- 멈춰서 회전하지 않고 '항상 전진하면서' 회전(turn-in-motion)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

from std_msgs.msg import Empty, Float32, String
from geometry_msgs.msg import Twist


class Phase:
    IDLE = 0
    CRUISE = 1
    DONE = 2


class LidarSideOnly(Node):
    def __init__(self):
        super().__init__('first_course_lidar_only')

        # -------------------- 파라미터 --------------------
        # 토픽명
        self.declare_parameter('topic_sos', '/object_detection/sos')
        self.declare_parameter('topic_cmd_vel', '/cmd_vel')
        self.declare_parameter('topic_left_dist', '/perception/left_distance')
        self.declare_parameter('topic_right_dist', '/perception/right_distance')

        # 주행/제어
        self.declare_parameter('cruise_speed',      0.25)       # [m/s] 기본 전진 속도
        self.declare_parameter('turn_rate',         1.2)        # [rad/s] 회전 세기(좌/우 임계 위반 시 부호만 바뀌어 적용)
        self.declare_parameter('side_threshold_m',  0.50)       # [m] 좌/우 임계(이 이하이면 벽에 '가깝다'고 판단)
        self.declare_parameter('side_hysteresis_m', 0.05)       # [m] 채터 억제용 히스테리시스
        self.declare_parameter('consecutive_hits',  2)          # 임계 이하 연속 샘플 수(채터 억제)
        self.declare_parameter('control_rate_hz',   10.0)       # 제어 주기 [Hz]

        # 시작 후 라이다 '무시' 시간(복도 진입용) — 이 동안은 좌/우 측정값을 사용하지 않고 직진만 함
        self.declare_parameter('side_ignore_sec', 2.0)          # [s] 요구사항: 시작 후 2초 무시(가변)

        # 네트워크 절약: (0,0) 반복 발행 억제
        self.declare_parameter('suppress_zero_cmd', True)

        # -------------------- 파라미터 로드 --------------------
        g = self.get_parameter
        self.topic_sos = g('topic_sos').value
        self.topic_cmd_vel = g('topic_cmd_vel').value
        self.topic_left = g('topic_left_dist').value
        self.topic_right = g('topic_right_dist').value

        self.v = float(g('cruise_speed').value)
        self.w_turn = float(g('turn_rate').value)
        self.th = float(g('side_threshold_m').value)
        self.hyst = float(g('side_hysteresis_m').value)
        self.nhits = int(g('consecutive_hits').value)
        self.dt = 1.0 / float(g('control_rate_hz').value)
        self.side_ignore_sec = float(g('side_ignore_sec').value)
        self.suppress_zero_cmd = bool(g('suppress_zero_cmd').value)

        # -------------------- 내부 상태 --------------------
        self.phase = Phase.IDLE
        self.left = None
        self.right = None
        self.hitL = 0
        self.hitR = 0
        self.start_time = None
        self.last_zero_sent = False

        # -------------------- I/O --------------------
        self.state_pub = self.create_publisher(String, '/first_course/state', 10)
        self.cmd_pub = self.create_publisher(Twist, self.topic_cmd_vel, 10)

        self.sub_sos = self.create_subscription(Empty, self.topic_sos, self.cb_sos, 10)
        self.sub_left = self.create_subscription(Float32, self.topic_left, self.cb_left, qos_profile_sensor_data)
        self.sub_right = self.create_subscription(Float32, self.topic_right, self.cb_right, qos_profile_sensor_data)

        self.timer = self.create_timer(self.dt, self.on_timer)

        self.get_logger().info('[lidar_side_only] Ready. Waiting for /object_detection/sos')

    # ==================== 콜백 ====================
    def cb_sos(self, _msg: Empty):
        """시작 트리거: CRUISE 진입 + 시작시각 기록"""
        if self.phase != Phase.IDLE:
            return
        self.phase = Phase.CRUISE
        self.start_time = self.get_clock().now()
        self.last_zero_sent = False
        self.get_logger().info('============= START (side-only, turn-in-motion) =============')

    def cb_left(self, msg: Float32):
        self.left = float(msg.data)

    def cb_right(self, msg: Float32):
        self.right = float(msg.data)

    # ==================== 메인 루프 ====================
    def on_timer(self):
        # 상태 브로드캐스트(디버그용)
        self.state_pub.publish(String(data={0: 'IDLE', 1: 'CRUISE', 2: 'DONE'}[self.phase]))

        if self.phase == Phase.IDLE:
            # 대기 중에는 0,0을 한 번만 발행
            self.maybe_zero_once()
            return

        if self.phase == Phase.CRUISE:
            # 항상 '가면서' 제어: 기본 선속도는 유지
            lin = self.v
            ang = 0.0

            # 1) 시작 후 side_ignore_sec 동안은 좌/우 라이다를 '무시'하고 직진만 함
            #    - 복도에 아직 진입하지 않았을 수 있으므로 초기 과민 반응 방지
            if self.start_time is not None:
                elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
                if elapsed < self.side_ignore_sec:
                    # (옵션) 첫 통과 시점에 한 번만 로그
                    if abs(elapsed - 0.0) < 2 * self.dt:
                        self.get_logger().info(f'side-ignore {self.side_ignore_sec:.1f}s → straight cruise only')
                    self.publish_cmd(lin, ang)  # 직진 유지(멈추지 않음)
                    return
                else:
                    # 무시 기간 종료 시 한 번만 로그
                    if abs(elapsed - self.side_ignore_sec) < 2 * self.dt:
                        self.get_logger().info('side-ignore done → start side-based steering')

            # 2) 좌/우 임계 검사(아주 단순)
            #    - 히스테리시스 + 연속 감지로 채터 억제
            if self.left is not None:
                if self.left <= self.th:
                    self.hitL += 1
                elif self.left >= self.th + self.hyst:
                    self.hitL = 0
            if self.right is not None:
                if self.right <= self.th:
                    self.hitR += 1
                elif self.right >= self.th + self.hyst:
                    self.hitR = 0

            # 3) 조향 결정(항상 전진하면서 회전)
            #    - 좌가 가깝다(임계 이하 연속 만족) → 오른쪽으로 틀기(ang<0)
            #    - 우가 가깝다 → 왼쪽으로 틀기(ang>0)
            #    - 둘 다 여유 → 직진
            if self.hitL >= self.nhits and (self.right is None or self.right >= self.th):
                ang = -self.w_turn
                self.get_logger().info(f'LEFT {self.left:.2f} m ≤ {self.th:.2f} m → turn RIGHT (ang={-self.w_turn:.2f})')
            elif self.hitR >= self.nhits and (self.left is None or self.left >= self.th):
                ang = +self.w_turn
                self.get_logger().info(f'RIGHT {self.right:.2f} m ≤ {self.th:.2f} m → turn LEFT (ang={self.w_turn:.2f})')
            else:
                ang = 0.0  # 중앙 유지

            # 4) 명령 발행(항상 전진)
            self.publish_cmd(lin, ang)
            return

        if self.phase == Phase.DONE:
            self.maybe_zero_once()
            return

    # ==================== 유틸 ====================
    def publish_cmd(self, lin, ang, force=False):
        """(0,0) 반복 발행 억제 옵션 포함한 cmd_vel 퍼블리시"""
        if not force and self.suppress_zero_cmd and lin == 0.0 and ang == 0.0:
            return
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)
        self.last_zero_sent = (lin == 0.0 and ang == 0.0)

    def maybe_zero_once(self):
        """정지 명령을 딱 한 번만 보내도록 보조"""
        if self.suppress_zero_cmd:
            if not self.last_zero_sent:
                self.publish_cmd(0.0, 0.0, force=True)
        else:
            self.publish_cmd(0.0, 0.0, force=True)


def main():
    rclpy.init()
    node = LidarSideOnly()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.publish_cmd(0.0, 0.0, force=True)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
