#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corridor_navigator.py
- /perception/left_distance, /perception/right_distance, /perception/front_distance (Float32) 사용
- 초기 20초(가변) 동안: 좌/우만 이용해 '가운데 정렬' 주행 (front는 아예 구독하지 않음)
- 20초 이후: 좌/우 구독을 해제하고, front만 구독(좌/우는 받지 않음)
- 가운데 정렬 로직:
    * 좌/우 중 하나가 0.5m 이하 → 반대편으로 '약하게' 회전
    * 좌/우 중 하나가 0.2m 이하 → 반대편으로 '강하게' 회전
- 기본 전진 속도로 복도를 진행 (회전은 angular.z로 보정)
- 복도 폭 가정: 2.0 m, 차체 폭 0.7 m → 이론적 중앙에서 좌/우 약 0.65 m (참고값)
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


class CorridorNavigator(Node):
    def __init__(self):
        super().__init__('corridor_navigator')

        # ===================== 튜닝 가능한 기본 파라미터 =====================
        # 정렬 유지 시간(초) - 런타임에서 ros2 param으로 바꾸고 싶으면 declare_parameter 사용
        self.declare_parameter('align_duration_sec', 20.0)
        self.align_duration_sec: float = float(self.get_parameter('align_duration_sec').value)

        # 주행 속도 및 회전량 (필요 시 조정)
        self.LIN_SPEED = 0.25        # [m/s] 기본 전진 속도
        self.YAW_GENTLE = 0.25       # [rad/s] 약한 회전
        self.YAW_STRONG = 0.60       # [rad/s] 강한 회전

        # 임계값 (요청 조건)
        self.THR_NEAR = 0.5          # [m] 0.5m 이내 → 약하게 반대편
        self.THR_VERY_NEAR = 0.2     # [m] 0.2m 이내 → 강하게 반대편

        # 제어 주기 (컨트롤 루프)
        self.CTRL_HZ = 20.0          # [Hz]
        self.dt = 1.0 / self.CTRL_HZ

        # 참고: 복도 폭 2.0m, 차체 폭 0.7m → 중앙 기준 좌/우 기대 거리 약 0.65m
        self.EXPECTED_SIDE = 0.65    # [m] (직접 사용하진 않지만, 주석 참고용/향후 PID 등 확장용)

        # ===================== 퍼블리셔 =====================
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # ===================== 구독자(초기 상태: 좌/우만) =====================
        # 초기 20초 동안은 좌/우만 구독해서 가운데 정렬 → front는 아예 구독하지 않음
        self.sub_left  = self.create_subscription(
            Float32, '/perception/left_distance',  self.cb_left,  qos_profile_sensor_data
        )
        self.sub_right = self.create_subscription(
            Float32, '/perception/right_distance', self.cb_right, qos_profile_sensor_data
        )
        self.sub_front = None  # 20초 후에 생성

        # 최신 거리값 저장 변수 (None이면 아직 수신 전/무시)
        self.left_dist  = None
        self.right_dist = None
        self.front_dist = None

        # ===================== 상태 관리 =====================
        # 상태 A: 'ALIGN' (좌/우만) → align_duration_sec 초 유지 후
        # 상태 B: 'FRONT' (front만)
        self.state = 'ALIGN'
        self.start_t = time.monotonic()

        self.get_logger().info(
            f"[{self.state}] 시작. 좌/우 라이다로 가운데 정렬 {self.align_duration_sec:.1f}s 유지. "
            f"이 동안 front는 구독하지 않습니다."
        )

        # 제어 타이머
        self.timer = self.create_timer(self.dt, self.control_loop)

    # ===================== 콜백: 거리 수신 =====================
    def cb_left(self, msg: Float32):
        # ALIGN 상태에서만 의미 있음 (FRONT 상태가 되면 구독 자체를 해제)
        self.left_dist = self._sanitize(msg.data)

    def cb_right(self, msg: Float32):
        # ALIGN 상태에서만 의미 있음 (FRONT 상태가 되면 구독 자체를 해제)
        self.right_dist = self._sanitize(msg.data)

    def cb_front(self, msg: Float32):
        # FRONT 상태에서만 의미 있음 (ALIGN 상태에서는 front 구독을 만들지 않음)
        self.front_dist = self._sanitize(msg.data)

    # ===================== 제어 루프 =====================
    def control_loop(self):
        now = time.monotonic()

        # ----- 상태 전환: ALIGN → FRONT -----
        if self.state == 'ALIGN' and (now - self.start_t) >= self.align_duration_sec:
            self._switch_to_front_only()

        # ----- 상태별 제어 -----
        if self.state == 'ALIGN':
            twist = self._compute_twist_align()
        else:  # self.state == 'FRONT'
            twist = self._compute_twist_front()

        # 퍼블리시
        self.pub_cmd.publish(twist)

    # ===================== ALIGN 제어(좌/우 중심 정렬) =====================
    def _compute_twist_align(self) -> Twist:
        """
        좌/우 중 하나가 0.5m 이하면 '약하게' 반대편 회전,
        0.2m 이내면 '강하게' 반대편 회전.
        그 외에는 직진 유지.
        """
        ang = 0.0

        # 안전성: 아직 수신 안 된 경우(None)는 보정 없이 직진
        L = self.left_dist
        R = self.right_dist

        # 회전 방향 기준:
        #  - 왼쪽이 너무 가까우면(=왼벽에 붙음) → 오른쪽으로 회전(시계방향, angular.z 음수)
        #  - 오른쪽이 너무 가까우면(=오른벽에 붙음) → 왼쪽으로 회전(반시계, angular.z 양수)

        if L is not None and L <= self.THR_VERY_NEAR:
            ang = -self.YAW_STRONG
        elif R is not None and R <= self.THR_VERY_NEAR:
            ang = +self.YAW_STRONG
        elif L is not None and L <= self.THR_NEAR:
            ang = -self.YAW_GENTLE
        elif R is not None and R <= self.THR_NEAR:
            ang = +self.YAW_GENTLE
        else:
            ang = 0.0  # 중앙에 가까움 → 직진

        # 트위스트 구성
        tw = Twist()
        tw.linear.x  = self.LIN_SPEED   # 전진 유지
        tw.angular.z = ang              # 좌/우에 따라 보정
        return tw

    # ===================== FRONT 제어(front만 구독) =====================
    def _compute_twist_front(self) -> Twist:
        """
        요청: front는 우선 별다른 플래그 없이 '받기만' 하면 됨.
        여기서는 단순히 직진을 유지(추후 원하시면 앞벽 임계값에 따라 감속/정지 로직을 추가하세요).
        """
        tw = Twist()
        tw.linear.x  = self.LIN_SPEED   # 계속 전진
        tw.angular.z = 0.0              # 좌/우는 더 이상 받지 않으므로 회전 보정 없음
        return tw

    # ===================== 상태 전환 유틸 =====================
    def _switch_to_front_only(self):
        """20초 경과 시 좌/우 구독 해제 → front만 구독하도록 전환."""
        self.state = 'FRONT'

        # 좌/우 구독 해제(실제로 더 이상 메시지를 '받지 않음')
        if self.sub_left:
            self.destroy_subscription(self.sub_left)
            self.sub_left = None
        if self.sub_right:
            self.destroy_subscription(self.sub_right)
            self.sub_right = None

        # front 구독 시작
        if self.sub_front is None:
            self.sub_front = self.create_subscription(
                Float32, '/perception/front_distance', self.cb_front, qos_profile_sensor_data
            )

        self.get_logger().info(
            "[FRONT]로 전환. 이제 front만 구독하고, 좌/우는 더 이상 받지 않습니다."
        )

    # ===================== 헬퍼 =====================
    @staticmethod
    def _sanitize(x: float):
        """음수/NaN/inf 등 비정상 값은 None 처리."""
        if x is None:
            return None
        if math.isnan(x) or math.isinf(x) or x < 0.0:
            return None
        return float(x)


def main():
    rclpy.init()
    node = CorridorNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
