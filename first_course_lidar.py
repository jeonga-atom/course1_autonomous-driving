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

- ALIGN → FRONT → TURN_RIGHT → DRIVE_OUT → DONE 로 동작함
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Empty
from geometry_msgs.msg import Twist

# =============== 수정 가능한 기본 값 =================
LINE_SPEED_CALL                 = 0.25      # [m/s] 기본 전진 속도

ALIGN_DURATION_SEC              = 30.0      # 좌/우 정렬 유지 시간 [s] -> 이후에는 front만 구독
FRONT_TURN_TRIGGER              = 0.4       # front 벽 임계값 [m]
TURN_SPEED                      = 0.7       # 회전 각속도 [rad/s]
DEG_TOLERANCE                   = 2.0       # 허용 오차 [deg]
FRONT_REARM_DIST                = 1.20      # 회전 끝난 뒤, front가 이 거리 이상 멀어져야 재트리거 허용
TURN_COOLDOWN_SEC               = 2.0       # 회전 종료 후 쿨다운 시간 [s]

THR_NEAR_CALL                   = 0.38       # [m] 옆 벽 임계값 → 약하게 반대편으로 / 대회에서는 0.5~0.55정도
THR_VERY_NEAR_CALL              = 0.30       # [m] 더 가까울 때 옆 벽 임계값 → 강하게 반대편으로/ 대회에서는 0.35~0.40 정도
YAW_GENTLE_CALL                 = 0.25      # [rad/s] 약한 회전 (0.25 rad/s -> 약 14.3°/s)/ 대회에서 0.25로 사용해도 될듯?
YAW_STRONG_CALL                 = 0.35      # [rad/s] 강한 회전 (0.60 rad/s -> 약 34.4°/s)/ 대회에서는 0.50~0.60 정도

CTRL_HZ_CALL                    = 10.0      # [Hz] 기본 값 10hz
DRIVE_OUT_SEC_CALL              = 5.0       # 회전 완료 후 전진 유지 시간(초)

# -------------- 유틸 함수: 쿼터니언→Yaw, 각도 정규화 ------------------
def yaw_from_quat(x, y, z, w) -> float: # 쿼터니언 → Z-Yaw(rad)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def normalize_angle(angle: float) -> float: # 임의의 라디안 각도를 [-pi, pi]로 정규화
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def shortest_angular_distance(from_ang: float, to_ang: float) -> float: # from → to 최단 각(rad, [-pi, pi])
    return normalize_angle(to_ang - from_ang)


class CorridorNavigator(Node):
    def __init__(self):
        super().__init__('corridor_navigator')

        # ------------------- 튜닝 가능한 기본 파라미터 ----------------------
        # 정렬 유지 시간(초) - 런타임에서 ros2 param으로 바꾸고 싶으면 declare_parameter 사용
        self.declare_parameter('align_duration_sec', ALIGN_DURATION_SEC)   
        self.declare_parameter('front_turn_trigger', FRONT_TURN_TRIGGER)   
        self.declare_parameter('turn_speed',         TURN_SPEED)            
        self.declare_parameter('deg_tolerance',      DEG_TOLERANCE)        
        self.declare_parameter('line_speed',         LINE_SPEED_CALL)
        self.declare_parameter('yaw_gentle',         YAW_GENTLE_CALL)
        self.declare_parameter('yaw_strong',         YAW_STRONG_CALL)
        self.declare_parameter('thr_near',           THR_NEAR_CALL)
        self.declare_parameter('thr_very_near',      THR_VERY_NEAR_CALL)
        self.declare_parameter('ctrl_hz',            CTRL_HZ_CALL)
        self.declare_parameter('drive_out_sec',      DRIVE_OUT_SEC_CALL)

        self.align_duration_sec: float  = float(self.get_parameter('align_duration_sec').value)
        self.FRONT_TURN_TRIG            = float(self.get_parameter('front_turn_trigger').value)
        self.TURN_SPEED                 = float(self.get_parameter('turn_speed').value)
        self.DEG_TOL                    = float(self.get_parameter('deg_tolerance').value)
        # 주행 속도 및 회전량 (필요 시 조정)
        self.LIN_SPEED          = float(self.get_parameter('line_speed').value)
        self.YAW_GENTLE         = float(self.get_parameter('yaw_gentle').value)
        self.YAW_STRONG         = float(self.get_parameter('yaw_strong').value)
        # 임계값 (요청 조건)
        self.THR_NEAR           = float(self.get_parameter('thr_near').value)  
        self.THR_VERY_NEAR      = float(self.get_parameter('thr_very_near').value)
        # 제어 주기 (컨트롤 루프)
        self.CTRL_HZ            = float(self.get_parameter('ctrl_hz').value)
        self.dt = 1.0 / self.CTRL_HZ
        # 참고: 복도 폭 2.0m, 차체 폭 0.7m → 중앙 기준 좌/우 기대 거리 약 0.65m 여유가 필요
        self.EXPECTED_SIDE      = 0.65    # [m] (직접 사용하진 않지만, 주석 참고용/향후 PID 등 확장용)
        self.drive_out_sec      = float(self.get_parameter('drive_out_sec').value)

        self.clock = self.get_clock()

        # -------------------------------- 토픽 -------------------------------
        # 초기 20초 동안은 좌/우만 구독해서 가운데 정렬 → front는 아예 구독하지 않음
        self.sub_left  = self.create_subscription(Float32, '/perception/left_distance',  self.cb_left,  qos_profile_sensor_data)
        self.sub_right = self.create_subscription(Float32, '/perception/right_distance', self.cb_right, qos_profile_sensor_data)
        self.sub_imu   = self.create_subscription(Imu, '/imu/data', self.cb_imu, qos_profile_sensor_data)
        self.sub_sos   = self.create_subscription(Empty, '/object_detection/sos', self.cb_sos, 10)
        self.pub_cmd   = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_front = None  # 20초 후에 생성

        # -------------- 최신 거리값 저장 변수 (None이면 아직 수신 전/무시) -------------
        self.left_dist          = None
        self.right_dist         = None
        self.front_dist         = None
        self.current_yaw        = None      # 최신 imu yaw
        self.turn_target_yaw    = None      # 목표 yaw (90도 우회전 후)
        self.started            = False
        self.turned_once        = False     # 이미 한 번 회전했는지
        self.drive_out_start    = None      # 전진 시작 시각

        # ------------------- 상태 관리 --------------------------
        # 상태 A: 'ALIGN' (좌/우만) → align_duration_sec 초 유지 후
        # 상태 B: 'FRONT' (front만)
        self.state = 'ALIGN'
        self.start_t = None

        self.get_logger().info(
            f"[{self.state}] 시작. 좌/우 라이다로 가운데 정렬 {self.align_duration_sec:.1f}s 유지. "
            f"이 동안 front는 구독하지 않습니다.")
        # 제어 타이머
        self.timer = self.create_timer(self.dt, self.control_loop)

    # ===================== 콜백: 거리 수신 =====================
    def cb_sos(self, msg: Empty):         # SOS 신호를 받으면 주행 시작
        if self.started:                  # 이미 시작했으면 추가 SOS 무시
            return
        self.started = True
        self.state = 'ALIGN'              # 시작 시 항상 ALIGN로 복귀
        self.start_t = self.clock.now()   # 여기서 20초 타이머 리셋
        self.get_logger().info("[START] SOS 신호 수신 → 주행 시작")

    def cb_imu(self, msg: Imu):
        q = msg.orientation
        self.current_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)  # 라디안 [-pi, pi]

    def cb_left(self, msg: Float32): # ALIGN 상태에서만 의미 있음 (FRONT 상태가 되면 구독 자체를 해제)
        self.left_dist = self._sanitize(msg.data)

    def cb_right(self, msg: Float32): # ALIGN 상태에서만 의미 있음 (FRONT 상태가 되면 구독 자체를 해제)
        self.right_dist = self._sanitize(msg.data)

    def cb_front(self, msg: Float32): # FRONT 상태에서만 의미 있음 (ALIGN 상태에서는 front 구독을 만들지 않음)
        self.front_dist = self._sanitize(msg.data)

    # ===================== 제어 루프 =====================
    def control_loop(self):
        now = self.clock.now()
        if not self.started:
            return
        # ----- 상태 전환: ALIGN → FRONT -----
        if self.state == 'ALIGN' and (now - self.start_t).nanoseconds / 1e9 >= self.align_duration_sec:
            self._switch_to_front_only()

        # FRONT에서 한 번만 트리거: front ≤ 임계 → 회전 시작(IMU)
        if self.state == 'FRONT' and (not self.turned_once):
            if (self.front_dist is not None) and (self.front_dist <= self.FRONT_TURN_TRIG):
                self._start_turn_right()

        # DONE 상태면 더 이상 아무 것도 하지 않음(퍼블리시 X)
        if self.state == 'DONE':
            return
        
        # ----- 상태별 제어 -----
        if self.state == 'ALIGN':
            twist = self._compute_twist_align()
        elif self.state == 'TURN_RIGHT':              # 회전 상태 분기
            twist = self._compute_twist_turn_right()
        elif self.state == 'DRIVE_OUT':               # 회전 후 5초 전진
            twist = self._compute_twist_drive_out(now)
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
            ang = -self.YAW_STRONG                  # 시계 방향으로 회전
            self.get_logger().info("왼쪽에 많이 붙음 → 오른쪽으로 많이 회전")
        elif R is not None and R <= self.THR_VERY_NEAR:
            ang = +self.YAW_STRONG                  # 반시계 방향으로 회전
            self.get_logger().info("오른쪽에 많이 붙음 → 왼쪽으로 많이 회전")
        elif L is not None and L <= self.THR_NEAR:
            ang = -self.YAW_GENTLE
            self.get_logger().info("왼쪽에 붙음 → 오른쪽으로 약하게 회전")
        elif R is not None and R <= self.THR_NEAR:
            ang = +self.YAW_GENTLE
            self.get_logger().info("오른쪽에 붙음 → 왼쪽으로 약하게 회전")
        else:
            ang = 0.0       # 중앙에 가까움 → 직진

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
    
    # ------------------ FRONT 다음 단계: 회전 후 5초 전진 ---------------------
    def _compute_twist_drive_out(self, now) -> Twist:
        """
        회전 완료 후 "drive_out_sec"초 동안 직진 → 그 다음 DONE 상태로 전환하여 종료
        """
        tw = Twist()
        # 안전: 시작 시각이 없으면 바로 종료
        if self.drive_out_start is None:
            self.state = 'DONE'
            self.get_logger().info("[DRIVE_OUT] 시작 시각 없음 → DONE")
            return tw  # 정지(0)

        elapsed = (now - self.drive_out_start).nanoseconds / 1e9
        if elapsed >= self.drive_out_sec:
            # 종료: 이번 틱에는 정지 명령 1회 발행, 다음 루프부터는 control_loop()에서 return
            self.state = 'DONE'
            self.get_logger().info("[DRIVE_OUT] 5초 경과 → DONE(주행 종료)")
            return Twist()  # 정지

        # 5초 동안은 직진 유지
        tw.linear.x = self.LIN_SPEED
        tw.angular.z = 0.0
        return tw

    # ----------------------- IMU 기반 90도 우회전 제어 ----------------------
    def _start_turn_right(self):
        if self.current_yaw is None:
            self.get_logger().warn("IMU yaw 없음 → 회전 대기")
            return

        self.state = 'TURN_RIGHT'
        start_yaw = self.current_yaw
        self.turn_target_yaw = normalize_angle(start_yaw - math.radians(90.0))  # 목표 = 현재 - 90°
        self.get_logger().info(f"TURN 시작: start={math.degrees(start_yaw):.1f}°, \n"
                            f"target={math.degrees(self.turn_target_yaw):.1f}°")

    #------------------------- 상태 전환 유틸 -------------------------------
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
            self.sub_front = self.create_subscription(Float32, '/perception/front_distance', self.cb_front, qos_profile_sensor_data)

        self.get_logger().info("[FRONT]로 전환. 이제 front만 구독하고, 좌/우는 더 이상 받지 않습니다.")

    # --------------------------- 회전중 동작 ---------------------------
    def _compute_twist_turn_right(self) -> Twist:
        tw = Twist()                    # 기본 0 (전진 X)
        if self.current_yaw is None or self.turn_target_yaw is None:
            return tw  # 안전상 정지

        # 현재 yaw와 목표 yaw 차이
        err = shortest_angular_distance(self.current_yaw, self.turn_target_yaw)
        err_deg = abs(math.degrees(err))

        # 1) 허용오차 이내면 종료
        if err_deg <= self.DEG_TOL:
            self._finish_turn_right()
            return Twist()  # 정지 후 FRONT/DRIVE_OUT 등

        # 2) 오버슈트(목표를 지나침)인데 아직 허용오차 밖이면 즉시 종료
        if err > 0 and err_deg > self.DEG_TOL:
            self._finish_turn_right()
            return Twist()

        tw.angular.z = -abs(self.TURN_SPEED)  # 우회전(음수)
        return tw
    
    # --------------------- 회전 완료 후 FRONT 복귀 ------------------------
    def _finish_turn_right(self):
        self.turned_once = True
        self.state = 'DRIVE_OUT'
        self.drive_out_start = self.clock.now()
        self.turn_target_yaw = None
        self.get_logger().info("[TURN_RIGHT] 완료 → DRIVE_OUT(5초 전진)")

    # ------------------------ 헬퍼 ----------------------------
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