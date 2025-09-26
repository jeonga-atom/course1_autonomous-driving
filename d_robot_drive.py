#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Twist를 발행하여 로봇이 구동되는 코드

import time, math, serial, struct, threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from dynamixel_sdk import PortHandler, PacketHandler


# ------------------------------
# In-Wheel Motor Controller (원본과 동일 포맷/포트/ID)
# ------------------------------
class InWheelMotorController:
    """
    포트/ID 매핑:
      FR: port=/dev/ttyACM1, id=3
      FL: port=/dev/ttyACM2, id=4
      RR: port=/dev/ttyACM3, id=2
      RL: port=/dev/ttyACM4, id=1
    """
    def __init__(self):
        self.port_list = ["/dev/ttyACM1","/dev/ttyACM2","/dev/ttyACM3","/dev/ttyACM4"]  # [FR,FL,RR,RL]
        self.velocity_ids = (3,4,2,1)
        self.serial_connections = [self.connect_serial(p) for p in self.port_list]

        # 스레드용 공유 상태
        self._last_wheel_speeds = [0,0,0,0]
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # TX 스레드: 마지막 명령을 50Hz로 지속 전송
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._tx_thread.start()

    def connect_serial(self, port, baudrate=115200, timeout=1):
        return serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    @staticmethod
    def decimal_to_hex_bytes(decimal):
        # 16비트 signed -> 상위/하위 바이트
        hex_string = format(struct.unpack('>H', struct.pack('>h', int(decimal)))[0],'x').zfill(4)
        return int(hex_string[:2],16), int(hex_string[2:],16)

    @staticmethod
    def calculate_crc(data):
        CRC8_MAXIM_table = (
            0x00,0x5e,0xbc,0xe2,0x61,0x3f,0xdd,0x83,0xc2,0x9c,0x7e,0x20,0xa3,0xfd,0x1f,0x41,
            0x9d,0xc3,0x21,0x7f,0xfc,0xa2,0x40,0x1e,0x5f,0x01,0xe3,0xbd,0x3e,0x60,0x82,0xdc,
            0x23,0x7d,0x9f,0xc1,0x42,0x1c,0xfe,0xa0,0xe1,0xbf,0x5d,0x03,0x80,0xde,0x3c,0x62,
            0xbe,0xe0,0x02,0x5c,0xdf,0x81,0x63,0x3d,0x7c,0x22,0xc0,0x9e,0x1d,0x43,0xa1,0xff,
            0x46,0x18,0xfa,0xa4,0x27,0x79,0x9b,0xc5,0x84,0xda,0x38,0x66,0xe5,0xbb,0x59,0x07,
            0xdb,0x85,0x67,0x39,0xba,0xe4,0x06,0x58,0x19,0x47,0xa5,0xfb,0x78,0x26,0xc4,0x9a,
            0x65,0x3b,0xd9,0x87,0x04,0x5a,0xb8,0xe6,0xa7,0xf9,0x1b,0x45,0xc6,0x98,0x7a,0x24,
            0xf8,0xa6,0x44,0x1a,0x99,0xc7,0x25,0x7b,0x3a,0x64,0x86,0xd8,0x5b,0x05,0xe7,0xb9,
            0x8c,0xd2,0x30,0x6e,0xed,0xb3,0x51,0x0f,0x4e,0x10,0xf2,0xac,0x2f,0x71,0x93,0xcd,
            0x11,0x4f,0xad,0xf3,0x70,0x2e,0xcc,0x92,0xd3,0x8d,0x6f,0x31,0xb2,0xec,0x0e,0x50,
            0xaf,0xf1,0x13,0x4d,0xce,0x90,0x72,0x2c,0x6d,0x33,0xd1,0x8f,0x0c,0x52,0xb0,0xee,
            0x32,0x6c,0x8e,0xd0,0x53,0x0d,0xef,0xb1,0xf0,0xae,0x4c,0x12,0x91,0xcf,0x2d,0x73,
            0xca,0x94,0x76,0x28,0xab,0xf5,0x17,0x49,0x08,0x56,0xb4,0xea,0x69,0x37,0xd5,0x8b,
            0x57,0x09,0xeb,0xb5,0x36,0x68,0x8a,0xd4,0x95,0xcb,0x29,0x77,0xf4,0xaa,0x48,0x16,
            0xe9,0xb7,0x55,0x0b,0x88,0xd6,0x34,0x6a,0x2b,0x75,0x97,0xc9,0x4a,0x14,0xf6,0xa8,
            0x74,0x2a,0xc8,0x96,0x15,0x4b,0xa9,0xf7,0xb6,0xe8,0x0a,0x54,0xd7,0x89,0x6b,0x35
        )
        crc = 0x00
        for byte in data:
            index = (crc ^ int(byte)) & 0xFF
            crc = CRC8_MAXIM_table[index]
        return crc

    def set_velocity(self, ser, ID, speed):
        # speed: signed int16
        hi, lo = self.decimal_to_hex_bytes(speed)
        data_temp = bytes([ID & 0xFF, 0x64, hi, lo, 0, 0, 0, 0, 0])
        crc = self.calculate_crc(data_temp)
        ser.write(data_temp + bytes([crc]))

    def set_velocity_individual(self, wheel_speeds):
        # [FR, FL, RR, RL]
        with self._lock:
            self._last_wheel_speeds = list(map(int, wheel_speeds))

    def _tx_loop(self):
        rate_hz = 50.0
        dt = 1.0 / rate_hz
        while not self._stop.is_set():
            with self._lock:
                speeds = list(self._last_wheel_speeds)
            for ser, motor_id, speed in zip(self.serial_connections, self.velocity_ids, speeds):
                self.set_velocity(ser, motor_id, int(speed))
            time.sleep(dt)

    def shutdown(self):
        self._stop.set()
        try:
            self._tx_thread.join(timeout=1.0)
        except:
            pass
        # 안전 정지 후 포트 닫기
        for ser, motor_id in zip(self.serial_connections, self.velocity_ids):
            try:
                self.set_velocity(ser, motor_id, 0)
            except:
                pass
            try:
                ser.close()
            except:
                pass


# ------------------------------
# Dynamixel 초기화(토크 ON + 정중앙 2048, 이후 미사용)
# ------------------------------
class DynamixelInitializer:
    def __init__(self, port_name='/dev/ttyACM0', baud_rate=1000000, ids=(0,1), init_pos=2048):
        self.port = PortHandler(port_name)
        self.packet = PacketHandler(2.0)
        if not (self.port.openPort() and self.port.setBaudRate(baud_rate)):
            raise RuntimeError("Dynamixel 포트 연결 실패")
        for _id in ids:
            # Torque ON & 중앙 위치
            self.packet.write1ByteTxRx(self.port, _id, 64, 1)     # Torque On
            self.packet.write1ByteTxRx(self.port, _id, 11, 3)     # Position Mode
            self.packet.write4ByteTxRx(self.port, _id, 116, init_pos)

    def shutdown(self):
        # 요청: 다이나믹셀은 토크 ON 상태로 두고 사용하지 않음 -> 포트만 닫음
        try:
            self.port.closePort()
        except:
            pass


# ------------------------------
# ROS2 Node: /cmd_vel -> wheel RPM
# ------------------------------
class CmdVelToRPMNode(Node):
    def __init__(self):
        super().__init__('cmdvel_to_rpm_dd')
        # 로봇 기하 파라미터 (원본과 동일)
        self.TRACK = 0.347        # 좌우 바퀴 간격 [m]
        self.WHEEL_RADIUS = 0.05035 # 바퀴 반지름 [m]

        self.declare_parameter('cmd_vel_topic', '/cmd_vel') # 구독 토픽
        cmd_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value

        # 변환 상수
        self.RPM_PER_MPS = 60.0 / (2.0 * math.pi * self.WHEEL_RADIUS)

        # 모터 컨트롤러
        self.inwheel = InWheelMotorController()

        # 다이나믹셀: 토크 ON + 2048, 이후 미사용
        try:
            self.dxl = DynamixelInitializer(port_name='/dev/ttyACM0', baud_rate=1000000, ids=(0,1), init_pos=2048)
        except Exception as e:
            self.get_logger().warn(f"Dynamixel 초기화 실패(무시): {e}")
            self.dxl = None

        # /cmd_vel 구독
        self.sub = self.create_subscription(Twist, cmd_topic, self.cmdvel_cb, 10)

        # 마지막 명령 유지 타이머(워치독)
        self.last_cmd_time = self.get_clock().now()
        self.timeout_sec = 0.5
        self.timer = self.create_timer(0.02, self.safety_watchdog)  # 50Hz

    def cmdvel_cb(self, msg: Twist):
        v = float(msg.linear.x)          # m/s
        omega = float(msg.angular.z)     # rad/s

        # 차동 구동 선속도 계산
        v_r = v + (omega * self.TRACK / 2.0)  # m/s
        v_l = v - (omega * self.TRACK / 2.0)  # m/s

        # m/s -> RPM
        rpm_r = v_r * self.RPM_PER_MPS
        rpm_l = v_l * self.RPM_PER_MPS

        # 기존 하드웨어 방향성 반영:
        # 오른쪽 휠은 부호 반전, 왼쪽은 그대로
        fr = int(-rpm_r)
        rr = int(-rpm_r)
        fl = int(rpm_l)
        rl = int(rpm_l)

        # 전송 (순서: [FR, FL, RR, RL])
        self.inwheel.set_velocity_individual([fr, fl, rr, rl])

        # 타임스탬프 갱신
        self.last_cmd_time = self.get_clock().now()

    def safety_watchdog(self):
        # 일정 시간 /cmd_vel 미수신 시 안전 정지
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9 > self.timeout_sec:
            self.inwheel.set_velocity_individual([0,0,0,0])

    def shutdown(self):
        if self.dxl:
            self.dxl.shutdown()
        self.inwheel.shutdown()


def main():
    rclpy.init()
    node = CmdVelToRPMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
