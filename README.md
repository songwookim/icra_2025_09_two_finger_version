# hri_falcon_robot_bridge 디버그 런치 요약

## 기본 빌드
```bash
colcon build --packages-select hri_falcon_robot_bridge && source install/setup.bash
```

## VS Code Launch (한 줄 설명)
- `0. Current Python File` — 현재 열린 파이썬 파일을 ROS 2 환경에서 실행.
- `1. hand_tracker` — RealSense+MediaPipe 손 추적 노드를 디버그 실행.
- `2. force_sensor` — 힘 센서 노드를 실데이터 모드(`use_mock:=false`)로 실행.
- `3. dclaw` — Dynamixel 손가락 제어 노드를 `safe_mode:=false`로 실행.
- `4. falcon` — libnifalcon C++ 브리지를 gdb로 실행.
- `5. deformity_tracker` — HSV 기반 변형 추적 노드를 GUI 포함으로 실행.
- `6. emg_node` — Myo EMG 8채널 스트리머를 플롯 없이 실행.
- `7. data_logger` — 힘·EMG·EE 포즈 동기화 CSV 로거를 100 Hz로 실행.
- `8. plot_result_csv` — 동기화 CSV를 빠르게 시각화하는 플롯 스크립트 실행.