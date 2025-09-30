# Variable Stiffness Calculation Scripts

이 디렉토리에는 EMG, 힘 센서, end-effector 위치 데이터로부터 가변 강성(stiffness) K를 계산하는 스크립트가 포함되어 있습니다.

## 개요

`calculate_stiffness.py` 스크립트는 `data_logger_node`에서 기록된 CSV 파일을 읽어 다음 정보를 기반으로 가변 강성 K를 계산합니다:

- **EMG 신호**: 근육 활성도를 추정
- **힘 센서 데이터**: 작업에 필요한 지원 정도를 평가
- **End-effector 위치/속도**: 움직임 특성을 분석

## 알고리즘

이 구현은 다음 논문들의 개념을 기반으로 합니다:
- Learning Target-Directed Skill and Variable Impedance Control From Interactive Demonstrations
- Tele-Impedance control of a virtual avatar based on EMG and M-IMU sensors
- Patient's Healthy-Limb Motion Characteristic-Based Assist-As-Needed Control Strategy (Section 4.2.2)

### 강성 계산 원리

강성 K는 다음 세 가지 요소의 가중 조합으로 계산됩니다:

1. **EMG 기반 근육 활성도** (기본 가중치: 0.3)
   - EMG 신호를 정류(rectification) 및 저역통과 필터링
   - 8채널의 평균 활성도 계산
   - 높은 근육 활성도 → 높은 강성 (사용자가 적극적으로 참여)

2. **힘 크기** (기본 가중치: 0.4)
   - 힘 센서(s2)의 3축 힘 크기 계산
   - 높은 힘 → 높은 강성 (작업에 더 많은 지원 필요)

3. **속도** (기본 가중치: 0.3)
   - End-effector 위치로부터 속도 계산
   - 높은 속도 → 낮은 강성 (자연스러운 움직임 허용)

최종 강성: `K = K_min + normalized_value * (K_max - K_min)`

여기서:
- `K_min`: 최소 강성 (기본값: 10.0)
- `K_max`: 최대 강성 (기본값: 1000.0)
- `normalized_value`: 0~1 범위로 정규화된 조합 값

## 사용법

### 기본 사용

```bash
python3 scripts/calculate_stiffness.py <CSV_파일_경로>
```

예시:
```bash
python3 scripts/calculate_stiffness.py /home/songwoo/ros2_ws/icra2025_twofinger/outputs/logs/20250929/trial_1.csv
```

### 옵션

```bash
python3 scripts/calculate_stiffness.py <CSV_파일> [옵션]

옵션:
  --output, -o PATH          출력 CSV 파일 경로 (기본: <입력>_stiffness.csv)
  --plot, -p PATH            그래프 저장 경로 (기본: <입력>_stiffness_plot.png)
  --k-min VALUE              최소 강성 값 (기본: 10.0)
  --k-max VALUE              최대 강성 값 (기본: 1000.0)
  --emg-weight VALUE         EMG 가중치 (기본: 0.3)
  --force-weight VALUE       힘 가중치 (기본: 0.4)
  --velocity-weight VALUE    속도 가중치 (기본: 0.3)
```

### 고급 예시

가중치 조정 및 강성 범위 변경:
```bash
python3 scripts/calculate_stiffness.py trial_1.csv \
  --k-min 50 \
  --k-max 2000 \
  --emg-weight 0.5 \
  --force-weight 0.3 \
  --velocity-weight 0.2 \
  --output results/trial_1_stiffness.csv \
  --plot results/trial_1_plot.png
```

## 출력

### CSV 파일

출력 CSV 파일에는 다음 열이 포함됩니다:

- `time`: 시간 (초)
- `stiffness_K`: 계산된 가변 강성
- `muscle_activation`: 근육 활성도 (0~1 정규화)
- `force_magnitude`: 힘 크기 (N)
- `velocity`: 속도 (m/s)
- `displacement`: 초기 위치로부터의 변위 (m)
- `emg_ch1` ~ `emg_ch8`: 원본 EMG 채널 데이터

### 시각화

스크립트는 다음을 포함하는 5개의 subplot으로 구성된 그래프를 생성합니다:

1. **강성 K**: 시간에 따른 가변 강성
2. **근육 활성도**: EMG 기반 활성도
3. **힘 크기**: 센서에서 측정된 힘
4. **속도**: End-effector 속도
5. **변위**: 초기 위치로부터의 변위

## 샘플 데이터 생성

테스트용 샘플 데이터를 생성하려면:

```bash
python3 scripts/generate_sample_data.py --output /tmp/sample_trial.csv --duration 10 --rate 100
python3 scripts/calculate_stiffness.py /tmp/sample_trial.csv
```

## 의존성

- Python 3.x
- numpy
- scipy
- matplotlib

설치:
```bash
pip install numpy scipy matplotlib
```

## 입력 CSV 형식

입력 CSV는 `data_logger_node`에서 생성된 형식이어야 합니다:

필수 열:
- `t_sec`, `t_nanosec`: 타임스탬프
- `s2_fx`, `s2_fy`, `s2_fz`: 센서 2 힘 데이터
- `emg_ch1` ~ `emg_ch8`: EMG 채널
- `ee_px`, `ee_py`, `ee_pz`: End-effector 위치

## 알고리즘 세부 사항

### EMG 처리
1. **정류(Rectification)**: 절대값 취함
2. **저역통과 필터**: 4차 Butterworth 필터 (기본 차단 주파수: 10 Hz)
3. **정규화**: 각 채널을 0~1 범위로 정규화
4. **활성도**: 8채널의 평균값 사용

### 힘 처리
1. **유클리드 노름**: 3축 힘의 크기 계산 (`sqrt(fx² + fy² + fz²)`)
2. **정규화**: 최대값으로 나누어 0~1 범위로 조정

### 속도 처리
1. **변위 계산**: 초기 위치로부터의 거리
2. **속도 추정**: 시간에 대한 변위의 기울기 (`numpy.gradient`)
3. **평활화**: 이동 평균 필터 적용
4. **정규화**: 최대값으로 나누어 0~1 범위로 조정

## 매개변수 튜닝 가이드

- **EMG 가중치 증가**: 사용자의 의도를 더 강하게 반영
- **힘 가중치 증가**: 외부 부하에 더 민감하게 반응
- **속도 가중치 증가**: 움직임 속도에 더 적응적으로 대응
- **K_min/K_max 조정**: 적용 가능한 강성 범위 제한

## 문제 해결

### EMG 데이터가 비어있음
- CSV 파일에 `emg_ch1` ~ `emg_ch8` 열이 있는지 확인
- `data_logger_node`의 `emg_log_every_n` 매개변수 확인

### 위치 데이터가 비어있음
- `ee_pose_topic` 또는 `ee_pose_odometry_topic`이 올바르게 설정되었는지 확인
- CSV에 `ee_px`, `ee_py`, `ee_pz` 열이 있는지 확인

### 그래프가 생성되지 않음
- matplotlib 백엔드 문제일 수 있음
- `export MPLBACKEND=Agg` 설정 시도

## 라이선스

MIT License
