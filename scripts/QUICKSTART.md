# Quick Start Guide: Stiffness Calculation

## 빠른 시작 (Quick Start)

### 1. CSV 파일로부터 강성 계산

```bash
# 기본 사용법
python3 scripts/calculate_stiffness.py /path/to/your/trial_data.csv

# 예시: 특정 로그 파일 처리
python3 scripts/calculate_stiffness.py /home/songwoo/ros2_ws/icra2025_twofinger/outputs/logs/20250929/trial_1.csv
```

출력:
- `trial_1_stiffness.csv`: 계산된 강성 K 및 중간 결과
- `trial_1_stiffness_plot.png`: 시각화 그래프

### 2. 매개변수 조정

```bash
# 강성 범위와 가중치 조정
python3 scripts/calculate_stiffness.py trial_1.csv \
  --k-min 50 \
  --k-max 2000 \
  --emg-weight 0.5 \
  --force-weight 0.3 \
  --velocity-weight 0.2
```

### 3. Python 코드에서 사용

```python
from calculate_stiffness import StiffnessCalculator

# 계산기 생성 및 데이터 로드
calc = StiffnessCalculator('trial_1.csv')
calc.load_data()

# 강성 계산
stiffness = calc.calculate_stiffness(
    emg_weight=0.3,
    force_weight=0.4,
    velocity_weight=0.3,
    k_min=10.0,
    k_max=1000.0
)

# 결과 저장
calc.save_results('output_stiffness.csv')
calc.plot_results('output_plot.png')

# 중간 결과 접근
print(f"Muscle activation: {calc.muscle_activation}")
print(f"Force magnitude: {calc.force_mag}")
print(f"Velocity: {calc.velocity}")
```

## 출력 데이터

### CSV 열 (Columns)

| 열 이름 | 설명 |
|---------|------|
| `time` | 시간 (초) |
| `stiffness_K` | 계산된 가변 강성 |
| `muscle_activation` | 근육 활성도 (0~1) |
| `force_magnitude` | 힘 크기 (N) |
| `velocity` | 속도 (m/s) |
| `displacement` | 변위 (m) |
| `emg_ch1` ~ `emg_ch8` | EMG 채널 데이터 |

## 알고리즘 개요

강성 K는 다음 공식으로 계산됩니다:

```
K = K_min + normalized_value × (K_max - K_min)

where:
  normalized_value = w_emg × activation + w_force × force + w_vel × (1 - velocity)
  
  activation = mean(EMG_filtered[ch1..ch8])  // 0~1 정규화
  force = ||F_sensor||                        // 0~1 정규화
  velocity = ||dX/dt||                        // 0~1 정규화, 반전
```

### 설계 원리 (Assist-As-Needed)

- **높은 EMG 활성도** → 높은 강성: 사용자가 적극적으로 참여할 때 더 많은 지원
- **높은 힘** → 높은 강성: 외부 부하가 클 때 더 많은 지원  
- **높은 속도** → 낮은 강성: 자연스러운 빠른 움직임 허용

## 매개변수 튜닝 가이드

### 재활 응용 (Rehabilitation)
```bash
python3 scripts/calculate_stiffness.py data.csv \
  --emg-weight 0.6 \    # 사용자 의도 강조
  --force-weight 0.2 \
  --velocity-weight 0.2 \
  --k-min 20 \          # 안전을 위한 최소 강성
  --k-max 500           # 편안함을 위한 최대 강성
```

### 숙련 작업 지원 (Skilled Task Assistance)
```bash
python3 scripts/calculate_stiffness.py data.csv \
  --emg-weight 0.3 \
  --force-weight 0.5 \  # 힘 피드백 강조
  --velocity-weight 0.2 \
  --k-min 10 \
  --k-max 1500
```

### 부드러운 제어 (Smooth Control)
```bash
python3 scripts/calculate_stiffness.py data.csv \
  --emg-weight 0.25 \
  --force-weight 0.35 \
  --velocity-weight 0.4  # 속도 변화에 민감
```

## 문제 해결

### "No module named 'numpy'" 오류
```bash
pip install numpy scipy matplotlib
```

### EMG 데이터가 없음
CSV 파일에 `emg_ch1` ~ `emg_ch8` 열이 있는지 확인하세요.

### 위치 데이터가 없음  
CSV 파일에 `ee_px`, `ee_py`, `ee_pz` 열이 있는지 확인하세요.

### 그래프가 표시되지 않음
```bash
export MPLBACKEND=Agg  # 헤드리스 환경에서
```

## 추가 예제

전체 예제는 다음을 실행하세요:
```bash
python3 scripts/example_stiffness_usage.py
```

이 스크립트는 다음을 포함합니다:
- 기본 사용법
- 사용자 정의 매개변수
- 배치 처리
- 중간 데이터 접근

## 참고 논문

1. Learning Target-Directed Skill and Variable Impedance Control From Interactive Demonstrations for Robot-Assisted Soft Tissue Puncture Task
2. Tele-Impedance control of a virtual avatar based on EMG and M-IMU sensors: a proof-of-concept
3. Patient's Healthy-Limb Motion Characteristic-Based Assist-As-Needed Control Strategy for Upper-Limb Rehabilitation Robots (Section 4.2.2)

## 지원

더 자세한 정보는 `scripts/README_STIFFNESS.md`를 참조하세요.
