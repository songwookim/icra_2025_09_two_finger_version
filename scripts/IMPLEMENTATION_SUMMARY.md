# Variable Stiffness K Calculation - Implementation Summary

## 개요 (Overview)

본 구현은 EMG 신호, 힘 센서 데이터, end-effector 위치 정보를 이용하여 가변 강성(variable stiffness) K를 계산하는 시스템입니다.

## 제공 파일 (Files Provided)

### 1. calculate_stiffness.py
주요 강성 계산 스크립트입니다.

**기능:**
- CSV 데이터 로딩 및 파싱
- EMG 신호 처리 (정류, 필터링, 정규화)
- 힘 센서 데이터 분석
- End-effector 위치/속도 계산
- 가변 강성 K 계산
- 결과 저장 및 시각화

**사용법:**
```bash
python3 scripts/calculate_stiffness.py <CSV_파일> [옵션]
```

**명령행 옵션:**
- `--output, -o`: 출력 CSV 경로
- `--plot, -p`: 그래프 저장 경로
- `--k-min`: 최소 강성 (기본: 10.0)
- `--k-max`: 최대 강성 (기본: 1000.0)
- `--emg-weight`: EMG 가중치 (기본: 0.3)
- `--force-weight`: 힘 가중치 (기본: 0.4)
- `--velocity-weight`: 속도 가중치 (기본: 0.3)

### 2. generate_sample_data.py
테스트용 샘플 데이터 생성기입니다.

**사용법:**
```bash
python3 scripts/generate_sample_data.py --output <경로> --duration <초> --rate <Hz>
```

### 3. example_stiffness_usage.py
프로그래밍 방식 사용 예제 모음입니다.

**포함된 예제:**
- 기본 사용법
- 사용자 정의 매개변수
- 배치 처리
- 중간 데이터 접근

**실행:**
```bash
python3 scripts/example_stiffness_usage.py
```

### 4. README_STIFFNESS.md
상세 문서 (한국어/영어 혼합)

### 5. QUICKSTART.md
빠른 시작 가이드

## 알고리즘 상세 (Algorithm Details)

### 입력 데이터

CSV 파일은 `data_logger_node`에서 생성되며 다음을 포함합니다:

1. **EMG 신호** (`emg_ch1` ~ `emg_ch8`)
   - 8채널 EMG 데이터
   - 샘플링: 100Hz (기본)

2. **힘 센서 데이터** (`s2_fx`, `s2_fy`, `s2_fz`, etc.)
   - 센서 s2, s3의 6축 힘/토크
   - 단위: N, N·m

3. **End-Effector 위치** (`ee_px`, `ee_py`, `ee_pz`)
   - 3D 위치 좌표
   - 단위: m

### 처리 단계

#### 1. EMG 처리
```python
# 정류 (Rectification)
emg_rect = abs(emg_raw)

# 저역통과 필터 (Low-pass filtering)
# 4차 Butterworth, 차단 주파수 10Hz
emg_filtered = butterworth_filter(emg_rect, cutoff=10Hz)

# 정규화 (Normalization)
emg_normalized = emg_filtered / max(emg_filtered)

# 근육 활성도
activation = mean(emg_normalized[ch1:ch8])
```

#### 2. 힘 처리
```python
# 힘 크기 계산
force_mag = sqrt(fx² + fy² + fz²)

# 정규화
force_normalized = force_mag / max(force_mag)
```

#### 3. 속도 처리
```python
# 변위 계산
displacement = ||position - initial_position||

# 속도 추정
velocity = d(displacement)/dt

# 평활화 및 정규화
velocity_smoothed = moving_average(velocity, window=10)
velocity_normalized = velocity_smoothed / max(velocity_smoothed)
```

#### 4. 강성 계산
```python
# 각 컴포넌트 계산
k_emg = activation  # 높은 활성도 → 높은 강성
k_force = force_normalized  # 높은 힘 → 높은 강성
k_velocity = 1 - velocity_normalized  # 높은 속도 → 낮은 강성

# 가중 조합
k_normalized = (w_emg * k_emg + 
                w_force * k_force + 
                w_velocity * k_velocity)

# 스케일링
K = K_min + k_normalized * (K_max - K_min)
```

### 출력 데이터

**CSV 파일:**
- `time`: 시간 (s)
- `stiffness_K`: 계산된 강성
- `muscle_activation`: 근육 활성도 (0~1)
- `force_magnitude`: 힘 크기 (N)
- `velocity`: 속도 (m/s)
- `displacement`: 변위 (m)
- `emg_ch1` ~ `emg_ch8`: 원본 EMG

**시각화 그래프:**
5개 subplot:
1. 강성 K
2. 근육 활성도
3. 힘 크기
4. 속도
5. 변위

## 이론적 배경 (Theoretical Background)

### Assist-As-Needed 전략

본 구현은 다음 원리를 따릅니다:

1. **사용자 참여 기반 지원**
   - EMG 활성도가 높으면 → 사용자가 적극적 → 더 많은 지원
   - 재활 및 스킬 학습에 적합

2. **힘 기반 적응**
   - 외부 부하가 크면 → 더 높은 강성으로 지원
   - 안정성 및 안전성 향상

3. **속도 기반 조절**
   - 빠른 움직임 → 낮은 강성 → 자연스러운 동작 허용
   - 느린 움직임 → 높은 강성 → 정밀 제어 지원

### 참고 논문

1. **Learning Target-Directed Skill and Variable Impedance Control From Interactive Demonstrations**
   - 시연 기반 임피던스 학습
   - 작업 특성에 따른 강성 조절

2. **Tele-Impedance control of a virtual avatar based on EMG and M-IMU sensors**
   - EMG 기반 임피던스 추정
   - 근육 활성도와 강성의 관계

3. **Patient's Healthy-Limb Motion Characteristic-Based Assist-As-Needed Control Strategy** (Section 4.2.2)
   - 필요 시에만 지원하는 제어 전략
   - 사용자 의도 존중 및 능동적 참여 유도

## 사용 예시 (Usage Examples)

### 예시 1: 기본 사용
```bash
python3 scripts/calculate_stiffness.py \
  /home/songwoo/ros2_ws/icra2025_twofinger/outputs/logs/20250929/trial_1.csv
```

출력:
- `trial_1_stiffness.csv`
- `trial_1_stiffness_plot.png`

### 예시 2: 재활 응용
```bash
python3 scripts/calculate_stiffness.py trial_1.csv \
  --emg-weight 0.6 \      # 사용자 의도 강조
  --force-weight 0.2 \
  --velocity-weight 0.2 \
  --k-min 20 \            # 안전 강성
  --k-max 500             # 편안한 최대값
```

### 예시 3: Python 코드 통합
```python
from calculate_stiffness import StiffnessCalculator

# 초기화
calc = StiffnessCalculator('trial_1.csv')
calc.load_data()

# 계산
stiffness = calc.calculate_stiffness(
    emg_weight=0.4,
    force_weight=0.4,
    velocity_weight=0.2,
    k_min=50.0,
    k_max=1500.0
)

# 결과 분석
import numpy as np
print(f"평균 강성: {np.mean(stiffness):.2f}")
print(f"최대 강성: {np.max(stiffness):.2f}")
print(f"최소 강성: {np.min(stiffness):.2f}")

# 특정 이벤트 분석
high_activation_idx = np.where(calc.muscle_activation > 0.7)[0]
if len(high_activation_idx) > 0:
    print(f"높은 활성도 시점: {len(high_activation_idx)}개")
    print(f"해당 시점 평균 강성: {np.mean(stiffness[high_activation_idx]):.2f}")
```

## 매개변수 튜닝 (Parameter Tuning)

### 가중치 선택 가이드

| 응용 분야 | EMG | Force | Velocity | 설명 |
|----------|-----|-------|----------|------|
| 재활 | 0.6 | 0.2 | 0.2 | 사용자 의도 우선 |
| 숙련 작업 | 0.3 | 0.5 | 0.2 | 힘 피드백 중시 |
| 일반 보조 | 0.3 | 0.4 | 0.3 | 균형잡힌 제어 |
| 부드러운 제어 | 0.2 | 0.3 | 0.5 | 속도 적응형 |

### 강성 범위 선택

| 응용 | K_min | K_max | 이유 |
|-----|-------|-------|------|
| 재활 | 20-50 | 300-500 | 안전 및 편안함 |
| 정밀 작업 | 100-200 | 1500-2000 | 높은 정밀도 |
| 일반 보조 | 10-20 | 800-1200 | 유연성 |

## 의존성 (Dependencies)

### Python 패키지
```bash
pip install numpy scipy matplotlib
```

- `numpy`: 수치 계산
- `scipy`: 신호 처리 (필터링)
- `matplotlib`: 시각화

### ROS 2 (선택사항)
- `data_logger_node`에서 CSV 생성 시 필요
- 스크립트 자체는 독립 실행 가능

## 검증 (Validation)

### 단위 테스트
```bash
# 샘플 데이터 생성 및 처리
python3 scripts/generate_sample_data.py --output /tmp/test.csv
python3 scripts/calculate_stiffness.py /tmp/test.csv
```

### 예제 실행
```bash
# 모든 예제 실행
python3 scripts/example_stiffness_usage.py
```

예상 출력:
- 4개 예제 모두 성공
- 여러 CSV 및 PNG 파일 생성
- 통계 정보 출력

## 문제 해결 (Troubleshooting)

### CSV 파싱 오류
- 파일 형식이 `data_logger_node` 출력과 일치하는지 확인
- 필수 열: `t_sec`, `t_nanosec`, `emg_ch1-8`, `s2_fx-fz`, `ee_px-pz`

### EMG 데이터 누락
- `data_logger_node`의 `emg_log_every_n` 확인
- EMG 노드가 올바르게 퍼블리시하는지 확인

### 성능 문제
- 큰 파일의 경우 처리 시간 증가
- 필터 윈도우 크기 조정으로 최적화 가능

## 향후 개선 사항 (Future Enhancements)

1. **실시간 처리**
   - ROS 2 노드로 변환하여 실시간 강성 계산
   
2. **머신러닝 기반 최적화**
   - 사용자별 최적 가중치 학습
   
3. **다중 센서 융합**
   - IMU 데이터 통합
   - 추가 생체신호 활용

4. **적응형 필터**
   - 상황에 따라 필터 매개변수 자동 조정

## 라이선스 (License)

MIT License

## 문의 (Contact)

문제가 발생하거나 제안 사항이 있으면 GitHub Issues를 이용해 주세요.

---

**작성일**: 2024
**버전**: 1.0.0
