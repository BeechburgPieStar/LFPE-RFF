"""
IQ信号数据增强模块 - 3增强 (N, P, A)
==========================================
3种有效增强，每种有X个强度级别（可调节）:
  1. gaussian_noise (N): AWGN噪声
  2. phase_noise (P): 相位噪声
  3. pa_nonlinear (A): 功放非线性

配置说明:
  - NUM_LEVELS: 增强级别数量
  - 每种增强的范围可在参数配置区调节
"""

import numpy as np
from typing import List
from dataclasses import dataclass


# ==============================================================================
#                              配置区 (可调节)
# ==============================================================================
NUM_LEVELS = 20  # 增强级别数量 5 10 15 20

# N - AWGN: SNR范围 (dB)，从高到低
AWGN_SNR_MIN = -10.0   # 最强噪声 (severity=NUM_LEVELS)
AWGN_SNR_MAX = 30.0    # 最弱噪声 (severity=1)

# P - Phase Noise: 相位噪声标准差范围 (度)
PHASE_MIN = 2.0        # 最弱 (severity=1)
PHASE_MAX = 40.0       # 最强 (severity=NUM_LEVELS)

# A - PA Nonlinearity: 压缩量范围 (dB)
PA_MIN = 1.0           # 最弱 (severity=1)
PA_MAX = 12.0          # 最强 (severity=NUM_LEVELS)


# ==============================================================================
#                              生成参数字典
# ==============================================================================
def _linspace_dict(min_val, max_val, num_levels, reverse=False):
    """生成线性间隔的参数字典"""
    d = {0: None}
    values = np.linspace(min_val, max_val, num_levels)
    if reverse:
        values = values[::-1]
    for i in range(1, num_levels + 1):
        d[i] = float(values[i - 1])
    return d

# N - AWGN: SNR从高到低 (reverse=True: severity越大，SNR越低)
AWGN_SNR_DB = _linspace_dict(AWGN_SNR_MAX, AWGN_SNR_MIN, NUM_LEVELS)

# P - Phase Noise: 相位噪声从小到大
PHASE_NOISE_STD = _linspace_dict(PHASE_MIN, PHASE_MAX, NUM_LEVELS)

# A - PA Nonlinearity: 压缩量从小到大
PA_COMPRESSION_DB = _linspace_dict(PA_MIN, PA_MAX, NUM_LEVELS)


# ==============================================================================
#                              工具函数
# ==============================================================================
def _safe(x: np.ndarray) -> np.ndarray:
    """转为float32并处理NaN/Inf"""
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _check(x: np.ndarray) -> np.ndarray:
    """检查输入形状为 (2, L)"""
    x = _safe(x)
    assert x.ndim == 2 and x.shape[0] == 2, f"Expected (2, L), got {x.shape}"
    return x


def _iq_to_complex(x: np.ndarray) -> np.ndarray:
    """(2, L) -> complex (L,)"""
    return x[0] + 1j * x[1]


def _complex_to_iq(c: np.ndarray) -> np.ndarray:
    """complex (L,) -> (2, L)"""
    return _safe(np.stack([np.real(c), np.imag(c)], axis=0))


# ==============================================================================
#                              3种基础增强函数
# ==============================================================================

def gaussian_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    """
    加性高斯白噪声 (AWGN)
    SNR越低，噪声越大
    """
    x = _check(x)
    
    sig_power = np.mean(x[0]**2 + x[1]**2)
    if sig_power < 1e-10:
        return x
    
    noise_power = sig_power / (10 ** (snr_db / 10) + 1e-10)
    noise = np.sqrt(noise_power / 2) * np.random.randn(2, x.shape[1])
    
    return _safe(x + noise)


def phase_noise(x: np.ndarray, std_deg: float) -> np.ndarray:
    """
    相位噪声 - 本振相位不稳定
    每个采样点独立的相位抖动
    """
    x = _check(x)
    L = x.shape[1]
    
    I, Q = x[0], x[1]
    amp = np.sqrt(I**2 + Q**2 + 1e-10)
    phase = np.arctan2(Q, I)
    
    # 添加相位噪声
    pn = np.random.randn(L).astype(np.float32) * np.deg2rad(std_deg)
    new_phase = phase + pn
    
    return _safe(np.stack([amp * np.cos(new_phase), amp * np.sin(new_phase)], axis=0))


def pa_nonlinear(x: np.ndarray, compression_db: float) -> np.ndarray:
    """
    功放非线性 - Rapp模型
    compression_db越大，失真越严重
    """
    x = _check(x)
    
    c = _iq_to_complex(x)
    amp = np.abs(c)
    phase = np.angle(c)
    
    amp_max = np.max(amp) + 1e-10
    amp_norm = amp / amp_max
    
    # Rapp模型
    saturation = 10 ** (-compression_db / 20)
    p = 2.0
    
    amp_out = amp_norm / ((1 + (amp_norm / saturation) ** (2 * p)) ** (1 / (2 * p)))
    amp_out = amp_out * amp_max
    
    # AM-PM 失真
    phase_shift = 0.1 * compression_db * (amp_norm ** 2)
    phase_out = phase + np.deg2rad(phase_shift)
    
    c_out = amp_out * np.exp(1j * phase_out)
    
    return _complex_to_iq(c_out)


# ==============================================================================
#                              增强配置和执行器
# ==============================================================================

@dataclass
class AugConfig:
    """增强配置 - 3种增强"""
    awgn_sev: int = 0       # N: 0-NUM_LEVELS
    phase_sev: int = 0      # P: 0-NUM_LEVELS
    pa_sev: int = 0         # A: 0-NUM_LEVELS
    
    @property
    def name(self) -> str:
        """生成配置名称"""
        parts = []
        if self.awgn_sev > 0:
            parts.append(f"N{self.awgn_sev}")
        if self.phase_sev > 0:
            parts.append(f"P{self.phase_sev}")
        if self.pa_sev > 0:
            parts.append(f"A{self.pa_sev}")
        return "_".join(parts) if parts else "Clean"
    
    @property
    def total_severity(self) -> int:
        """总severity"""
        return self.awgn_sev + self.phase_sev + self.pa_sev


class CombinedAugmentor:
    """组合增强器"""
    
    def __init__(self, awgn_sev: int = 0, phase_sev: int = 0, pa_sev: int = 0):
        self.config = AugConfig(awgn_sev, phase_sev, pa_sev)
        
        # 验证参数范围
        for name, val in [("awgn", awgn_sev), ("phase", phase_sev), ("pa", pa_sev)]:
            if not 0 <= val <= NUM_LEVELS:
                raise ValueError(f"{name}_sev must be 0-{NUM_LEVELS}, got {val}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """应用所有配置的增强"""
        out = _check(x.copy())
        
        if self.config.awgn_sev > 0:
            snr = AWGN_SNR_DB[self.config.awgn_sev]
            out = gaussian_noise(out, snr)
        
        if self.config.phase_sev > 0:
            std = PHASE_NOISE_STD[self.config.phase_sev]
            out = phase_noise(out, std)
        
        if self.config.pa_sev > 0:
            comp = PA_COMPRESSION_DB[self.config.pa_sev]
            out = pa_nonlinear(out, comp)
        
        return _safe(out)
    
    @property
    def name(self) -> str:
        return self.config.name


# ==============================================================================
#                              预定义配置
# ==============================================================================

def generate_predefined_configs() -> List[AugConfig]:
    """生成预定义配置列表 - 只使用 N, P, A"""
    configs = []
    
    # 0. Clean
    configs.append(AugConfig(0, 0, 0))
    
    # 1. 单一增强 - 完整所有级别 (3 × NUM_LEVELS)
    for sev in range(1, NUM_LEVELS + 1):
        configs.append(AugConfig(awgn_sev=sev))      # N
        configs.append(AugConfig(phase_sev=sev))     # P
        configs.append(AugConfig(pa_sev=sev))        # A
    
    return configs


PREDEFINED_CONFIGS = generate_predefined_configs()


def get_all_configs() -> List[AugConfig]:
    """获取所有预定义配置"""
    return PREDEFINED_CONFIGS


def create_augmentor(config: AugConfig) -> CombinedAugmentor:
    """从配置创建增强器"""
    return CombinedAugmentor(
        awgn_sev=config.awgn_sev,
        phase_sev=config.phase_sev,
        pa_sev=config.pa_sev
    )


def create_augmentor_by_name(name: str) -> CombinedAugmentor:
    """根据名称创建增强器，如 'N5_D3_P2' """
    if name == "Clean":
        return CombinedAugmentor(0, 0, 0)
    
    awgn_sev = phase_sev = pa_sev = 0
    
    for part in name.split("_"):
        if part.startswith("N"):
            awgn_sev = int(part[1:])
        elif part.startswith("P"):
            phase_sev = int(part[1:])
        elif part.startswith("A"):
            pa_sev = int(part[1:])
    
    return CombinedAugmentor(awgn_sev, phase_sev, pa_sev)


# ==============================================================================
#                              统计函数
# ==============================================================================

def print_statistics():
    """打印统计信息"""
    configs = get_all_configs()
    
    print("=" * 60)
    print("IQ信号数据增强模块 - 3增强 (N, P, A)")
    print("=" * 60)
    print(f"\n增强级别数: {NUM_LEVELS}")
    print(f"\n参数范围:")
    print(f"  N - AWGN:     SNR {AWGN_SNR_DB[1]:.1f}dB ~ {AWGN_SNR_DB[NUM_LEVELS]:.1f}dB")
    print(f"  P - Phase:    std {PHASE_NOISE_STD[1]:.1f}° ~ {PHASE_NOISE_STD[NUM_LEVELS]:.1f}°")
    print(f"  A - PA:       comp {PA_COMPRESSION_DB[1]:.1f}dB ~ {PA_COMPRESSION_DB[NUM_LEVELS]:.1f}dB")
    print(f"\n预定义配置: {len(configs)} 种")
    print(f"  - Clean: 1")
    print(f"  - 单增强: {3 * NUM_LEVELS}")
    print("=" * 60)


def print_param_table():
    """打印参数表格"""
    print("\n参数详细表:")
    print("-" * 70)
    print(f"{'Sev':>3} | {'AWGN(dB)':>10} | {'Phase(°)':>10} | {'PA(dB)':>10}")
    print("-" * 70)
    for i in range(1, NUM_LEVELS + 1):
        print(f"{i:3d} | {AWGN_SNR_DB[i]:10.2f} | {PHASE_NOISE_STD[i]:10.2f} | {PA_COMPRESSION_DB[i]:10.2f}")
    print("-" * 70)


# ==============================================================================
#                              测试
# ==============================================================================
if __name__ == "__main__":
    print_statistics()
    print_param_table()
    
    # 测试
    print("\n测试增强效果:")
    x = np.random.randn(2, 1024).astype(np.float32)
    
    test_names = ["Clean", "N5", "P5", "A5", "N5_A5"]
    for name in test_names:
        aug = create_augmentor_by_name(name)
        y = aug(x)
        diff = np.mean(np.abs(y - x))
        print(f"  {name:15s}: diff = {diff:.4f}")
    
    print(f"\n总配置数: {len(get_all_configs())}")