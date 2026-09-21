# fsr1-python

AMD FidelityFX Super Resolution 1（FSR1）的 NumPy 独立实现。不是为了调用 AMD 的 shader，而是把 FSR1 的两趟算法用 Python 重写一遍，跑得动、看得见，方便理解它到底做了什么。

参考实现在 `../repo/FidelityFX-FSR/`（AMD 官方仓库克隆，只读，不属于本仓库）。

## FSR1 是什么

FSR1 是纯**空间**超分：不看历史帧，只对当前这一帧做滤波。它由两趟组成，按顺序执行：

```
低清 RGB --EASU--> 高清 RGB --RCAS--> 锐化后的高清 RGB
```

- **EASU**（Edge Adaptive Spatial Upsampling）：边缘自适应空间上采样。用 12 抽头窗口，窗口按局部梯度方向旋转、拉伸，平坦区做类似 Lanczos 的插值，边缘处收紧，从而在放大的同时保住边缘锐度。
- **RCAS**（Robust Contrast Adaptive Sharpening）：鲁棒对比度自适应锐化。按局部对比度给 4 邻域重新加权，并做硬限幅，避免过冲产生振铃。

## 环境与运行

本项目是独立的 uv 项目，依赖只有 `numpy` 和 `pillow`。

```bash
uv run python fsr1.py --selftest              # 内置自检
uv run python fsr1.py low.png -o up.png --scale 2 --sharpness 0.25
```

- `--scale`：放大倍数，任意 ≥ 1 的浮点（FSR1 的 EASU 本就支持任意比例，不是固定 2x）。
- `--sharpness`：RCAS 锐化，单位是“档”（stops），`0` 最锐，越大越柔和，AMD 官方 sample 默认 `0.25`。传 `--sharpness` 为负值不合法；想跳过 RCAS 用库 API 传 `sharpness=None`。

## 库 API

```python
import fsr1

img = fsr1._to_float("low.png")      # -> float32 (H,W,3) in [0,1]
up = fsr1.upscale(img, scale=2.0, sharpness=0.25)   # EASU + RCAS
fsr1._save(up, "up.png")

fsr1.easu(img, 2.0)                  # 单独跑 EASU
fsr1.rcas(img, 0.25)                 # 单独跑 RCAS
```

输入按 sRGB 处理，不做线性化——这正是 FSR1 的设计前提（它面向 sRGB 游戏帧）。全程 float32 运算，以贴合 shader 的 32-bit 浮点路径。

## 与 AMD 参考实现的一致性

这是**静态代码分析**层面的对照：本实现逐表达式对应参考的标量 32-bit 浮点路径（`FSR_EASU_F` / `FSR_RCAS_F`），不对齐 GPU 的位级浮点行为。下表给出对应关系，`fsr1.py` 里每段代码也带 `# ref:` 注释标注出处。

| 本实现 | 参考（`ffx_fsr1.h` / `ffx_a.h`） |
|---|---|
| `easu` 的坐标映射 `pp = ip*con0 + con0.zw` | `FsrEasuCon` L171-173，`FsrEasuF` L324-326 |
| 12 个抽头的位置与字母 | `FsrEasuF` L328-434 |
| `setf()` 梯度方向/长度累加 | `FsrEasuSetF` L275-313（四次调用 L383-386） |
| `dir` 归一化、`length`、`stretch`、`lob`、`clp` | `FsrEasuF` L389-409 |
| `min4` / `max4`（2x2 最近邻夹取） | `FsrEasuF` L416-419 |
| 12 次抽头加权累加 | `FsrEasuTapF` L239-272（调用 L423-434） |
| 归一化 + dering 夹取 | `FsrEasuF` L437 |
| `rcas` 的 `con = 2^-sharpness` | `FsrRcasCon` L667 |
| 3x3 “十字”邻域 | `FsrRcasF` L697-707 |
| `hitMin` / `hitMax` / `lobe` 限幅 | `FsrRcasF` L747-759 |
| resolve（用中精度倒数） | `FsrRcasF` L765-768 |
| `RCAS_LIMIT = 0.25 - 1/16` | `ffx_fsr1.h` L654 |
| `rcp_lo` / `rsq_lo` / `rcp_med` | `ffx_a.h` L1843-1845 |

### 有意的偏差（刻意为之，不影响“是 FSR1 的效果”）

1. **`textureGather` → 直接整数坐标取样 + clamp-to-edge**。参考用 4 次 gather 取 4 组 2x2 纹素块（为 GPU 的 SIMD 打包服务）；本实现直接按已推导出的 12 个抽头像素坐标取值，物理上取的是同一批像素，只是省掉了打包。
2. **`x/0` 取 0**。参考在纯黑/纯白区域的限幅里依赖 HLSL 的 nan 不敏感比较；本实现把除零的极限值直接取 0，等价且保证输出有限（不会出 nan）。
3. **只移植标量 32-bit 浮点路径**。参考里的 `H`（16-bit 打包）与 `Hx2`（双 tile）版本是 GPU 为吞吐做的向量化，与算法本身无关，未移植。

### 已验证

- `--selftest`：形状、值域、无 nan；平坦色块经过 EASU/RCAS 基本不变；纯黑/纯白不产生 nan。
- 两个 pass 对图像转置是对称的（12 抽头核在 x↔y 交换下自洽），实测误差 ≈ float32 机器精度，说明 x/y 轴没有接错。
- 真实帧 3x 放大的 acutance（平均 |Laplacian|）明显高于双线性：FSR1 比双线性锐利得多。