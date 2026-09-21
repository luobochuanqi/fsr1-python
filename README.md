# fsr1-python

AMD FidelityFX Super Resolution 1.0（FSR 1.0）的 NumPy 实现：把 EASU 上采样趟与 RCAS 锐化趟
用向量化 NumPy 重写一遍，跑得动、看得见。

算法、常量与系数取自 AMD 的 `ffx_fsr1.h` / `ffx_a.h`（MIT 许可）。参考实现是本仓库外的只读
克隆：`../repo/FidelityFX-FSR/`。

## 术语

本仓库统一使用下面这套说法，口径对齐 AMD 官方文档（`../repo/FidelityFX-FSR/docs/`），避免
同义词混用：

| 中文 | English | 说明 |
|---|---|---|
| 上采样 | upscaling | FSR 1.0 做的这件事本身；官方描述为 spatial upscaling |
| 上采样趟 | upscaling pass | EASU，第一趟 |
| 锐化趟 | sharpening pass | RCAS，第二趟 |
| 每维缩放倍数 | scale factor per dimension | 例如 1.5×，指单边长度之比 |
| 面积缩放倍数 | area scale factor | 每维倍数的平方；官方适用范围为 1X–4X |
| 质量档位 | quality mode | Ultra Quality / Quality / Balanced / Performance |
| 档（stops） | stops | RCAS 的锐化量单位，每整档减半 |
| 抽头 | tap | 参与滤波的采样点 |
| 感知色彩空间 | perceptual color space | sRGB / Gamma 2.0，非线性 |
| 去振铃 | deringing | EASU 末段把结果夹回局部 min/max |

## FSR 1.0 是什么

纯空间的两趟滤波，顺序固定：

```
低清 RGB --EASU（上采样趟）--> 高清 RGB --RCAS（锐化趟）--> 锐化后的高清 RGB
```

- **EASU**（Edge Adaptive Spatial Upsampling）：边缘自适应的空间上采样。12 抽头窗口的权重按
  局部梯度方向旋转、拉伸，边缘收紧、平坦区做柔性插值。
- **RCAS**（Robust Contrast Adaptive Sharpening）：鲁棒对比度自适应锐化。按局部对比度给
  4 邻域加权，并做硬限幅以抑制振铃。

不需要历史帧、运动矢量或额外缓冲——所以它是单帧算法，不是时域方法。

官方给的使用条件（照抄 `docs/FidelityFX-FSR-Overview-Integration.pdf`，都会直接影响观感）：

- 输入必须已抗锯齿，否则硬边缘会被当成细节进一步放大。
- 输入必须归一化到 [0, 1]，且位于感知色彩空间（sRGB / Gamma 2.0，**不是**线性）。负值输入会让
  RCAS 输出 NaN。
- 输入应当无噪点；颗粒、色差一类噪声效果应在 FSR 之后施加。

## 环境与运行

独立的 uv 项目，Python >= 3.12，依赖只有 `numpy` 与 `pillow`。

```bash
uv run python fsr1.py --selftest
uv run python fsr1.py low.png -o up.png --scale 2 --sharpness 0.25
```

- `--scale`：每维缩放倍数。
  FSR 1.0 官方适用范围是**面积** 1X–4X，即每维约 1.0–2.0；官方四档质量档位对应每维
  1.3（Ultra Quality）/ 1.5（Quality）/ 1.7（Balanced）/ 2.0（Performance）。
  本实现不对每维倍数设限，任意非整数倍数也能跑。
- `--sharpness`：RCAS 锐化量，单位是档（stops）。每整档减半，0.0 最锐，约 2.0 以上不再有可见
  差别。AMD 文档推荐 0.2，示例代码默认 0.25——这里的默认值是 0.25（跟示例代码）。

## 库 API

```python
import fsr1

img = fsr1._to_float("low.png")                    # -> float32 (H,W,3)，取值 [0,1]
up = fsr1.upscale(img, 2.0, sharpness=0.25)        # EASU 上采样 + RCAS 锐化
fsr1._save(up, "up.png")

fsr1.easu(img, 2.0)                                # 只跑上采样趟
fsr1.rcas(img, 0.25)                               # 只跑锐化趟
fsr1.upscale(img, 2.0, sharpness=None)             # 跳过 RCAS，只要 EASU 的结果
```

输入按感知色彩空间（sRGB）处理，不做线性化——这是 FSR 1.0 的设计前提。全程 float32，对应参考
的 FP32 路径。

## 与参考实现的对应

这是**静态代码分析**层面的对照：逐表达式对应参考的 FP32 标量路径（`FsrEasuF` / `FsrRcasF`），
不对齐 GPU 的位级浮点行为。`fsr1.py` 中每段代码都带 `# ref:` 注释标注出处。

| `fsr1.py` | 参考 |
|---|---|
| `easu()` 的 `pp = ip*con0 + con0.zw` | `FsrEasuCon` L171-173、`FsrEasuF` L324-326 |
| `TAPS` 的 12 个抽头位置与字母 | `FsrEasuF` L328-434 |
| `easu.setf()` 梯度方向与长度累加 | `FsrEasuSetF` L275-313（四次调用 L383-386） |
| `dir` 归一化、`length`、`stretch`、`lob`、`clp` | `FsrEasuF` L389-409 |
| `min4` / `max4` 去振铃边界 | `FsrEasuF` L416-419 |
| 12 次抽头加权累加 | `FsrEasuTapF` L239-272（调用 L423-434） |
| `aC * 1/aW` 归一化与去振铃夹取 | `FsrEasuF` L437 |
| `rcas()` 的 `con = 2^-sharpness` | `FsrRcasCon` L667 |
| `rcas.shift()` 的 3x3 十字邻域 | `FsrRcasF` L697-707 |
| `hit_min` / `hit_max` / `lobe` 限幅 | `FsrRcasF` L747-759 |
| `rcpL` 解算 | `FsrRcasF` L765-768 |
| `RCAS_LIMIT` | `ffx_fsr1.h` L654 |
| `rcp_lo` / `rsq_lo` / `rcp_med` / `_rcp` / `_sat` | `ffx_a.h` L1843-1845、L326、L747 |
| `_luma` | `ffx_fsr1.h` L363 |

三条对应约定（不是风格偏好，是行为等价的前提）：

1. **HLSL 的 `min`/`max` 用 `np.fmin`/`np.fmax` 对应。** D3D 规定其中一个操作数为 NaN 时返回
   另一个操作数，两者语义一致。这不是可有可无的细节：RCAS 的限幅里天然存在 `0/0`，用普通
   `min`/`max` 会让纯黑/纯白常数图整图变成 NaN。
2. **`ARcpF1` 用真除法对应。** 它就是普通 IEEE `1/x`，`x==0` 得到 `inf`，随后由上面那对
   NaN 容错的 `min`/`max` 吸收，因此不需要任何除零保护。
3. **`textureGather` 用直接取样对应。** 参考用 4 次 gather 取 4 组 2x2 纹素块，那是 GPU 的打包
   读取；这里直接按 12 个抽头的像素坐标取值，取到的是同一批像素。

范围：只实现 **FP32 标量路径**。参考里的 FP16 打包路径（`*H`）与双 tile 路径（`*Hx2`）是为
GPU 吞吐服务的，与算法无关。AMD 以 FP16 为默认、FP32 为回退，这里对应的是回退路径。

## 自检覆盖什么

`uv run python fsr1.py --selftest` 检查的是可观察行为，不是代码长相：

- 形状正确、输出有限；
- 平坦中灰经过两趟几乎不变（漂移 < 1e-3）；
- 硬边缘保持硬，且不越过 [0,1]、不被去振铃放行出振铃；
- 纯黑与纯白常数图**保持有限**——这一条专门盯住上面第 1 条约定的失效方式；
- 纯黑上的一颗孤立亮点**被锐化**（0.5 → 0.8627）。这一条是同一个 `0/0` 的另一面：若把
  `x/0` 简单取 0 抹平限幅，亮点就得不到锐化，而参考会锐化它。

两处实测数值（可由上面的自检复现，也是默认 `sharpness=0.25` 下的结果）：纯白常数图回来是
0.99825，不是 1.0——这是参考自身的中精度倒数 `APrxMedRcpF1` 带来的约 0.17% 误差（约 0.4/255），
本实现如实保留。

## 许可

算法与常量来自 AMD FidelityFX-FSR，MIT 许可。本仓库只重写了数学表达式，未复制其代码。