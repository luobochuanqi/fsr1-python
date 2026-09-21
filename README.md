# fsr1-python

AMD FidelityFX Super Resolution 1.0 的 NumPy 实现。用 NumPy 重写了 EASU 和 RCAS，能直接跑图。
代码在 `fsr1.py`。

算法、常量、系数来自 AMD 的 `ffx_fsr1.h` / `ffx_a.h`（MIT）。参考实现在 `../repo/FidelityFX-FSR/`，
只读克隆。

## 术语

这个仓库统一用下面这些说法，口径对齐 AMD 官方文档（`../repo/FidelityFX-FSR/docs/`）：

| 中文 | English | 说明 |
|---|---|---|
| 上采样 | upscaling | FSR 1.0 做的事；官方叫 spatial upscaling |
| EASU | EASU | 第一级，负责上采样 |
| RCAS | RCAS | 第二级，负责锐化 |
| 每维缩放倍数 | scale factor per dimension | 例如 1.5×，单边长度之比 |
| 面积缩放倍数 | area scale factor | 每维倍数的平方；官方适用范围 1X–4X |
| 质量档位 | quality mode | Ultra Quality / Quality / Balanced / Performance |
| 档 | stops | RCAS 锐化量的单位，每整档减半 |
| 抽头 | tap | 参与滤波的采样点 |
| 感知空间 | perceptual space | sRGB / Gamma 2.0，非线性 |
| 去振铃 | deringing | EASU 收尾时把结果夹回局部 min/max |

## FSR 1.0 是什么

两次纯空间滤波，顺序固定：

```
低清 RGB --EASU--> 高清 RGB --RCAS--> 输出
```

EASU 做上采样。用 12 个抽头：先估出该点的梯度方向和边缘强度，再把抽头按方向旋转、按强度拉伸，
加权求和。边缘上权重更集中，所以放大后边缘不糊。

RCAS 做锐化。只看中心像素和上下左右 4 个邻居：按中心离邻居多远算出负权重（lobe），把邻居的值
按 lobe 加到中心上，再限幅，防止过冲。它不改变图像尺寸。

EASU 和 RCAS 都不看历史帧，不需要运动矢量，也不需要额外缓冲。所以是单帧算法。

AMD 对输入的要求（照抄 `docs/FidelityFX-FSR-Overview-Integration.pdf`，每条都直接影响观感）：

- 输入要先做抗锯齿。否则硬边缘会被当成细节继续放大。
- 输入归一化到 [0, 1]，并且是感知空间（sRGB / Gamma 2.0，不是线性）。负值输入会让 RCAS 输出
  NaN。
- 输入不能有噪点。颗粒、色差这类效果要放在 FSR 之后。

## 环境与运行

独立的 uv 项目，Python >= 3.12，依赖只有 `numpy` 和 `pillow`。

```bash
uv run python fsr1.py --selftest
uv run python fsr1.py low.png -o up.png --scale 2 --sharpness 0.25
```

- `--scale`：每维缩放倍数。官方适用范围是**面积** 1X–4X，也就是每维约 1.0–2.0；官方四档质量
  档位对应每维 1.3（Ultra Quality）、1.5（Quality）、1.7（Balanced）、2.0（Performance）。这里
  不限制每维倍数，非整数也能跑。
- `--sharpness`：RCAS 锐化量，单位是档（stops）。每整档减半，0.0 最锐，约 2.0 以上看不出差别。
  AMD 文档推荐 0.2，示例代码默认 0.25——这里默认 0.25，跟示例代码。

## 库 API

```python
import fsr1

img = fsr1._to_float("low.png")                    # -> float32 (H,W,3)，取值 [0,1]
up = fsr1.upscale(img, 2.0, sharpness=0.25)        # EASU + RCAS
fsr1._save(up, "up.png")

fsr1.easu(img, 2.0)                                # 只跑 EASU
fsr1.rcas(img, 0.25)                               # 只跑 RCAS
fsr1.upscale(img, 2.0, sharpness=None)             # 只跑 EASU
```

输入按感知空间（sRGB）处理，不做线性化——这是 FSR 1.0 的前提。全程 float32，对应参考的 FP32
路径。

## 和参考实现的对应

这是静态代码对照：逐表达式对齐参考的 FP32 标量路径（`FsrEasuF` / `FsrRcasF`），不管 GPU 的
位级浮点行为。`fsr1.py` 里每段代码都带 `# ref:`，写明对应参考的哪一行。

| `fsr1.py` | 参考 |
|---|---|
| `easu()` 的 `pp = ip*con0 + con0.zw` | `FsrEasuCon` L171-173、`FsrEasuF` L324-326 |
| `TAPS` 的 12 个抽头位置和字母 | `FsrEasuF` L328-434 |
| `easu.setf()` 累加梯度和长度 | `FsrEasuSetF` L275-313（调用 L383-386） |
| `dir` 归一化、`length`、`stretch`、`lob`、`clp` | `FsrEasuF` L389-409 |
| `min4` / `max4` 去振铃边界 | `FsrEasuF` L416-419 |
| 12 次抽头加权累加 | `FsrEasuTapF` L239-272（调用 L423-434） |
| `aC * 1/aW` 归一化和去振铃夹取 | `FsrEasuF` L437 |
| `rcas()` 的 `con = 2^-sharpness` | `FsrRcasCon` L667 |
| `rcas.shift()` 的 3x3 十字邻域 | `FsrRcasF` L697-707 |
| `hit_min` / `hit_max` / `lobe` 限幅 | `FsrRcasF` L747-759 |
| `rcpL` 解算 | `FsrRcasF` L765-768 |
| `RCAS_LIMIT` | `ffx_fsr1.h` L654 |
| `rcp_lo` / `rsq_lo` / `rcp_med` / `_rcp` / `_sat` | `ffx_a.h` L1843-1845、L326、L747 |
| `_luma` | `ffx_fsr1.h` L363 |

三条对应（不是风格问题，是行为对不对的问题）：

1. **HLSL 的 `min`/`max` 用 `np.fmin`/`np.fmax`。** D3D 规定有一个操作数是 NaN 时返回另一个，
   两者一致。这条不能省：RCAS 限幅里天然出现 `0/0`，换成普通 `min`/`max`，纯黑或纯白图会整张
   变 NaN。
2. **`ARcpF1` 用真除法。** 它就是 IEEE 的 `1/x`，`x` 为 0 时得到 `inf`，被上面那对 min/max
   吸收，所以不需要除零保护。
3. **`textureGather` 用直接取样。** 参考用 4 次 gather 取 4 组 2x2 纹素块，那是 GPU 的打包读法。
   这里直接按 12 个抽头的坐标取像素，取到的是同一批像素。

只做 **FP32 标量路径**。参考里的 FP16 打包路径（`*H`）和双 tile 路径（`*Hx2`）是为 GPU 吞吐
服务的，和算法无关。AMD 默认走 FP16、FP32 是回退，这里对应回退路径。

## 自检覆盖什么

`uv run python fsr1.py --selftest` 查的是行为，不是代码长相：

- 形状对、输出有限；
- 平坦中灰走完整个管线几乎不变（漂移 < 1e-3）；
- 硬边缘保持硬，不越出 [0,1]，也不被去振铃放行出振铃；
- 纯黑、纯白常数图保持有限——这条专门盯上面第 1 条的失效方式；
- 纯黑上的孤立亮点被锐化（0.5 → 0.8627）——这是同一个 `0/0` 的另一面：把 `x/0` 取 0 抹平限幅，
  亮点就锐化不了，而参考会锐化它。

两个实测数值，可由自检复现（默认 `sharpness=0.25`）：纯白常数图回来是 0.99825 而不是 1.0，
这是参考自身中精度倒数 `APrxMedRcpF1` 的误差（约 0.17%，约 0.4/255），这里如实保留。

## 许可

算法和常量来自 AMD FidelityFX-FSR，MIT 许可。这里只重写了数学表达式，没有复制它的代码。