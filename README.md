# fsr1-python

AMD FidelityFX Super Resolution 1.0（FSR1）的 NumPy 移植，逐位对应 AMD 官方参考实现。
用于学习算法本身：算法、常量、系数全部取自 `ffx_fsr1.h` / `ffx_a.h`（MIT），参考实现在
`https://github.com/GPUOpen-Effects/FidelityFX-FSR`，代码中所有不直观处均带 `# ref:` 标注。

## 算法概要

FSR 1.0 为两次纯空间滤波，以固定顺序：

    低清 RGB --EASU--> 高清 RGB --RCAS--> 输出

- **EASU**（上采样）：12 个抽头（tap）。先在输出点处估出梯度方向和边缘强度，再按方向
  旋转抽头、按强度拉伸，加权求和。边缘处权重集中，放大后边缘不糊。
- **RCAS**（锐化）：仅使用中心像素及上下左右 4 个邻像素。按中心与邻像素的距离导出负权重
  lobe，以 lobe 将邻像素叠加到中心，再限幅抑制过冲。不改变图像尺寸。

两级滤波均不引用历史帧，不需要运动矢量，亦无额外缓冲——这是 FSR1 与 FSR2/3 的
本质区别（后者为时序算法，依赖运动矢量与历史帧）。

仅实现 FP32 标量路径（`FsrEasuF` / `FsrRcasF`）。参考中的 FP16 打包路径与双 tile
路径服务于 GPU 吞吐，与算法无关；AMD 默认 FP16、FP32 为回退，本实现即对应回退路径。

## 使用

```bash
uv run python fsr1.py INPUT.png -o OUTPUT.png --scale 2.0 --sharpness 0.25
uv run python fsr1.py --selftest
```

| 参数          | 说明                                                                                                                   |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `--scale`     | 每维倍率，无需为整数，默认 2.0                                                                                         |
| `--sharpness` | RCAS 锐化量，单位为档（stops）：每整档锐化量减半，0.0 最锐，约 2.0 以上无可辨差异。AMD 文档推荐 0.2，示例代码默认 0.25 |

Python 库接口：`upscale(img, scale, sharpness=None)`，输入输出均为 float32 RGB，
取值 [0,1]；`sharpness=None` 表示仅执行 EASU。

## 依赖

Python >= 3.12，uv 管理。运行仅依赖 NumPy，命令行另需 Pillow。

## 自检

`--selftest` 覆盖形状、值域、有限性，以及两条必须与参考一致的行为：
纯黑/纯白常数图的 NaN 处理（`0/0` 由 NaN 容错 min/max 吸收）与孤立亮点的锐化。
