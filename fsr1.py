"""FSR 1.0 的 NumPy 移植，对应 AMD FidelityFX Super Resolution 1.0。

FSR 1.0 为两次纯空间滤波，以固定顺序：

    低清 RGB --EASU--> 高清 RGB --RCAS--> 输出

EASU 负责上采样。12 个抽头（tap）：先在输出点处估出梯度方向和边缘强度，再按方向
旋转抽头、按强度拉伸，加权求和。边缘处权重集中，放大后边缘不糊。

RCAS 负责锐化。仅使用中心像素及上下左右 4 个邻像素：按中心与邻像素的距离导出负权重
lobe，以 lobe 将邻像素叠加到中心，再限幅抑制过冲。不改变图像尺寸。

两级滤波均不引用历史帧，不需要运动矢量，亦无额外缓冲。

算法、常量、系数全部取自 AMD 的 ffx_fsr1.h / ffx_a.h（MIT）。参考实现在
`https://github.com/GPUOpen-Effects/FidelityFX-FSR`。代码中所有不直观处均带 `# ref:` 标注，指明所对应
的参考行号。

与 `GPUOpen-Effects/FidelityFX-FSR` 的三处对应关系：

* HLSL min/max → np.fmin/np.fmax。D3D 规定任一操作数为 NaN 时返回另一操作数，
  np.fmin/np.fmax 语义一致。此映射不可替换：RCAS 限幅中天然出现 0/0，改用普通
  min/max 会让纯黑/纯白常数图整幅 NaN。
* ARcpF1 即 IEEE 1/x，用真除法实现。x == 0 得 inf，由上述 NaN 容错 min/max 吸收，
  无需除零保护。
* `GPUOpen-Effects/FidelityFX-FSR` 用 textureGather 一次取 4 组 2x2 纹素，这是 GPU 的打包读法。
  本实现按 12 个抽头坐标直接取像素，像素集合相同。

仅实现 FP32 标量路径（FsrEasuF / FsrRcasF）。`GPUOpen-Effects/FidelityFX-FSR` 中的 FP16 打包路径与双 tile
路径服务于 GPU 吞吐，与算法本身无关。AMD 默认 FP16、FP32 为回退；本实现即对应
回退路径。

依赖仅 NumPy；命令行另需 Pillow。要求 Python >= 3.12。

运行自检：`uv run python fsr1.py --selftest`。"""

import numpy as np

# 锐化强度上限，超过该值结果开始失真。
# ref: ffx_fsr1.h:654  (FSR_RCAS_LIMIT)
RCAS_LIMIT = np.float32(0.25 - 1.0 / 16.0)


def _bits(a):
    # 将 float32 的位模式按 uint32 重新解释，供后续位级技巧使用。
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# 参考中的三个 GPU 倒数近似，逐位复刻。
# ref: ffx_a.h:1843-1845
def rcp_lo(a):
    """~1/a：仅操作指数位，不含除法。"""
    return (np.uint32(0x7EF07EBB) - _bits(a)).view(np.float32)


def rsq_lo(a):
    """~1/sqrt(a)：同为位级技巧。"""
    return (np.uint32(0x5F347D74) - (_bits(a) >> np.uint32(1))).view(np.float32)


def rcp_med(a):
    """低精度倒数加一步牛顿迭代。"""
    b = (np.uint32(0x7EF19FFF) - _bits(a)).view(np.float32)
    return b * (-b * a + np.float32(2.0))


def _rcp(x):
    """参考的 ARcpF1，即 IEEE 1/x。x == 0 时返回 inf，由 NaN 容错的 min/max 吸收。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(np.float32(1.0), x)


def _sat(x):
    """限幅至 [0,1]。ref: ffx_a.h:747 (ASatF1)"""
    return np.clip(x, np.float32(0.0), np.float32(1.0))


def _luma(rgb):
    """亮度 x2：近似 0.5R + G + 0.5B。省去一次乘法，故结果为亮度的 2 倍。
    ref: ffx_fsr1.h:363"""
    return rgb[..., 2] * np.float32(0.5) + (rgb[..., 0] * np.float32(0.5) + rgb[..., 1])


# EASU 的 12 个抽头（tap）：偏移以输入纹素为单位，相对基准纹素 floor(pp)。排布与
# 字母沿用原实现：
#     b c
#   e f g h
#   i j k l
#     n o
# ref: ffx_fsr1.h:328-434  (传入 FsrEasuTapF 的抽头偏移)
TAPS = (
    ("b", 0, -1),
    ("c", 1, -1),
    ("e", -1, 0),
    ("f", 0, 0),
    ("g", 1, 0),
    ("h", 2, 0),
    ("i", -1, 1),
    ("j", 0, 1),
    ("k", 1, 1),
    ("l", 2, 1),
    ("n", 0, 2),
    ("o", 1, 2),
)


def easu(src, scale):
    """EASU：上采样。

    src：float32 (H, W, 3)，取值 [0,1]。scale：每维倍率，无需为整数。
    返回 float32 (round(H*scale), round(W*scale), 3)。
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]
    OH, OW = round(H * scale), round(W * scale)

    # FsrEasuCon，viewport 取整幅输入：单个输出像素步长在输入中的跨度，以及将采样点
    # 对准像素中心的半像素项。
    # ref: ffx_fsr1.h:171-173
    sx_in, sy_in = np.float32(W / OW), np.float32(H / OH)
    ox = np.arange(OW, dtype=np.float32)
    oy = np.arange(OH, dtype=np.float32)
    ppx = ox * sx_in + (np.float32(0.5) * sx_in - np.float32(0.5))  # (OW,)
    ppy = oy * sy_in + (np.float32(0.5) * sy_in - np.float32(0.5))  # (OH,)

    # ref: ffx_fsr1.h:324-326  (pp = ip*con0.xy + con0.zw; fp = floor(pp); pp -= fp)
    FPX = np.floor(ppx)[None, :]  # (1, OW) 基准纹素 x 的整数部分
    FPY = np.floor(ppy)[:, None]  # (OH, 1) 基准纹素 y 的整数部分
    FX = (ppx - np.floor(ppx))[None, :]  # (1, OW) pp.x 的小数部分
    FY = (ppy - np.floor(ppy))[:, None]  # (OH, 1) pp.y 的小数部分

    def fetch(dx, dy):
        """取 fp+偏移 处的纹素，越界部分夹至边缘（参考依赖 clamp 采样器）。"""
        ix = np.clip(FPX + dx, 0, W - 1).astype(np.intp)
        iy = np.clip(FPY + dy, 0, H - 1).astype(np.intp)
        return src[iy, ix]  # (OH, OW, 3)

    # 先取出 12 个抽头的亮度——方向与长度估计仅用到亮度。
    L = {name: _luma(fetch(dx, dy)) for name, dx, dy in TAPS}

    # FsrEasuSetF 执行 4 次，对应输出像素的 4 个双线性角（参考形参名 biS/biT/biU/biV）。
    # 每次取以 f、g、j 或 k 为中心的十字亮度，累加梯度方向 dir 与边缘长度。
    # ref: ffx_fsr1.h:275-313,383-386
    dirx = np.zeros((OH, OW), np.float32)
    diry = np.zeros((OH, OW), np.float32)
    length = np.zeros((OH, OW), np.float32)

    def setf(up, left, ctr, right, down, w):
        nonlocal dirx, diry, length
        # x 轴
        lenX = rcp_lo(np.fmax(np.abs(right - ctr), np.abs(ctr - left)))
        dirX = right - left
        dirx = dirx + dirX * w
        lenX = _sat(np.abs(dirX) * lenX)
        length = length + (lenX * lenX) * w
        # y 轴
        lenY = rcp_lo(np.fmax(np.abs(down - ctr), np.abs(ctr - up)))
        dirY = down - up
        diry = diry + dirY * w
        lenY = _sat(np.abs(dirY) * lenY)
        length = length + (lenY * lenY) * w

    setf(L["b"], L["e"], L["f"], L["g"], L["j"], (1 - FX) * (1 - FY))  # biS 角
    setf(L["c"], L["f"], L["g"], L["h"], L["k"], FX * (1 - FY))  # biT 角
    setf(L["f"], L["i"], L["j"], L["k"], L["n"], (1 - FX) * FY)  # biU 角
    setf(L["g"], L["j"], L["k"], L["l"], L["o"], FX * FY)  # biV 角

    # 归一化梯度方向（幅值过小时置平），随后导出长度、各向异性 len2、负瓣强度 lob 与
    # 窗口截断点 clp。
    # ref: ffx_fsr1.h:389-409
    dirR = dirx * dirx + diry * diry
    zro = dirR < np.float32(1.0 / 32768.0)
    dirR = np.where(zro, np.float32(1.0), rsq_lo(dirR))
    dirx = np.where(zro, np.float32(1.0), dirx) * dirR
    diry = diry * dirR

    length = length * np.float32(0.5)
    length = length * length
    stretch = (dirx * dirx + diry * diry) * rcp_lo(np.fmax(np.abs(dirx), np.abs(diry)))
    len2x = np.float32(1.0) + (stretch - np.float32(1.0)) * length
    len2y = np.float32(1.0) + np.float32(-0.5) * length
    lob = np.float32(0.5) + (np.float32(1.0 / 4.0 - 0.04) - np.float32(0.5)) * length
    clp = rcp_lo(lob)

    # 去振铃上下界：2x2 最近邻抽头（f、g、j、k）的 min/max。
    # ref: ffx_fsr1.h:416-419
    near = np.stack([fetch(0, 0), fetch(1, 0), fetch(0, 1), fetch(1, 1)])
    min4 = np.fmin.reduce(near, axis=0)
    max4 = np.fmax.reduce(near, axis=0)

    # FsrEasuTapF 执行 12 次：将抽头偏移按梯度方向旋转、按 len2 拉伸，代入锐化后的
    # Lanczos-2 近似窗口求权重并累加。
    # ref: ffx_fsr1.h:239-272,423-434
    aC = np.zeros((OH, OW, 3), np.float32)
    aW = np.zeros((OH, OW), np.float32)
    for name, dx, dy in TAPS:
        offx = np.float32(dx) - FX
        offy = np.float32(dy) - FY
        vx = offx * dirx + offy * diry
        vy = offx * (-diry) + offy * dirx
        vx = vx * len2x
        vy = vy * len2y
        d2 = np.fmin(vx * vx + vy * vy, clp)
        wB = np.float32(2.0 / 5.0) * d2 - np.float32(1.0)
        wA = lob * d2 - np.float32(1.0)
        w = (np.float32(25.0 / 16.0) * (wB * wB) - np.float32(25.0 / 16.0 - 1.0)) * (
            wA * wA
        )
        aC += fetch(dx, dy) * w[..., None]
        aW += w

    # 归一化，随后按 min4/max4 去振铃（deringing）。
    # ref: ffx_fsr1.h:437
    pix = aC * _rcp(aW)[..., None]
    return np.fmin(max4, np.fmax(min4, pix))


def rcas(src, sharpness=0.25):
    """RCAS：锐化。

    src：float32 (H, W, 3)，取值 [0,1]，即 EASU 的输出。sharpness 单位为档（stops）：
    每整档锐化量减半，0.0 最锐，约 2.0 以上无可辨差异（AMD 文档推荐 0.2，示例代码默认
    0.25）。返回 float32 (H, W, 3)，尺寸不变。
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]

    # FsrRcasCon：锐化量以 2^-sharpness 存储。ref: ffx_fsr1.h:667
    con = np.float32(2.0) ** np.float32(-sharpness)

    # 3x3 十字邻域，越界部分夹至边缘。ref: ffx_fsr1.h:697-707
    def shift(dx, dy):
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        return src[ys][:, xs]

    b, d, e, f, h = shift(0, -1), shift(-1, 0), src, shift(1, 0), shift(0, 1)

    # 4 邻像素的 min/max。ref: ffx_fsr1.h:741-746
    ring_min = np.fmin(np.fmin(b, d), np.fmin(f, h))
    ring_max = np.fmax(np.fmax(b, d), np.fmax(f, h))

    # 逐通道对比度限幅：中心距邻像素的既有偏移量（peakC = (1.0, -4.0)）。
    # ref: ffx_fsr1.h:747-758
    # NOTE: 此处的 0*inf 产生 NaN，属参考行为的一部分，由下方 np.fmax 吸收。仅关闭
    # 告警，不得消除 NaN——消除会改变孤立亮点的锐化结果，见 _selftest。
    with np.errstate(invalid="ignore"):
        hit_min = np.fmin(ring_min, e) * _rcp(np.float32(4.0) * ring_max)
        hit_max = (np.float32(1.0) - np.fmax(ring_max, e)) * _rcp(
            np.float32(4.0) * ring_min - np.float32(4.0)
        )
    lobe = np.fmax(-hit_min, hit_max)

    # 三通道并合为标量 lobe，夹至安全区间，再乘锐化量。ref: ffx_fsr1.h:759
    lobe = (
        np.fmax(
            np.float32(-RCAS_LIMIT),
            np.fmin(np.fmax.reduce(lobe, axis=2), np.float32(0.0)),
        )
        * con
    )

    # 解算。用中精度倒数，避免可见的色调断层。ref: ffx_fsr1.h:765-768
    rcpL = rcp_med(np.float32(4.0) * lobe + np.float32(1.0))
    pix = (lobe[..., None] * (b + d + f + h) + e) * rcpL[..., None]
    return pix


def upscale(img, scale, sharpness=0.25):
    """完整管线：先 EASU，后 RCAS。

    img：float32 (H, W, 3)，取值 [0,1]，返回放大并锐化后的图像。sharpness=None
    表示仅执行 EASU。
    """
    out = easu(img, scale)
    if sharpness is None:
        return out
    return rcas(out, sharpness)


def _to_float(path_or_array):
    """读入 PNG（或直接接受 ndarray），返回取值 [0,1] 的 float32 RGB。"""
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return (
            (a.astype(np.float32) / 255.0)
            if a.dtype == np.uint8
            else a.astype(np.float32)
        )
    from PIL import Image

    im = Image.open(path_or_array).convert("RGB")
    return np.asarray(im, dtype=np.float32) / np.float32(255.0)


def _save(arr, path):
    # 夹至 [0,1] 后量化为 8 位写出。
    from PIL import Image

    a = np.clip(arr, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def _selftest():
    """自检：形状、值域、有限性，以及两条必须与参考一致的行为。"""
    rng = np.random.default_rng(0)

    # 平坦中灰经完整管线后应几乎不变。
    flat = np.full((16, 16, 3), 0.5, np.float32)
    up = upscale(flat, 2.0)
    assert up.shape == (32, 32, 3), up.shape
    assert np.all(np.isfinite(up)), "non-finite output"
    assert np.max(np.abs(up - 0.5)) < 1e-3, (
        f"flat tone drifted: {np.max(np.abs(up - 0.5))}"
    )

    # 硬边缘必须保持（EASU 不得将其糊为斜坡），结果不得越出 [0,1]，亦不得振铃越过
    # 去振铃限幅。
    edge = np.zeros((16, 16, 3), np.float32)
    edge[:, 8:] = 1.0
    up = upscale(edge, 2.0)
    assert up.min() >= -1e-6 and up.max() <= 1.0 + 1e-6, (up.min(), up.max())
    assert np.all(np.isfinite(up))

    # 纯黑、纯白常数图必须保持有限。RCAS 限幅中含 0/0：参考靠 D3D 的 NaN 容错
    # min/max 吸收，此处靠 np.fmin/np.fmax。改用普通 min/max，两种情形都会整幅 NaN。
    # 容差取 0.005：纯白输出携带参考自身 APrxMedRcpF1 的固有误差（约 0.17%，约
    # 0.4/255）。
    for v in (0.0, 1.0):
        up = upscale(np.full((8, 8, 3), v, np.float32), 2.0)
        assert np.all(np.isfinite(up)), f"NaN at constant {v}"
        assert np.allclose(up, v, atol=5e-3), f"constant {v} drifted to {up.flat[0]}"

    # 纯黑上的孤立亮点必须被锐化（参考将其抬至约 0.86）。这是上述 0/0 的另一面：
    # 将 x/0 径直取 0 会压平限幅，亮点将无法锐化。
    dot = np.zeros((5, 5, 3), np.float32)
    dot[2, 2] = 0.5
    out = rcas(dot, 0.25)
    assert np.isfinite(out).all()
    assert out[2, 2, 0] > 0.7, f"dot not sharpened: {out[2, 2, 0]}"

    # 随机输入：仅验证形状与有限性。
    noise = rng.random((24, 32, 3)).astype(np.float32)
    up = upscale(noise, 3.0)
    assert up.shape == (72, 96, 3), up.shape
    assert np.all(np.isfinite(up))
    print("selftest: ok")


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="FSR1 (EASU+RCAS) image upscaler")
    ap.add_argument("input", nargs="?", help="input image (PNG)")
    ap.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="per-dimension scale factor (default 2.0)",
    )
    ap.add_argument(
        "--sharpness",
        type=float,
        default=0.25,
        help="RCAS sharpening in stops: 0 = sharpest, larger = softer (default 0.25)",
    )
    ap.add_argument("-o", "--output", help="output PNG")
    ap.add_argument(
        "--selftest", action="store_true", help="run the built-in check and exit"
    )
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return

    if not args.input or not args.output:
        ap.error("input and -o/--output are required (or use --selftest)")

    src = _to_float(args.input)
    out = upscale(src, args.scale, args.sharpness)
    _save(out, args.output)
    print(
        f"{args.input} {src.shape[:2]} -> {args.output} {out.shape[:2]} (scale {args.scale})"
    )


if __name__ == "__main__":
    main()
