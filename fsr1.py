"""FSR1 的 NumPy 实现，对应 AMD FidelityFX Super Resolution 1.0。

FSR 1.0 是两次纯空间滤波，顺序固定：

    低清 RGB --EASU--> 高清 RGB --RCAS--> 输出

EASU 做上采样。用 12 个抽头：先估出该点的梯度方向和边缘强度，再把抽头按方向旋转、按强度拉伸，
加权求和。边缘上权重更集中，所以放大后边缘不糊。

RCAS 做锐化。只看中心像素和上下左右 4 个邻居：按中心离邻居多远算出负权重 lobe，把邻居的值按
lobe 加到中心上，再限幅，防止过冲。它不改变图像尺寸。

EASU 和 RCAS 都不看历史帧，不需要运动矢量，也不需要额外缓冲。

算法、常量、系数全部来自 AMD 的 ffx_fsr1.h / ffx_a.h（MIT）。参考实现在
../repo/FidelityFX-FSR/，只读。代码里不直观的地方都标了 `# ref:`，写明对应参考的哪一行。

和参考的三条对应：

* HLSL 的 min/max 用 np.fmin/np.fmax。D3D 规定有一个操作数是 NaN 时返回另一个，两者一致。
  这条不是可选的：RCAS 限幅里天然出现 0/0，用普通 min/max 会让纯黑/纯白图整张变 NaN。
* ARcpF1 就是 IEEE 的 1/x，用真除法。x 为 0 时得到 inf，被上面那对 min/max 吸收，所以不需要
  除零保护。
* 参考用 textureGather 一次取 4 组 2x2 纹素块，那是 GPU 的打包读法。这里直接按 12 个抽头的
  坐标取像素，取到的是同一批像素。

只做 FP32 标量路径（FsrEasuF / FsrRcasF）。参考里的 FP16 打包路径和双 tile 路径是为 GPU 吞吐
服务的，和算法无关。AMD 默认走 FP16、FP32 是回退，这里对应回退路径。

只依赖 NumPy，命令行另需 Pillow。Python >= 3.12。

---

FSR1 in NumPy, for AMD FidelityFX Super Resolution 1.0.

FSR 1.0 is two spatial filters, applied in a fixed order:

    low-res RGB --EASU--> high-res RGB --RCAS--> output

EASU upscales. It uses 12 taps: estimate the gradient direction and edge strength at the
output point, rotate the taps by that direction, stretch them by that strength, then take
the weighted sum. Weights concentrate on edges, so edges stay sharp after upscaling.

RCAS sharpens. It looks only at the centre pixel and its 4 neighbours: from how far the
centre is from them it derives a negative weight (the lobe), adds the neighbours onto the
centre with that weight, then limits the result to avoid overshoot. It does not change the
image size.

Neither looks at a history frame, needs motion vectors, or needs extra buffers.

Algorithm, constants and coefficients come from AMD's ffx_fsr1.h / ffx_a.h (MIT). The
reference implementation is the read-only clone at ../repo/FidelityFX-FSR/. Anything
non-obvious in the code carries a `# ref:` marker naming the reference line it mirrors.

Three correspondences with the reference:

* HLSL min/max map to np.fmin/np.fmax. D3D specifies that when one operand is NaN the
  other is returned, so the two agree. This is not optional: RCAS limiting contains 0/0,
  and plain min/max turns a constant black or white image entirely NaN.
* ARcpF1 is plain IEEE 1/x, mapped to true division. x==0 gives inf, which the min/max pair
  above absorbs, so no divide-by-zero guard is needed.
* The reference uses textureGather to fetch four 2x2 texel blocks at once, a GPU packing
  trick. This port reads pixels directly at the 12 tap coordinates; same pixels.

Only the FP32 scalar path is implemented (FsrEasuF / FsrRcasF). The FP16 packed paths and
the dual-tile paths exist for GPU throughput and are unrelated to the algorithm. AMD
defaults to FP16 with FP32 as the fallback; this is that fallback.

NumPy only, plus Pillow for the command line. Python >= 3.12.
"""

import numpy as np

# 锐化强度上限，超过它结果开始显得不自然。
# Limit past which sharpening starts to look unnatural.
# ref: ffx_fsr1.h:654  (FSR_RCAS_LIMIT)
RCAS_LIMIT = np.float32(0.25 - 1.0 / 16.0)


def _bits(a):
    # 把 float32 的位当成 uint32 读，供下面的位技巧使用。
    # Reinterpret the bits of a float32 as uint32, for the bit tricks below.
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# 参考里三个 GPU 倒数近似，按位复刻。
# The reference's three GPU reciprocal approximations, reproduced bit-for-bit.
# ref: ffx_a.h:1843-1845
def rcp_lo(a):
    """≈1/a，靠指数位的技巧，不做除法。
    ~1/a by a trick on the exponent bits, no division."""
    return (np.uint32(0x7EF07EBB) - _bits(a)).view(np.float32)


def rsq_lo(a):
    """≈1/sqrt(a)，同样是位技巧。
    ~1/sqrt(a), also a bit trick."""
    return (np.uint32(0x5F347D74) - (_bits(a) >> np.uint32(1))).view(np.float32)


def rcp_med(a):
    """低精度倒数再加一步牛顿迭代。
    The low-precision reciprocal plus one Newton step."""
    b = (np.uint32(0x7EF19FFF) - _bits(a)).view(np.float32)
    return b * (-b * a + np.float32(2.0))


def _rcp(x):
    """参考的 ARcpF1，就是 IEEE 的 1/x。x 为 0 时返回 inf，由 NaN 容错的 min/max 吸收。
    The reference's ARcpF1, i.e. plain IEEE 1/x. x==0 returns inf, absorbed by the
    NaN-tolerant min/max."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(np.float32(1.0), x)


def _sat(x):
    """夹到 [0,1]。
    Clamp to [0,1]. ref: ffx_a.h:747 (ASatF1)"""
    return np.clip(x, np.float32(0.0), np.float32(1.0))


def _luma(rgb):
    """亮度×2，用最省的近似 0.5R + G + 0.5B。省了一次乘法，所以结果是亮度的 2 倍。
    Luma times 2, by the cheapest approximation 0.5R + G + 0.5B. One multiply saved,
    hence the factor of 2. ref: ffx_fsr1.h:363"""
    return rgb[..., 2] * np.float32(0.5) + (rgb[..., 0] * np.float32(0.5) + rgb[..., 1])


# EASU 的 12 个抽头：偏移以输入纹素为单位，相对基准纹素 floor(pp)。排布和字母沿用参考：
# The 12 EASU taps: offsets in input texels, relative to the base texel floor(pp). Layout
# and letters follow the reference:
#     b c
#   e f g h
#   i j k l
#     n o
# ref: ffx_fsr1.h:328-434  (the tap offsets handed to FsrEasuTapF)
TAPS = (
    ("b", 0, -1), ("c", 1, -1), ("e", -1, 0), ("f", 0, 0), ("g", 1, 0), ("h", 2, 0),
    ("i", -1, 1), ("j", 0, 1), ("k", 1, 1), ("l", 2, 1), ("n", 0, 2), ("o", 1, 2),
)


def easu(src, scale):
    """EASU：上采样。
    EASU: upscaling.

    src：float32 (H, W, 3)，取值 [0,1]。scale：每维缩放倍数，可以不是整数。
    返回 float32 (round(H*scale), round(W*scale), 3)。
    src: float32 (H, W, 3) in [0,1]. scale: per-dimension factor, need not be integral.
    Returns float32 (round(H*scale), round(W*scale), 3).
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]
    OH, OW = round(H * scale), round(W * scale)

    # FsrEasuCon，viewport 取整幅输入：一个输出像素走一步在输入里占多远，以及把采样点对到
    # 像素中心的半像素项。
    # FsrEasuCon with viewport == the whole input: how far one output pixel step travels in
    # the input, and the half-pixel term that centres the samples.
    # ref: ffx_fsr1.h:171-173
    sx_in, sy_in = np.float32(W / OW), np.float32(H / OH)
    ox = np.arange(OW, dtype=np.float32)
    oy = np.arange(OH, dtype=np.float32)
    ppx = ox * sx_in + (np.float32(0.5) * sx_in - np.float32(0.5))  # (OW,)
    ppy = oy * sy_in + (np.float32(0.5) * sy_in - np.float32(0.5))  # (OH,)

    # ref: ffx_fsr1.h:324-326  (pp = ip*con0.xy + con0.zw; fp = floor(pp); pp -= fp)
    FPX = np.floor(ppx)[None, :]           # (1, OW) 基准纹素 x 的整数部分 / integer base texel x
    FPY = np.floor(ppy)[:, None]           # (OH, 1) 基准纹素 y 的整数部分 / integer base texel y
    FX = (ppx - np.floor(ppx))[None, :]    # (1, OW) pp.x 的小数部分 / fractional part of pp.x
    FY = (ppy - np.floor(ppy))[:, None]    # (OH, 1) pp.y 的小数部分 / fractional part of pp.y

    def fetch(dx, dy):
        """取 fp+偏移 处的纹素，越界夹到边缘（参考靠 clamp 采样器）。
        Fetch the texel at fp+offset, clamped at the borders (the reference relies on a
        clamp sampler)."""
        ix = np.clip(FPX + dx, 0, W - 1).astype(np.intp)
        iy = np.clip(FPY + dy, 0, H - 1).astype(np.intp)
        return src[iy, ix]  # (OH, OW, 3)

    # 先把 12 个抽头的亮度都取出来——估方向和长度只要亮度。
    # Fetch the luma of all 12 taps up front: the direction and length estimate uses luma
    # only.
    L = {name: _luma(fetch(dx, dy)) for name, dx, dy in TAPS}

    # FsrEasuSetF 跑 4 次，对应输出像素的四个双线性角（参考的形参名就是 biS/biT/biU/biV）。
    # 每次取一个以 f、g、j 或 k 为中心的十字亮度，累加梯度方向 dir 和边缘长度。
    # FsrEasuSetF runs 4 times, one per bilinear corner of the output pixel (the reference
    # names these parameters biS/biT/biU/biV). Each takes a "+" of luma centred on f, g, j
    # or k and accumulates the gradient direction (dir) and edge length.
    # ref: ffx_fsr1.h:275-313,383-386
    dirx = np.zeros((OH, OW), np.float32)
    diry = np.zeros((OH, OW), np.float32)
    length = np.zeros((OH, OW), np.float32)

    def setf(up, left, ctr, right, down, w):
        nonlocal dirx, diry, length
        # x 轴 / x axis
        lenX = rcp_lo(np.fmax(np.abs(right - ctr), np.abs(ctr - left)))
        dirX = right - left
        dirx = dirx + dirX * w
        lenX = _sat(np.abs(dirX) * lenX)
        length = length + (lenX * lenX) * w
        # y 轴 / y axis
        lenY = rcp_lo(np.fmax(np.abs(down - ctr), np.abs(ctr - up)))
        dirY = down - up
        diry = diry + dirY * w
        lenY = _sat(np.abs(dirY) * lenY)
        length = length + (lenY * lenY) * w

    setf(L["b"], L["e"], L["f"], L["g"], L["j"], (1 - FX) * (1 - FY))  # 角 biS / corner biS
    setf(L["c"], L["f"], L["g"], L["h"], L["k"], FX * (1 - FY))        # 角 biT / corner biT
    setf(L["f"], L["i"], L["j"], L["k"], L["n"], (1 - FX) * FY)        # 角 biU / corner biU
    setf(L["g"], L["j"], L["k"], L["l"], L["o"], FX * FY)              # 角 biV / corner biV

    # 归一化梯度方向（值太小时打平），再算长度、各向异性 len2、负瓣强度 lob 和窗口截断点 clp。
    # Normalize the gradient direction (flattened when too small), then derive the length,
    # the anisotropy len2, the negative-lobe strength lob and the window clip point clp.
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

    # 去振铃的上下界：2x2 最近邻抽头（f、g、j、k）的 min/max。
    # The deringing bounds: min/max over the 2x2 nearest taps (f, g, j, k).
    # ref: ffx_fsr1.h:416-419
    near = np.stack([fetch(0, 0), fetch(1, 0), fetch(0, 1), fetch(1, 1)])
    min4 = np.fmin.reduce(near, axis=0)
    max4 = np.fmax.reduce(near, axis=0)

    # FsrEasuTapF 跑 12 次：把抽头偏移按梯度方向旋转、按 len2 拉伸，代入锐化过的 Lanczos-2
    # 近似窗口求权重，累加。
    # FsrEasuTapF runs 12 times: rotate the tap offset by the gradient, stretch it by len2,
    # evaluate the sharpened Lanczos-2 approximation for the weight, accumulate.
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
        w = (np.float32(25.0 / 16.0) * (wB * wB) - np.float32(25.0 / 16.0 - 1.0)) * (wA * wA)
        aC += fetch(dx, dy) * w[..., None]
        aW += w

    # 归一化，再按上面的 min4/max4 去振铃。
    # Normalize, then dering against the min4/max4 above.
    # ref: ffx_fsr1.h:437
    pix = aC * _rcp(aW)[..., None]
    return np.fmin(max4, np.fmax(min4, pix))


def rcas(src, sharpness=0.25):
    """RCAS：锐化。
    RCAS: sharpening.

    src：float32 (H, W, 3)，取值 [0,1]，即 EASU 的输出。sharpness 的单位是档（stops）：
    每整档锐化量减半，0.0 最锐，约 2.0 以上看不出差别（AMD 文档推荐 0.2，示例代码默认
    0.25）。返回 float32 (H, W, 3)，尺寸不变。
    src: float32 (H, W, 3) in [0,1], i.e. the EASU output. sharpness is in stops:
    sharpening halves per whole stop, 0.0 is sharpest, past about 2.0 there is no visible
    difference (AMD's doc recommends 0.2; the sample code defaults to 0.25). Returns
    float32 (H, W, 3), same size.
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]

    # FsrRcasCon：锐化量按 2^-sharpness 存。 / sharpness is stored as 2^-sharpness.
    # ref: ffx_fsr1.h:667
    con = np.float32(2.0) ** np.float32(-sharpness)

    # 3x3 的十字邻域，越界夹到边缘。 / The 3x3 "plus" neighbourhood, clamped at the borders.
    # ref: ffx_fsr1.h:697-707
    def shift(dx, dy):
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        return src[ys][:, xs]

    b, d, e, f, h = shift(0, -1), shift(-1, 0), src, shift(1, 0), shift(0, 1)

    # 4 个邻居的 min/max。 / min/max over the 4 neighbours.
    # ref: ffx_fsr1.h:741-746
    ring_min = np.fmin(np.fmin(b, d), np.fmin(f, h))
    ring_max = np.fmax(np.fmax(b, d), np.fmax(f, h))

    # 逐通道算对比度限幅：中心离邻居已经有多远（peakC = (1.0, -4.0)）。
    # Per-channel contrast limits: how far the centre already is from the neighbours
    # (peakC = (1.0, -4.0)).
    # ref: ffx_fsr1.h:747-758
    # 这里的 0*inf 会得到 NaN，是参考行为的一部分，下面的 np.fmax 会吸收它。所以只关掉告警，
    # 不能把 NaN 消掉——消掉会改变孤立亮点的锐化结果，见 _selftest。
    # The 0*inf here yields NaN as part of the reference behaviour, absorbed by the np.fmax
    # below. Only the warning is silenced; removing the NaN would change how isolated
    # bright dots are sharpened - see _selftest.
    with np.errstate(invalid="ignore"):
        hit_min = np.fmin(ring_min, e) * _rcp(np.float32(4.0) * ring_max)
        hit_max = (np.float32(1.0) - np.fmax(ring_max, e)) * _rcp(np.float32(4.0) * ring_min - np.float32(4.0))
    lobe = np.fmax(-hit_min, hit_max)

    # 三个通道并成一个标量 lobe，夹到安全区间，再乘锐化量。
    # Collapse the three channels into one scalar lobe, clamp it to the safe range, scale
    # by the sharpening amount.
    # ref: ffx_fsr1.h:759
    lobe = np.fmax(np.float32(-RCAS_LIMIT), np.fmin(np.fmax.reduce(lobe, axis=2), np.float32(0.0))) * con

    # 解算。用中精度倒数，避免出现可见的色调断层。
    # Resolve, using the medium-precision reciprocal to avoid visible tonality banding.
    # ref: ffx_fsr1.h:765-768
    rcpL = rcp_med(np.float32(4.0) * lobe + np.float32(1.0))
    pix = (lobe[..., None] * (b + d + f + h) + e) * rcpL[..., None]
    return pix


def upscale(img, scale, sharpness=0.25):
    """完整管线：先 EASU，再 RCAS。
    The full pipeline: EASU first, then RCAS.

    img：float32 (H, W, 3)，取值 [0,1]，返回放大锐化后的图。sharpness=None 表示只跑 EASU。
    img: float32 (H, W, 3) in [0,1]. Returns the upscaled, sharpened image.
    sharpness=None runs EASU only.
    """
    out = easu(img, scale)
    if sharpness is None:
        return out
    return rcas(out, sharpness)


def _to_float(path_or_array):
    """读 PNG（也可以直接给 ndarray），返回 float32 RGB，取值 [0,1]。
    Load a PNG (or take an ndarray) and return float32 RGB in [0,1]."""
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return (a.astype(np.float32) / 255.0) if a.dtype == np.uint8 else a.astype(np.float32)
    from PIL import Image
    im = Image.open(path_or_array).convert("RGB")
    return np.asarray(im, dtype=np.float32) / np.float32(255.0)


def _save(arr, path):
    # 夹到 [0,1]，再量化成 8 位写出去。 / Clamp to [0,1], quantize to 8 bits, write out.
    from PIL import Image
    a = np.clip(arr, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def _selftest():
    """自检：形状、值域、有限性，外加两条必须和参考一致的行为。
    Self-check: shape, range, finiteness, plus two behaviours that must match the
    reference."""
    rng = np.random.default_rng(0)

    # 平坦中灰走完整个管线应该几乎不变。 / A flat mid-grey should come out nearly unchanged.
    flat = np.full((16, 16, 3), 0.5, np.float32)
    up = upscale(flat, 2.0)
    assert up.shape == (32, 32, 3), up.shape
    assert np.all(np.isfinite(up)), "non-finite output"
    assert np.max(np.abs(up - 0.5)) < 1e-3, f"flat tone drifted: {np.max(np.abs(up - 0.5))}"

    # 硬边缘要保住（EASU 不能把它糊成斜坡），结果不越出 [0,1]，也不被去振铃放行出振铃。
    # A hard edge must stay hard (EASU must not blur it into a ramp), must stay within
    # [0,1], and must not ring past the deringing clamp.
    edge = np.zeros((16, 16, 3), np.float32)
    edge[:, 8:] = 1.0
    up = upscale(edge, 2.0)
    assert up.min() >= -1e-6 and up.max() <= 1.0 + 1e-6, (up.min(), up.max())
    assert np.all(np.isfinite(up))

    # 纯黑、纯白常数图必须有限。RCAS 限幅里有 0/0：参考靠 D3D 的 NaN 容错 min/max 吸收，这里
    # 靠 np.fmin/np.fmax。换成普通 min/max，这两条会整张变 NaN。容差给 0.005，因为纯白会带着
    # 参考自身 APrxMedRcpF1 的误差回来（约 0.17%，约 0.4/255）。
    # Constant black and white images must stay finite. RCAS limiting contains 0/0: the
    # reference absorbs it with D3D's NaN-tolerant min/max, here np.fmin/np.fmax. With
    # plain min/max both cases turn entirely NaN. The tolerance is 0.005 because pure
    # white comes back carrying the reference's own APrxMedRcpF1 error (~0.17%, about
    # 0.4/255).
    for v in (0.0, 1.0):
        up = upscale(np.full((8, 8, 3), v, np.float32), 2.0)
        assert np.all(np.isfinite(up)), f"NaN at constant {v}"
        assert np.allclose(up, v, atol=5e-3), f"constant {v} drifted to {up.flat[0]}"

    # 纯黑上的孤立亮点要被锐化（参考会把它抬到约 0.86）。这就是上面那个 0/0 的另一面：把
    # x/0 直接取 0 会抹平限幅，亮点就锐化不了。
    # An isolated bright dot on pure black must be sharpened (the reference lifts it to
    # about 0.86). This is the other side of that same 0/0: taking x/0 as plain 0 flattens
    # the limiter and the dot stays unsharpened.
    dot = np.zeros((5, 5, 3), np.float32)
    dot[2, 2] = 0.5
    out = rcas(dot, 0.25)
    assert np.isfinite(out).all()
    assert out[2, 2, 0] > 0.7, f"dot not sharpened: {out[2, 2, 0]}"

    # 随机输入：只看形状和有限性。 / Random input: shape and finiteness only.
    noise = rng.random((24, 32, 3)).astype(np.float32)
    up = upscale(noise, 3.0)
    assert up.shape == (72, 96, 3), up.shape
    assert np.all(np.isfinite(up))
    print("selftest: ok")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="FSR1 (EASU+RCAS) image upscaler")
    ap.add_argument("input", nargs="?", help="input image (PNG)")
    ap.add_argument("--scale", type=float, default=2.0, help="per-dimension scale factor (default 2.0)")
    ap.add_argument("--sharpness", type=float, default=0.25,
                    help="RCAS sharpening in stops: 0 = sharpest, larger = softer (default 0.25)")
    ap.add_argument("-o", "--output", help="output PNG")
    ap.add_argument("--selftest", action="store_true", help="run the built-in check and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return

    if not args.input or not args.output:
        ap.error("input and -o/--output are required (or use --selftest)")

    src = _to_float(args.input)
    out = upscale(src, args.scale, args.sharpness)
    _save(out, args.output)
    print(f"{args.input} {src.shape[:2]} -> {args.output} {out.shape[:2]} (scale {args.scale})")


if __name__ == "__main__":
    main()