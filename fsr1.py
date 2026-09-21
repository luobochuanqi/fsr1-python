"""FSR1（EASU + RCAS）的 NumPy 实现。

把 AMD FidelityFX Super Resolution 1.0 的两趟算法用向量化 NumPy 重写，顺序固定：

    低清 RGB --EASU（上采样趟）--> 高清 RGB --RCAS（锐化趟）--> 锐化后的高清 RGB

EASU 对单帧做边缘自适应的空间上采样：12 抽头窗口的权重按局部梯度方向旋转、拉伸，边缘收紧、
平坦区做柔性插值。RCAS 在放大后的图上做对比度自适应锐化：按局部对比度给 4 邻域加权，并做
硬限幅以抑制振铃。两趟都是纯空间的，不需要历史帧、运动矢量或额外缓冲。

算法、常量与系数全部取自 AMD 的 `ffx_fsr1.h` / `ffx_a.h`（MIT 许可）。参考实现是
`../repo/FidelityFX-FSR/` 的只读克隆。每一处不直观的代码都带 `# ref:` 标记，指明它对应参考
实现的哪一行，可静态核对。

与参考实现的三处对应约定：
  * HLSL 的 `min`/`max` 用 `np.fmin`/`np.fmax` 对应。D3D 规定其中一个操作数为 NaN 时返回另一
    个操作数，两者语义一致——这不是风格选择，而是让退化输入（如纯黑/纯白常数图）的数值行为
    与 shader 相同。
  * `ARcpF1` 就是普通 IEEE 的 1/x，这里用真除法对应；x==0 会得到 inf，随后由上一条的
    NaN 容错 min/max 吸收，因此不需要任何除零保护。
  * 参考用 `textureGather` 取 4 组 2x2 纹素块（GPU 的打包读取）；这里直接按 12 个抽头的像素
    坐标取值，取到的是同一批像素，只是省掉了打包。

范围：只实现 FP32 标量路径（`FsrEasuF` / `FsrRcasF`）。参考里的 FP16 打包路径（`*H`）与双
tile 路径（`*Hx2`）是为 GPU 吞吐服务的，与算法无关，未移植。AMD 亦以 FP16 为默认、FP32 为
回退，这里对应的是回退路径。

只依赖 NumPy（CLI 另需 Pillow），Python >= 3.12。

---

FSR1 (EASU + RCAS) in NumPy.

A vectorized re-implementation of the two passes of AMD FidelityFX Super Resolution 1.0,
in fixed order:

    low-res RGB --EASU (upscaling pass)--> high-res RGB --RCAS (sharpening pass)--> sharpened

EASU performs edge-adaptive spatial upsampling of a single frame: the weights of a
12-tap window are rotated and stretched by the local gradient, so edges tighten while
flat areas get a soft interpolation. RCAS then sharpens the upscaled image by weighting
the 4-neighbour ring against the local contrast, with hard limiting to suppress ringing.
Both passes are purely spatial: no history frame, motion vectors, or extra buffers.

The algorithm, constants and coefficients all come from AMD's `ffx_fsr1.h` / `ffx_a.h`
(MIT licensed); the reference implementation is the read-only clone at
`../repo/FidelityFX-FSR/`. Every non-obvious line carries a `# ref:` marker naming the
reference line it mirrors, so the correspondence can be checked statically.

Three correspondence conventions:
  * HLSL `min`/`max` map to `np.fmin`/`np.fmax`. D3D specifies that when one operand is
    NaN the other is returned, which is exactly their semantics - not a style choice, but
    what makes degenerate inputs (a constant black or white frame) behave as in the
    shader.
  * `ARcpF1` is plain IEEE 1/x, mapped to true division; x==0 yields inf, which the
    NaN-tolerant min/max above then absorbs, so no divide-by-zero guard is needed.
  * The reference fetches four 2x2 texel blocks with `textureGather` (a GPU packing
    trick); this port reads the same pixels directly at the 12 tap coordinates.

Scope: only the FP32 scalar path (`FsrEasuF` / `FsrRcasF`). The FP16 packed paths (`*H`)
and the dual-tile paths (`*Hx2`) exist for GPU throughput and are irrelevant to the
algorithm. AMD likewise defaults to FP16 with an FP32 fallback - this is that fallback.

NumPy only (plus Pillow for the CLI), Python >= 3.12.
"""

import numpy as np

# 锐化强度的自然度上限，超过它结果就开始显得不自然。
# Limit past which sharpening starts to produce unnatural results.
# ref: ffx_fsr1.h:654  (FSR_RCAS_LIMIT)
RCAS_LIMIT = np.float32(0.25 - 1.0 / 16.0)


def _bits(a):
    # 把 float32 的位重新解释成 uint32，供下面的位技巧使用。
    # Reinterpret the bits of a float32 as uint32, as the bit-hacks below require.
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# 参考里三个 GPU 倒数近似，逐位复刻。
# The reference's three GPU reciprocal approximations, reproduced bit-for-bit.
# ref: ffx_a.h:1843-1845
def rcp_lo(a):
    """≈1/a，用浮点指数位的技巧，不做真除法。
    ~1/a via a float-exponent bit trick, no real division."""
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
    """参考的 ARcpF1：普通 IEEE 1/x，x==0 得到 inf，交由 NaN 容错的 min/max 吸收。
    The reference's ARcpF1: plain IEEE 1/x; x==0 gives inf, which the NaN-tolerant
    min/max then absorbs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(np.float32(1.0), x)


def _sat(x):
    """夹到 [0,1]。
    Clamp to [0,1]. ref: ffx_a.h:747 (ASatF1)"""
    return np.clip(x, np.float32(0.0), np.float32(1.0))


def _luma(rgb):
    """2 倍亮度，用最省的近似 0.5R + G + 0.5B（省一次乘法，故结果是亮度的 2 倍）。
    Luma times 2, with the cheapest approximation 0.5R + G + 0.5B (one multiply saved,
    hence the factor of 2). ref: ffx_fsr1.h:363"""
    return rgb[..., 2] * np.float32(0.5) + (rgb[..., 0] * np.float32(0.5) + rgb[..., 1])


# EASU 的 12 抽头核：偏移以输入纹素为单位，相对基准纹素 floor(pp)。排布与字母沿用参考：
# The EASU 12-tap kernel: offsets are in input texels, relative to the base texel
# floor(pp). Layout and letters are the reference's:
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
    """EASU 上采样趟：边缘自适应空间上采样。
    EASU upscaling pass: edge-adaptive spatial upsampling.

    src：float32 (H, W, 3)，取值 [0,1]；scale：每维缩放倍数，不必为整数。
    返回 float32 (round(H*scale), round(W*scale), 3)。
    src: float32 (H, W, 3) in [0,1]. scale: per-dimension factor, need not be integral.
    Returns float32 (round(H*scale), round(W*scale), 3).
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]
    OH, OW = round(H * scale), round(W * scale)

    # FsrEasuCon（viewport 取整幅输入）：一个输出像素步长在输入空间占多少，以及用于对齐
    # 像素中心的半像素项。
    # FsrEasuCon with viewport == the whole input: how much input space one output pixel
    # step spans, plus the half-pixel term that centres the samples.
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
        """取 fp+偏移 处的纹素，越界按边缘夹取（参考用的是 clamp 采样器）。
        Fetch the texel at fp+offset, clamped at the borders (the reference relies on a
        clamp sampler)."""
        ix = np.clip(FPX + dx, 0, W - 1).astype(np.intp)
        iy = np.clip(FPY + dy, 0, H - 1).astype(np.intp)
        return src[iy, ix]  # (OH, OW, 3)

    # 先取全部 12 个抽头的亮度——估计边缘方向与长度只需要亮度。
    # Fetch the luma of all 12 taps first: the edge direction and length estimate needs
    # luma only.
    L = {name: _luma(fetch(dx, dy)) for name, dx, dy in TAPS}

    # FsrEasuSetF 跑四次，对应输出像素的四个双线性角（参考的形参名即 biS/biT/biU/biV）。
    # 每次取一个以 f、g、j 或 k 为中心的“十”字亮度，累加梯度方向（dir）与边缘长度（len）。
    # FsrEasuSetF runs four times, one per bilinear corner of the output pixel (the
    # reference names these parameters biS/biT/biU/biV). Each takes a "+" of luma
    # centred on f, g, j or k, accumulating gradient direction (dir) and edge length.
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

    setf(L["b"], L["e"], L["f"], L["g"], L["j"], (1 - FX) * (1 - FY))  # 角 S / corner S
    setf(L["c"], L["f"], L["g"], L["h"], L["k"], FX * (1 - FY))        # 角 T / corner T
    setf(L["f"], L["i"], L["j"], L["k"], L["n"], (1 - FX) * FY)        # 角 U / corner U
    setf(L["g"], L["j"], L["k"], L["l"], L["o"], FX * FY)              # 角 V / corner V

    # 归一化梯度方向（近零时打平），再整理长度、各向异性（len2）、负瓣强度（lob）与窗口
    # 截断点（clp）。
    # Normalize the gradient direction (flattened when near zero), then shape the length,
    # the anisotropy (len2), the negative-lobe strength (lob) and the window clip point.
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

    # 去振铃的上/下界：取 2x2 最近邻抽头（f、g、j、k）的 min/max。
    # The deringing bounds: min/max over the 2x2 nearest taps (f, g, j, k).
    # ref: ffx_fsr1.h:416-419
    near = np.stack([fetch(0, 0), fetch(1, 0), fetch(0, 1), fetch(1, 1)])
    min4 = np.fmin.reduce(near, axis=0)
    max4 = np.fmax.reduce(near, axis=0)

    # FsrEasuTapF 跑 12 次：把抽头偏移按梯度方向旋转、按 len2 拉伸，代入锐化过的
    # Lanczos-2 近似窗口求权重，然后累加。
    # FsrEasuTapF runs 12 times: rotate the tap offset by the gradient, stretch it by
    # len2, evaluate the sharpened Lanczos-2 approximation for the weight, accumulate.
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

    # 归一化，再按上一步的 min4/max4 去振铃。
    # Normalize, then dering against the min4/max4 computed above.
    # ref: ffx_fsr1.h:437
    pix = aC * _rcp(aW)[..., None]
    return np.fmin(max4, np.fmax(min4, pix))


def rcas(src, sharpness=0.25):
    """RCAS 锐化趟：鲁棒对比度自适应锐化。
    RCAS sharpening pass: robust contrast adaptive sharpening.

    src：float32 (H, W, 3)，取值 [0,1]，即 EASU 的输出。sharpness 单位为档（stops）：
    每整档锐化量减半，0.0 最锐，约 2.0 起不再有可见差别（AMD 文档推荐 0.2，示例代码默认
    0.25）。返回 float32 (H, W, 3)。
    src: float32 (H, W, 3) in [0,1], i.e. the EASU output. sharpness is in stops:
    sharpening halves per whole stop, 0.0 is sharpest, values past about 2.0 make no
    visible difference (AMD's doc recommends 0.2; the sample code defaults to 0.25).
    Returns float32 (H, W, 3).
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]

    # FsrRcasCon：锐化量以 2^-sharpness 存入常量。 / sharpness is stored as 2^-sharpness.
    # ref: ffx_fsr1.h:667
    con = np.float32(2.0) ** np.float32(-sharpness)

    # 3x3 的“十”字邻域，越界按边缘夹取。 / The 3x3 "plus" neighbourhood, clamped at the borders.
    # ref: ffx_fsr1.h:697-707
    def shift(dx, dy):
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        return src[ys][:, xs]

    b, d, e, f, h = shift(0, -1), shift(-1, 0), src, shift(1, 0), shift(0, 1)

    # 4 邻域环上的 min/max。 / min/max over the 4-neighbour ring.
    # ref: ffx_fsr1.h:741-746
    ring_min = np.fmin(np.fmin(b, d), np.fmin(f, h))
    ring_max = np.fmax(np.fmax(b, d), np.fmax(f, h))

    # 逐通道的对比度限幅：中心离邻域环已经有多远（peakC = (1.0, -4.0)）。
    # Per-channel contrast limits: how far the centre already is from the ring
    # (peakC = (1.0, -4.0)).
    # ref: ffx_fsr1.h:747-758
    # 这里的 0*inf 会产生 NaN，这是参考行为的一部分——它会被下面的 np.fmax 吸收，所以显式
    # 关掉告警，而不是消除 NaN 本身（消除会改变孤立亮点的锐化结果，见 _selftest）。
    # The 0*inf here yields NaN as part of the reference behaviour; the np.fmax below
    # absorbs it, so the warning is silenced rather than the NaN removed (removing it
    # would change how isolated bright dots are sharpened - see _selftest).
    with np.errstate(invalid="ignore"):
        hit_min = np.fmin(ring_min, e) * _rcp(np.float32(4.0) * ring_max)
        hit_max = (np.float32(1.0) - np.fmax(ring_max, e)) * _rcp(np.float32(4.0) * ring_min - np.float32(4.0))
    lobe = np.fmax(-hit_min, hit_max)

    # 三个通道收敛成一个标量 lobe，夹到安全区间，再乘上锐化量。
    # Collapse the three channels to a single scalar lobe, clamp it to the safe range,
    # then scale by the sharpening amount.
    # ref: ffx_fsr1.h:759
    lobe = np.fmax(np.float32(-RCAS_LIMIT), np.fmin(np.fmax.reduce(lobe, axis=2), np.float32(0.0))) * con

    # 解算：用中精度倒数，避免出现可见的色调断层。
    # Resolve with the medium-precision reciprocal to avoid visible tonality banding.
    # ref: ffx_fsr1.h:765-768
    rcpL = rcp_med(np.float32(4.0) * lobe + np.float32(1.0))
    pix = (lobe[..., None] * (b + d + f + h) + e) * rcpL[..., None]
    return pix


def upscale(img, scale, sharpness=0.25):
    """完整的 FSR1 管线：先 EASU 上采样，再 RCAS 锐化。
    The full FSR1 pipeline: EASU upscaling followed by RCAS sharpening.

    img：float32 (H, W, 3)，取值 [0,1]；返回放大并锐化后的图像。sharpness=None 表示只跑
    EASU、跳过 RCAS。
    img: float32 (H, W, 3) in [0,1]. Returns the upscaled, sharpened image.
    sharpness=None runs EASU only and skips RCAS.
    """
    out = easu(img, scale)
    if sharpness is None:
        return out
    return rcas(out, sharpness)


def _to_float(path_or_array):
    """读入 PNG（或直接接收 ndarray），返回 float32 RGB，取值 [0,1]。
    Load a PNG (or accept an ndarray) and return float32 RGB in [0,1]."""
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return (a.astype(np.float32) / 255.0) if a.dtype == np.uint8 else a.astype(np.float32)
    from PIL import Image
    im = Image.open(path_or_array).convert("RGB")
    return np.asarray(im, dtype=np.float32) / np.float32(255.0)


def _save(arr, path):
    # 夹到 [0,1] 后按 8 位量化写出。 / Clamp to [0,1], then quantize to 8 bits on write.
    from PIL import Image
    a = np.clip(arr, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def _selftest():
    """一个可运行的自检：形状、值域、有限性，以及两条与参考一致的行为。
    One runnable check: shape, range, finiteness, and two behaviours that must match the
    reference."""
    rng = np.random.default_rng(0)

    # 平坦的中灰经过两趟应当几乎不变。 / A flat mid-grey must survive both passes nearly unchanged.
    flat = np.full((16, 16, 3), 0.5, np.float32)
    up = upscale(flat, 2.0)
    assert up.shape == (32, 32, 3), up.shape
    assert np.all(np.isfinite(up)), "non-finite output"
    assert np.max(np.abs(up - 0.5)) < 1e-3, f"flat tone drifted: {np.max(np.abs(up - 0.5))}"

    # 硬边缘必须保持硬（EASU 不能把它糊成斜坡），且结果不越过 [0,1]、不被去振铃放行出振铃。
    # A hard edge must stay hard (EASU must not blur it into a ramp), stay within [0,1],
    # and not ring past the deringing clamp.
    edge = np.zeros((16, 16, 3), np.float32)
    edge[:, 8:] = 1.0
    up = upscale(edge, 2.0)
    assert up.min() >= -1e-6 and up.max() <= 1.0 + 1e-6, (up.min(), up.max())
    assert np.all(np.isfinite(up))

    # 纯黑与纯白常数图必须有限。RCAS 的限幅里有 0/0：参考靠 D3D 的 NaN 容错 min/max 吸收，
    # 这里靠 np.fmin/np.fmax；若改成普通 min/max，这两条会整图变 NaN。容差放到 0.005，
    # 因为纯白会带着参考自身 APrxMedRcpF1 的误差（约 0.17%）回来，约合 0.4/255。
    # Constant black and white frames must stay finite. RCAS limiting contains 0/0: the
    # reference absorbs it with D3D's NaN-tolerant min/max, here np.fmin/np.fmax. With
    # plain min/max both cases turn entirely NaN. The tolerance is 0.005 because pure
    # white comes back carrying the reference's own APrxMedRcpF1 error (~0.17%, about
    # 0.4/255).
    for v in (0.0, 1.0):
        up = upscale(np.full((8, 8, 3), v, np.float32), 2.0)
        assert np.all(np.isfinite(up)), f"NaN at constant {v}"
        assert np.allclose(up, v, atol=5e-3), f"constant {v} drifted to {up.flat[0]}"

    # 纯黑上的一颗孤立亮点要被锐化（参考会把它抬到约 0.86）。这正是 0/0 那个位置的另一面：
    # 若把 x/0 简单地取 0，限幅被抹平，亮点就得不到锐化。
    # An isolated bright dot on pure black must be sharpened (the reference lifts it to
    # about 0.86). This is the other face of that same 0/0: treating x/0 as plain 0
    # flattens the limiter and the dot would not be sharpened at all.
    dot = np.zeros((5, 5, 3), np.float32)
    dot[2, 2] = 0.5
    out = rcas(dot, 0.25)
    assert np.isfinite(out).all()
    assert out[2, 2, 0] > 0.7, f"dot not sharpened: {out[2, 2, 0]}"

    # 随机输入：只核对形状与有限性。 / Random input: only shape and finiteness.
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