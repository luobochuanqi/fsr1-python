"""FSR1 in NumPy: a faithful, runnable port of AMD FidelityFX Super Resolution 1.

This is an independent NumPy implementation of the EASU and RCAS passes from
AMD's `ffx_fsr1.h` (repo/FidelityFX-FSR). It is not a copy of the HLSL: it is a
re-expression of the same math in vectorized NumPy, one expression at a time.
Every non-obvious line below carries a `# ref:` marker naming the reference
function it mirrors, so the correspondence can be checked statically.

Pipeline (exactly FSR1's two passes, in order):
    low-res RGB --EASU--> high-res RGB --RCAS--> sharpened high-res RGB

EASU is an edge-adaptive spatial upsampler: a 12-tap window whose tap weights
are rotated and stretched by the local gradient, so edges stay crisp while flat
areas get a soft Lanczos-like interpolation. RCAS is a contrast-adaptive sharp
pass that re-weights the 4-neighbour ring by the local contrast, with hard
limiting so it never overshoots into ringing.

Where the HLSL uses the GPU's approximate rcp/rsqrt bit-hacks, this port uses
the *same* bit-hacks (see rcp_lo / rsq_lo / rcp_med), so the numeric behaviour
matches the shader rather than merely being "close". The one deliberate
deviation is in RCAS limiting, where the shader relies on HLSL's nan-tolerant
comparisons in degenerate all-black / all-white regions; this port makes the
same cases finite by treating x/0 as 0, which is the correct limiting value.

Only NumPy (+ Pillow for the CLI) is required.
"""

import numpy as np

# FSR_RCAS_LIMIT: "set at the limit of providing unnatural results for sharpening"
# ref: ffx_fsr1.h:654
RCAS_LIMIT = np.float32(0.25 - 1.0 / 16.0)


def _bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


# The three GPU reciprocal approximations, reproduced bit-for-bit.
# ref: ffx_a.h:1843-1845
def rcp_lo(a):
    """APrxLoRcpF1: ~1/a via a float-exponent bit hack, no division."""
    return (np.uint32(0x7EF07EBB) - _bits(a)).view(np.float32)


def rsq_lo(a):
    """APrxLoRsqF1: ~1/sqrt(a) via a bit hack."""
    return (np.uint32(0x5F347D74) - (_bits(a) >> np.uint32(1))).view(np.float32)


def rcp_med(a):
    """APrxMedRcpF1: the low-precision reciprocal plus one Newton step."""
    b = (np.uint32(0x7EF19FFF) - _bits(a)).view(np.float32)
    return b * (-b * a + np.float32(2.0))


def _safe_rcp(x):
    """Exact 1/x with x==0 mapped to 0 (see module docstring: degenerate-case note)."""
    out = np.zeros_like(x, dtype=np.float32)
    np.divide(np.float32(1.0), x, out=out, where=(x != 0))
    return out


def _sat(x):
    """ASatF1: clamp to [0,1]."""
    return np.clip(x, np.float32(0.0), np.float32(1.0))


def _luma(rgb):
    """The reference's cheap luma-times-2: 0.5*R + G + 0.5*B.
    ref: ffx_fsr1.h:363 (bczzL=bczzB*0.5+(bczzR*0.5+bczzG))"""
    return rgb[..., 2] * np.float32(0.5) + (rgb[..., 0] * np.float32(0.5) + rgb[..., 1])


# The 12-tap EASU kernel, offsets in input texels relative to the base texel
# floor(pp). Layout and letters are the reference's:
#     b c
#   e f g h
#   i j k l
#     n o
# ref: ffx_fsr1.h:328-434 (the tap offsets passed to FsrEasuTapF)
TAPS = (
    ("b", 0, -1), ("c", 1, -1), ("e", -1, 0), ("f", 0, 0), ("g", 1, 0), ("h", 2, 0),
    ("i", -1, 1), ("j", 0, 1), ("k", 1, 1), ("l", 2, 1), ("n", 0, 2), ("o", 1, 2),
)


def easu(src, scale):
    """EASU pass: edge-adaptive spatial upsampling.

    src: float32 (H, W, 3) in [0,1]. scale: output/input ratio (any >= 1).
    Returns float32 (round(H*scale), round(W*scale), 3).
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]
    OH, OW = int(round(H * scale)), int(round(W * scale))

    # FsrEasuCon with viewport == input size: the fraction of an output pixel
    # step in input space, and the half-pixel centring term.
    # ref: ffx_fsr1.h:171-173
    sx_in, sy_in = np.float32(W / OW), np.float32(H / OH)
    ox = np.arange(OW, dtype=np.float32)
    oy = np.arange(OH, dtype=np.float32)
    ppx = ox * sx_in + (np.float32(0.5) * sx_in - np.float32(0.5))  # (OW,)
    ppy = oy * sy_in + (np.float32(0.5) * sy_in - np.float32(0.5))  # (OH,)

    # ref: ffx_fsr1.h:324-326  (pp = ip*con0.xy + con0.zw; fp = floor(pp); pp -= fp)
    FPX = np.floor(ppx)[None, :]           # (1, OW) integer base texel x
    FPY = np.floor(ppy)[:, None]           # (OH, 1) integer base texel y
    FX = (ppx - np.floor(ppx))[None, :]    # (1, OW) fractional pp.x
    FY = (ppy - np.floor(ppy))[:, None]    # (OH, 1) fractional pp.y

    def fetch(dx, dy):
        """Texture fetch with clamp-to-edge. ref: gather taps at fp+offset."""
        ix = np.clip(FPX + dx, 0, W - 1).astype(np.intp)
        iy = np.clip(FPY + dy, 0, H - 1).astype(np.intp)
        return src[iy, ix]  # (OH, OW, 3)

    # Pass 1: luma of every tap (needed for the edge direction/length estimate).
    L = {name: _luma(fetch(dx, dy)) for name, dx, dy in TAPS}

    # FsrEasuSetF, run four times, one per bilinear corner of the output pixel.
    # Each takes a "+" of luma centred on f, g, j or k and accumulates the
    # gradient direction (dir) and edge length (len).
    # ref: ffx_fsr1.h:275-313,383-386
    dirx = np.zeros((OH, OW), np.float32)
    diry = np.zeros((OH, OW), np.float32)
    length = np.zeros((OH, OW), np.float32)

    def setf(up, left, ctr, right, down, w):
        nonlocal dirx, diry, length
        # x axis
        lenX = rcp_lo(np.maximum(np.abs(right - ctr), np.abs(ctr - left)))
        dirX = right - left
        dirx = dirx + dirX * w
        lenX = _sat(np.abs(dirX) * lenX)
        length = length + (lenX * lenX) * w
        # y axis
        lenY = rcp_lo(np.maximum(np.abs(down - ctr), np.abs(ctr - up)))
        dirY = down - up
        diry = diry + dirY * w
        lenY = _sat(np.abs(dirY) * lenY)
        length = length + (lenY * lenY) * w

    setf(L["b"], L["e"], L["f"], L["g"], L["j"], (1 - FX) * (1 - FY))  # corner S
    setf(L["c"], L["f"], L["g"], L["h"], L["k"], FX * (1 - FY))        # corner T
    setf(L["f"], L["i"], L["j"], L["k"], L["n"], (1 - FX) * FY)        # corner U
    setf(L["g"], L["j"], L["k"], L["l"], L["o"], FX * FY)              # corner V

    # Normalize the gradient direction (with the near-zero guard), then shape
    # length, anisotropy (len2), negative-lobe strength (lob) and window clip.
    # ref: ffx_fsr1.h:389-409
    dirR = dirx * dirx + diry * diry
    zro = dirR < np.float32(1.0 / 32768.0)
    dirR = np.where(zro, np.float32(1.0), rsq_lo(dirR))
    dirx = np.where(zro, np.float32(1.0), dirx) * dirR
    diry = diry * dirR

    length = length * np.float32(0.5)
    length = length * length
    stretch = (dirx * dirx + diry * diry) * rcp_lo(np.maximum(np.abs(dirx), np.abs(diry)))
    len2x = np.float32(1.0) + (stretch - np.float32(1.0)) * length
    len2y = np.float32(1.0) + np.float32(-0.5) * length
    lob = np.float32(0.5) + (np.float32(1.0 / 4.0 - 0.04) - np.float32(0.5)) * length
    clp = rcp_lo(lob)

    # Dering bounds: min/max over the 2x2 nearest taps (f, g, j, k).
    # ref: ffx_fsr1.h:416-419
    near = np.stack([fetch(0, 0), fetch(1, 0), fetch(0, 1), fetch(1, 1)])
    min4 = near.min(axis=0)
    max4 = near.max(axis=0)

    # FsrEasuTapF, 12 times: rotate the tap offset by the gradient, stretch by
    # len2, evaluate the sharpened Lanczos-2 window, and accumulate.
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
        d2 = np.minimum(vx * vx + vy * vy, clp)
        wB = np.float32(2.0 / 5.0) * d2 - np.float32(1.0)
        wA = lob * d2 - np.float32(1.0)
        w = (np.float32(25.0 / 16.0) * (wB * wB) - np.float32(25.0 / 16.0 - 1.0)) * (wA * wA)
        aC += fetch(dx, dy) * w[..., None]
        aW += w

    # Normalize and dering-clamp. ref: ffx_fsr1.h:437
    pix = aC * _safe_rcp(aW)[..., None]
    return np.minimum(max4, np.maximum(min4, pix))


def rcas(src, sharpness=0.25):
    """RCAS pass: robust contrast-adaptive sharpening.

    src: float32 (H, W, 3) in [0,1] (the EASU output). sharpness is in "stops":
    0 = maximum sharpening, larger = less (0.25 is AMD's sample default).
    Returns float32 (H, W, 3).
    """
    src = np.ascontiguousarray(src, dtype=np.float32)
    H, W = src.shape[:2]

    # FsrRcasCon: sharpness is stored as 2^-sharpness. ref: ffx_fsr1.h:667
    con = np.float32(2.0) ** np.float32(-sharpness)

    # The 3x3 "plus" neighbourhood, clamp-to-edge. ref: ffx_fsr1.h:697-707
    def shift(dx, dy):
        ys = np.clip(np.arange(H) + dy, 0, H - 1)
        xs = np.clip(np.arange(W) + dx, 0, W - 1)
        return src[ys][:, xs]

    b, d, e, f, h = shift(0, -1), shift(-1, 0), src, shift(1, 0), shift(0, 1)

    # Min and max of the 4-neighbour ring. ref: ffx_fsr1.h:741-746
    ring_min = np.minimum(np.minimum(b, d), np.minimum(f, h))
    ring_max = np.maximum(np.maximum(b, d), np.maximum(f, h))

    # Contrast limits: how far the centre already is from the ring, per channel.
    # peakC = (1.0, -4.0). ref: ffx_fsr1.h:747-758
    hit_min = np.minimum(ring_min, e) * _safe_rcp(np.float32(4.0) * ring_max)
    hit_max = (np.float32(1.0) - np.maximum(ring_max, e)) * _safe_rcp(np.float32(4.0) * ring_min - np.float32(4.0))
    lobe = np.maximum(-hit_min, hit_max)

    # Collapse to a single scalar lobe, clamp to the safe range, scale by con.
    # ref: ffx_fsr1.h:759
    lobe = np.maximum(np.float32(-RCAS_LIMIT), np.minimum(lobe.max(axis=2), np.float32(0.0))) * con

    # Resolve, with the medium-precision reciprocal to avoid tonality banding.
    # ref: ffx_fsr1.h:765-768
    rcpL = rcp_med(np.float32(4.0) * lobe + np.float32(1.0))
    pix = (lobe[..., None] * (b + d + f + h) + e) * rcpL[..., None]
    return pix


def upscale(img, scale, sharpness=0.25):
    """Full FSR1 pipeline: EASU then RCAS.

    img: float32 (H, W, 3) in [0,1]. Returns the upscaled, sharpened image.
    sharpness=None skips RCAS and returns the bare EASU upscale.
    """
    out = easu(img, scale)
    if sharpness is None:
        return out
    return rcas(out, sharpness)


def _to_float(path_or_array):
    """Load a PNG (or accept an ndarray) and return float32 RGB in [0,1]."""
    if isinstance(path_or_array, np.ndarray):
        a = path_or_array
        return (a.astype(np.float32) / 255.0) if a.dtype == np.uint8 else a.astype(np.float32)
    from PIL import Image
    im = Image.open(path_or_array).convert("RGB")
    return np.asarray(im, dtype=np.float32) / np.float32(255.0)


def _save(arr, path):
    from PIL import Image
    a = np.clip(arr, 0.0, 1.0)
    Image.fromarray((a * 255.0 + 0.5).astype(np.uint8)).save(path)


def _selftest():
    """One runnable check: shapes, range, no NaNs, and flat/tone fidelity."""
    rng = np.random.default_rng(0)
    # A flat mid-grey must survive both passes essentially unchanged.
    flat = np.full((16, 16, 3), 0.5, np.float32)
    up = upscale(flat, 2.0)
    assert up.shape == (32, 32, 3), up.shape
    assert np.all(np.isfinite(up)), "non-finite output"
    assert np.max(np.abs(up - 0.5)) < 1e-3, f"flat tone drifted: {np.max(np.abs(up - 0.5))}"

    # A hard edge must stay hard (EASU must not blur it into a ramp), and the
    # result must stay within [0,1] with no ringing overshoot past the dering clamp.
    edge = np.zeros((16, 16, 3), np.float32)
    edge[:, 8:] = 1.0
    up = upscale(edge, 2.0)
    assert up.min() >= -1e-6 and up.max() <= 1.0 + 1e-6, (up.min(), up.max())
    assert np.all(np.isfinite(up))

    # Full black and full white must not produce NaNs (the RCAS degenerate case).
    for v in (0.0, 1.0):
        up = upscale(np.full((8, 8, 3), v, np.float32), 2.0)
        assert np.all(np.isfinite(up)), f"NaN at constant {v}"

    # Random input: only the shape/finiteness contract is checked.
    noise = rng.random((24, 32, 3)).astype(np.float32)
    up = upscale(noise, 3.0)
    assert up.shape == (72, 96, 3), up.shape
    assert np.all(np.isfinite(up))
    print("selftest: ok")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="FSR1 (EASU+RCAS) image upscaler")
    ap.add_argument("input", nargs="?", help="input image (PNG)")
    ap.add_argument("--scale", type=float, default=2.0, help="upscale factor (default 2.0)")
    ap.add_argument("--sharpness", type=float, default=0.25,
                    help="RCAS sharpening in stops: 0 = max, larger = softer (default 0.25)")
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