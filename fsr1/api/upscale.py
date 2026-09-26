"""Vercel Function：接收上传图片，跑 FSR1 超分，返回 PNG。

接口：POST /api/upscale?scale=2.0&sharpness=0.25
  请求体 = 原图原始字节（PNG/JPEG，无需 multipart）
成功返回 200 + image/png；失败返回 4xx + text/plain。

部署形态：本目录（fsr1/）是自包含部署单元（Vercel 项目 Root Directory =
fsr1）。页面静态资源（web/dist 构建产物）由 Vercel CDN 直接服务（vercel.json
的 outputDirectory），仅 /api/upscale 路由到本函数。本地 serve.py 复用同一份
WSGI app，并自管 dist 静态文件，所见即所得。函数以部署根为 cwd，可直接
import 同级的 fsr1.py（依赖见 requirements.txt）。冷启动导入 NumPy 约
+0.5s。输入最大边 512px：NumPy 路径在该尺寸约 1.2s，时限余量充足。
"""

import io
import mimetypes
from pathlib import Path

import numpy as np
from PIL import Image

from fsr1 import upscale

MAX_EDGE = 512  # 输入最大边；NumPy 路径在 512 时约 1.2s
MAX_PIXELS = MAX_EDGE * MAX_EDGE * 4  # 4M px，防解压炸弹
MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB

DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def app(environ, start_response):
    """WSGI 入口：/api/upscale 归函数，其余 GET 由本地静态服务应答。"""
    if environ["PATH_INFO"] == "/api/upscale":
        return _upscale(environ, start_response)
    if environ["REQUEST_METHOD"] == "GET":
        rel = environ["PATH_INFO"].lstrip("/") or "index.html"
        return _static(rel, environ, start_response)
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"not found"]


def _static(rel, environ, start_response):
    """本地预览的静态文件服务（部署时该职责在 CDN）。

    resolve 后必须仍在 DIST 内，天然免疫路径穿越。"""
    p = (DIST / rel).resolve()
    if not p.is_relative_to(DIST) or not p.is_file():
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    body = p.read_bytes()
    # HTML 不缓存（改版即生效），其余短缓存；示例图路径含语义版本。
    cache = "no-cache" if ctype.startswith("text/html") else "public, max-age=3600"
    start_response("200 OK", [
        ("Content-Type", ctype),
        ("Content-Length", str(len(body))),
        ("Cache-Control", cache),
    ])
    return [body]


def _upscale(environ, start_response):
    """超分端点。"""

    def err(msg, code="400 Bad Request"):
        start_response(code, [("Content-Type", "text/plain; charset=utf-8")])
        return [msg.encode()]

    if environ["REQUEST_METHOD"] != "POST":
        return err("POST only", "405 Method Not Allowed")
    length = int(environ.get("CONTENT_LENGTH", 0) or 0)
    if length > MAX_UPLOAD:
        return err("upload too large", "413 Payload Too Large")
    # NOTE: 必须按 CONTENT_LENGTH 精确读取，不能对 wsgi.input 裸 read()——
    # keep-alive 连接上没有 EOF 可等，裸 read() 会永久阻塞。
    body = environ["wsgi.input"].read(length)

    try:
        im = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception:  # noqa: BLE001
        return err("not a valid image")
    if im.width * im.height > MAX_PIXELS:
        return err(
            f"image too large: max {MAX_EDGE}px per edge / {MAX_PIXELS} px total"
        )

    from urllib.parse import parse_qsl

    q = dict(parse_qsl(environ.get("QUERY_STRING", "")))
    try:
        scale = float(q.get("scale", "2.0"))
        sharpness = float(q.get("sharpness", "0.25"))
    except ValueError:
        return err("scale/sharpness must be numbers")
    scale = min(max(scale, 1.25), 2.0)
    sharpness = min(max(sharpness, 0.0), 2.0)

    src = np.asarray(im, dtype=np.float32) / np.float32(255.0)
    out = upscale(src, scale, sharpness)

    a = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    body = buf.getvalue()

    start_response(
        "200 OK",
        [
            ("Content-Type", "image/png"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]
