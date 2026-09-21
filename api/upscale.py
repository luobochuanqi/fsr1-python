"""Vercel Function：接收上传图片，跑 FSR1 超分，返回 PNG。

接口：POST /api/upscale?scale=2.0&sharpness=0.25
  请求体 = 原图原始字节（PNG/JPEG，无需 multipart）
成功返回 200 + image/png；失败返回 4xx + text/plain。

部署形态：Vercel 项目根 = 本目录，cwd 即项目根，故可直接 import 同目录的
fsr1.py（依赖见 requirements.txt）。冷启动导入 NumPy 约 +0.5s。输入最大边
512px：NumPy 路径在该尺寸约 1.2s，时限余量充足。
"""

import io

import numpy as np
from PIL import Image

from fsr1 import upscale

MAX_EDGE = 512  # 输入最大边；NumPy 路径在 512 时约 1.2s
MAX_PIXELS = MAX_EDGE * MAX_EDGE * 4  # 4M px，防解压炸弹
MAX_UPLOAD = 8 * 1024 * 1024  # 8 MB


def app(environ, start_response):
    """WSGI 入口：/ 与 /api/upscale 统一入口（tool.vercel.entrypoint 指向此处）。"""
    if environ["PATH_INFO"] == "/api/upscale":
        return _upscale(environ, start_response)
    if environ["REQUEST_METHOD"] == "GET" and environ["PATH_INFO"] in ("/", "/index.html"):
        return _file("index.html", "text/html; charset=utf-8", environ, start_response)
    if environ["REQUEST_METHOD"] == "GET" and environ["PATH_INFO"].startswith("/sample/"):
        return _sample(environ, start_response)
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"not found"]


def _sample(environ, start_response):
    # 内置示例图对（预跑好的 FSR1 结果），页面默认展示，无需上传即可看效果。
    name = environ["PATH_INFO"].removeprefix("/sample/")
    if name not in ("origin.jpg", "fsr1.jpg"):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    return _file(f"sample/{name}", "image/jpeg", environ, start_response)


def _file(path, ctype, environ, start_response):
    # Vercel 的 cwd 是项目根，文件就在根上。
    try:
        with open(path, "rb") as f:
            body = f.read()
    except OSError:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [f"{path} missing".encode()]
    start_response("200 OK", [
        ("Content-Type", ctype),
        ("Content-Length", str(len(body))),
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
