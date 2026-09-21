"""本地预览服务器：静态页 + 同一份 WSGI app，模拟 Vercel 部署形态。

用法：uv run python serve.py [port]   （默认 8000，仅本机回环）
"""

import sys
from wsgiref.simple_server import make_server

sys.path.insert(0, ".")

from api.upscale import app


def static(environ, start_response):
    if environ["PATH_INFO"] not in ("/", "/index.html"):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"not found"]
    with open("index.html", "rb") as f:
        body = f.read()
    start_response(
        "200 OK",
        [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def router(environ, start_response):
    if environ["PATH_INFO"] == "/api/upscale":
        return app(environ, start_response)
    return static(environ, start_response)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    with make_server("127.0.0.1", port, router) as httpd:
        print(f"demo: http://127.0.0.1:{port}")
        httpd.serve_forever()
