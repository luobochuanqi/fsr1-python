"""本地预览服务器：直接跑部署用的同一份 WSGI app，所见即所得。

用法：uv run python serve.py [port]   （默认 8000，仅本机回环）
"""

import sys

from api.upscale import app

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    from wsgiref.simple_server import make_server

    with make_server("127.0.0.1", port, app) as httpd:
        print(f"demo: http://127.0.0.1:{port}")
        httpd.serve_forever()
