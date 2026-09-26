# super-resolution

超分辨率研究仓库。用 pnpm workspace 管理，每个单元仓库含算法、演示和资源。

| 包             | 内容                                                       | 技术               |
| -------------- | ---------------------------------------------------------- | ------------------ |
| `fsr1/`        | FSR1（EASU + RCAS）NumPy 逐位复刻，Web 对比演示 + 超分 API | Python (uv) + Vite |
| `fsr2/`        | FSR2 时域超分复现（Godot 4.4 + Bistro 场景骨架）           | Godot              |
| `packages/ui/` | 共享前后对比查看器（缩放/平移/分割线）                     | ESM                |

## 命令

```bash
pnpm install   # 装全部 workspace
pnpm build     # 构建所有 web 包
pnpm dev       # 起 fsr1 web 演示（/api 另开：cd fsr1 && uv run python serve.py）
```

## 部署（Vercel）

`fsr1/` 是自包含部署单元（pyproject、api/、web/、vercel.json）。
