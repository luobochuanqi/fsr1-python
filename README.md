# super-resolution

超分辨率算法研究集。pnpm workspace 聚合多个研究单元，每个单元是一个自包含的包：
算法、演示、资源都在自己的目录里。

| 包 | 内容 | 技术 |
| --- | --- | --- |
| `fsr1/` | FSR1（EASU + RCAS）NumPy 逐位复刻 + Web 对比演示 + 超分 API | Python (uv) + Vite |
| `fsr2/` | FSR2 时域超分复现（Godot 4.4 + Bistro 场景骨架） | Godot |
| `packages/ui/` | 共享演示 UI：前后对比查看器（缩放/平移/分割线） | ESM |

## 命令

```bash
pnpm install        # 安装全部 workspace 成员
pnpm build          # 构建所有 web 包
pnpm dev            # fsr1 web 演示开发服务器（/api 需另开：cd fsr1 && uv run python serve.py）
```

## 约定

- 新研究 = 新目录 + `package.json`（并在 `pnpm-workspace.yaml` 登记）。
- Python 研究单元同时是独立 uv 项目（`uv init` / `uv add` / `uv run` 在包内执行）。
- `fsr2/` 的大二进制资产（Meshes/Textures，约 875MB）不入 git，获取方式见
  `fsr2/README.md`。

## 部署（Vercel）

`fsr1/` 是自包含部署单元（pyproject + api/ + web/ + vercel.json 都在包内）。
Vercel 项目设置两项：

1. Root Directory = `fsr1`
2. 勾选 Include files outside of the Root Directory（pnpm lockfile 在仓库根）
