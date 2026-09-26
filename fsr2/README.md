# fsr2

FSR2（时域超分）复现研究单元：Godot 4.4 工程，用 Godot 打开本目录即可运行。
场景沿用 Bistro-Demo-Tweaked 的完整演示系统：设置 UI（时间、分辨率缩放、画质）、
人物控制器（WASD 行走观察）、音效与昼夜光照均可直接使用。其中分辨率缩放控件
正适合作为后续 FSR2 复现的 A/B 对照开关。

相对上游仅剔除三处无用文件：`Build/`（空目录）、`Resources/MainScene.lmbake`
与 `MainScene.exr.import`（无任何 LightmapGI 引用的死烘焙残留，且其 `.exr`
上游即缺失）。

## 场景资产获取

`Meshes/`（约 233MB）与 `Textures/`（约 642MB）为 Bistro 资产，不入 git
（见仓库根 .gitignore）。复制方式：

```bash
git clone --depth 1 https://github.com/Jamsers/Bistro-Demo-Tweaked
cp -r Bistro-Demo-Tweaked/Meshes Bistro-Demo-Tweaked/Textures fsr2/
```

## 来源与许可

抽离自 [Jamsers/Bistro-Demo-Tweaked](https://github.com/Jamsers/Bistro-Demo-Tweaked)
（代码 MIT / 资产 CC-BY-4.0），资产源自 NVIDIA ORCA 的 Amazon Lumberyard
Bistro。署名义务见 `ATTRIBUTION`、`LICENSE-ASSETS`（CC-BY-4.0）、
`LICENSE-CODE`（MIT）。
