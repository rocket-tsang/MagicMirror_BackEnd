# 🚀 魔镜 (MagicMirror)

**魔镜** 是一款专为电商场景打造的本地化智能图像处理工具。它利用先进的 AI 模型，帮助电商团队或个人用户快速处理拍摄素材或网络图片，实现自动化的分类、修图和抠图，显著提升工作效率。

## ✨ 服务端核心功能

1.  **智能分类 (CLIP)[https://github.com/OFA-Sys/Chinese-CLIP]:** 基于语义理解自动对图片进行分类。只需提供图片和标签（如：主图、细节图、模特图），系统即可识别图片内容并将其归档至对应的文件夹中。
2.  **智能修补 (LaMa)[https://github.com/advimman/lama]:** 强大的文字擦除与图像修复功能。可自动识别并擦除图片中的水印、中文文案或杂物，并智能补全背景内容。支持模型切换与参数配置。
3.  **高精度抠图 (RMBG-1.4)[https://huggingface.co/briaai/RMBG-1.4]:** 发丝级精度的背景移除功能。能够将商品主体完美从背景中分离，支持生成透明背景或纯白背景图片。

## 📖 项目简介

本项目旨在为电商公司提供高效的批量图片处理解决方案。系统采用 **C/S 架构**：

*   **服务端 (Backend):** 基于 Python 编写，使用 FastAPI 提供 RESTful 接口。集成了多种离线 AI 大模型，支持封装为 Windows/Mac 系统后台服务运行。
*   **客户端 (Frontend):** 基于 Electron + Vue 3 + Vite 开发，提供友好的图形化操作界面。最终打包为 Windows (.exe) 和 macOS (.dmg/app) 安装包，用户一键安装即可使用。

## 🌟 后端 (Python) 特性

*   **模型支持:**
    *   **语义分类:** 集成 `CLIP_Chinese` 模型，完美支持中文语义标签。
    *   **图像修补:** 集成 `LaMa` 等主流 Inpainting 模型。
    *   **高精抠图:** 集成 `RMBG-1.4` 模型，提供商业级的抠图效果。
*   **离线优先:** 所有 AI 模型均为离线运行。首次调用时自动下载模型权重，后续运行完全无需联网，保障数据安全与处理速度。
*   **标准接口:** 基于 FastAPI 构建本地 HTTP RESTful API，方便 Electron 客户端或其他应用调用。
*   **系统服务化:** 支持打包为 Windows Service 或 macOS LaunchAgent/Daemon，随系统启动，提供稳定的后台支持。
*   **日志与监控:** 完善的日志记录机制，支持输出到本地文件或通过 webhook 发送至远程数据库/监控平台。
*   **高扩展性:** 模块化设计，业务逻辑与模型处理分离，便于后续接入新功能。

## 🔥 前提条件 (开发环境)

在运行或开发本项目服务端之前，请确保环境满足以下要求：

*   **Python:** 3.10 或更高版本
*   **包管理器:** pip

## 🔨 API 接口预览

服务端提供以下核心 API (建议采用 POST 方法):

-   **语义分类:** `/api/clip`
    -   *Input:* 具体参考[语义分类API](docs/API.md)
-   **智能修补:** `/api/magic`
    -   *Input:* 具体参考[智能修补API](docs/API.md)
-   **高精抠图:** `/api/removebg`
    -   *Input:* 具体参考[高精抠图API](docs/API.md)

## 🚀 项目结构建议

```text
imagehelper-backend/
├── main.py               # FastAPI 应用入口
├── api/                  # api目录
│   ├── index.py          # 提供接口，包括（'/api/clip','/api/magic','/api/removebg'）
│   ├── clip_processor.py     # 语义分类 (CLIP)
│   └── removebg_processor.py # 高精抠图 (RMBG)
├── models_manager.py     # AI 模型加载、管理与推理封装
├── processor/            # 业务逻辑处理模块
│   ├── magic_processor.py    # 智能修补 (LaMa)
│   ├── clip_processor.py     # 语义分类 (CLIP)
│   └── removebg_processor.py # 高精抠图 (RMBG)
├── requirements.txt      # Python 依赖清单
├── Dockerfile            # Docker 构建文件
├── storage/              # 临时文件存储 (处理后的图片等)
├── model/                # 大模型模型存放目录
├── test/                 # 测试类
└── db/                   # SQLite 数据库存放目录

```

## 🛠️ 技术栈

*   **编程语言:** Python 3.10.5
*   **Web 框架:** FastAPI
*   **Python核心框架:**onnxruntime-gpu pillow numpy uvicorn cn-clip等
*   **核心模型:**
    *   CLIP_Chinese (语义理解)
    *   LaMa (图像修补)
    *   RMBG-1.4 (背景移除)
*   **数据库:** SQLite
*   **打包工具:** PyInstaller (用于生成可执行文件/服务)
*   **配置框架:** python-dotenv
*   **数据库框架** SQLAlchemy Core