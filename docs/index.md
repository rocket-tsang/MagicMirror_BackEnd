# 服务端项目简介

**魔镜** 是一款专为电商场景打造的本地化智能图像处理工具。它利用先进的 AI 模型，帮助电商团队或个人用户快速处理拍摄素材或网络图片，实现自动化的分类、修图和抠图，显著提升工作效率。

## ✨ 服务端核心功能

1.  **智能分类 (CLIP)[https://github.com/OFA-Sys/Chinese-CLIP]:** 基于语义理解自动对图片进行分类。只需提供图片和标签（如：主图、细节图、模特图），系统即可识别图片内容并将其归档至对应的文件夹中。
2.  **智能修补 (LaMa)[https://github.com/advimman/lama]:** 强大的文字擦除与图像修复功能。可自动识别并擦除图片中的水印、中文文案或杂物，并智能补全背景内容。支持模型切换与参数配置。
3.  **高精度抠图 (RMBG-1.4)[https://huggingface.co/briaai/RMBG-1.4]:** 发丝级精度的背景移除功能。能够将商品主体完美从背景中分离，支持生成透明背景或纯白背景图片。
4.  **打包成安装包 (PyInstaller),直接点击进行安装。

## 业务功能

| 功能模块                      | 功能介绍                                                                                                   |
|------------------------------|------------------------------------------------------- -------------------------------------------------- |
| **智能分类 (CLIP)** | 客户端通过HttP POST调用API,传递图片和分类参数，具体参数可以访问(@API.md)[API.md]：文档语义分类API                          |
| **智能修补 (LaMa)** | 客户端通过HttP POST调用API,传递图片和修补参数，完成图片智能修补，具体参数可以访问(@API.md)[API.md]文档                      |                   
| **高精抠图 (RMBG-1.4)** | 客户端通过HttP POST调用API,传递图片和抠图参数，完成图片抠图操作，具体参数可以访问(@API.md)[API.md]文档                |


安装依赖
```bash
pip install onnxruntime-cpu numpy pillow cn-clip uvicorn pyinstaller
```

### **智能分类 (CLIP)**实例代码
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import cn_clip.clip as clip


def preprocess_image(image_path: str, resolution: int = 224) -> np.ndarray:
    """图片预处理：读取、Resize、归一化"""
    try:
        # 读取图片并转为RGB
        image = Image.open(image_path).convert('RGB')
        # 预处理流程（对齐CLIP标准）
        transform = clip._transform(resolution, False)
        image_tensor = transform(image).unsqueeze(0)  # 增加batch维度
        return image_tensor.numpy()
    except Exception as e:
        raise ValueError(f"图片预处理失败：{e}")


def softmax(x: np.ndarray) -> np.ndarray:
    """计算softmax概率，防止数值溢出"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def classify_image(
    image_path: str,
    categories: list[str],
    onnx_model_path: str = "./output/deploy/vit-b-16.img.fp32.onnx",
    text_onnx_path: str = "./output/deploy/vit-b-16.txt.fp32.onnx"
) -> dict:
    """
    核心分类函数
    :param image_path: 图片路径（绝对/相对）
    :param categories: 分类数组（如["猫", "狗", "汽车"]）
    :param onnx_model_path: 视觉ONNX模型路径
    :param text_onnx_path: 文本ONNX模型路径
    :return: 包含图片路径、大小、分类结果的字典
    """
    # 1. 基础校验
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在：{image_path}")
    if not categories:
        raise ValueError("分类数组不能为空")

    # 2. 加载ONNX模型（仅CPU）
    try:
        providers = ['CPUExecutionProvider']
        vision_session = ort.InferenceSession(onnx_model_path, providers=providers)
        text_session = ort.InferenceSession(text_onnx_path, providers=providers)
    except Exception as e:
        raise RuntimeError(f"ONNX模型加载失败：{e}")

    # 3. 处理图片：提取视觉特征
    image_data = preprocess_image(image_path)
    image_feat = vision_session.run(None, {'image': image_data.astype(np.float32)})[0]
    image_feat = image_feat / np.linalg.norm(image_feat)  # L2归一化

    # 4. 处理分类文本：提取文本特征
    text_features = []
    for cate in categories:
        # 构造CLIP风格prompt（提升分类准确率）
        prompt = f"一张{cate}的照片"
        # 文本token化（对齐cn-clip）
        token = clip.tokenize([prompt], context_length=52).numpy().astype(np.int64)
        # 文本特征推理
        txt_feat = text_session.run(None, {'text': token})[0]
        txt_feat = txt_feat / np.linalg.norm(txt_feat)  # L2归一化
        text_features.append(txt_feat)
    text_feats = np.vstack(text_features)

    # 5. 计算相似度 + 概率
    logits = np.dot(image_feat, text_feats.T)[0]  # 点积计算相似度
    probs = softmax(logits * 100)  # 放大差异后计算概率

    # 6. 确定最佳分类
    best_idx = np.argmax(probs)
    best_cate = categories[best_idx]
    best_score = float(probs[best_idx])

    # 7. 获取图片大小（宽x高）
    with Image.open(image_path) as img:
        img_size = f"{img.width}x{img.height}"

    # 8. 构造返回结果
    return {
        "image_path": os.path.abspath(image_path),  # 绝对路径
        "image_size": img_size,
        "classification": {
            "best_category": best_cate,
            "confidence": best_score,
            "all_categories": [
                {"category": cate, "confidence": float(probs[i])}
                for i, cate in enumerate(categories)
            ]
        }
    }


# 测试示例（可直接运行）
if __name__ == "__main__":
    # 测试参数
    test_image_path = "./test.jpg"  # 替换为你的图片路径
    test_categories = ["猫", "狗", "汽车", "桌子", "花"]  # 自定义分类数组

    # 执行分类
    try:
        result = classify_image(test_image_path, test_categories)
        print("===== 图片分类结果 =====")
        print(f"图片路径：{result['image_path']}")
        print(f"图片大小：{result['image_size']}")
        print(f"最佳分类：{result['classification']['best_category']}（置信度：{result['classification']['confidence']:.4f}）")
        print("所有分类结果：")
        for item in result['classification']['all_categories']:
            print(f"- {item['category']}: {item['confidence']:.4f}")
    except Exception as e:
        print(f"分类失败：{e}")
```


### 使用方式
```python
# 调用示例
res = classify_image(
    image_path="./your_image.png",
    categories=["苹果", "香蕉", "橙子", "西瓜"]
)
print(res)
```

### 返回结果实例

```python
{
    "image_path": "/home/user/your_image.png",  # 绝对路径
    "image_size": "500x300",  # 宽x高
    "classification": {
        "best_category": "香蕉",  # 最佳匹配分类
        "confidence": 0.9876,    # 最佳分类置信度
        "all_categories": [      # 所有分类的置信度
            {"category": "香蕉", "confidence": 0.9876},
            {"category": "橙子", "confidence": 0.0102},
            {"category": "苹果", "confidence": 0.0018},
            {"category": "西瓜", "confidence": 0.0004}
        ]
    }
}
```

### 注意事项：

##### 需确保 onnx_model_path 和 text_onnx_path 指向你的 CN-CLIP 导出的 ONNX 模型文件；
##### 分类数组建议为中文（匹配 cn-clip 的训练语料），且避免空值 / 空格；
##### 图片路径支持绝对路径 / 相对路径，支持 JPG/PNG 等常见格式；
##### 若仅需最简版本（无需 uvicorn），可删除 uvicorn 依赖，直接调用classify_image函数即可。
##### 如果需要基于 FastAPI+uvicorn 提供接口形式的调用，可补充以下精简接口代码（可选）：
 
 ```python
 from fastapi import FastAPI
import uvicorn

app = FastAPI(title="图片分类API")

@app.post("/api/clip")
async def api_clip(image_path: str, categories: list[str]):
    try:
        result = classify_image(image_path, categories)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

#### 接口调用示例（POST 请求）：
```text
http://127.0.0.1:8000/api/clip?image_path=./test.jpg&categories=猫&categories=狗&categories=汽车
```

