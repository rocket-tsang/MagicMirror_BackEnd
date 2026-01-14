from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import cn_clip.clip as clip
from typing import List
import torch

# ================= 配置与加载模型 =================

# 加载ONNX模型路径
TEXT_MODEL_PATH = "./output/deploy/vit-b-16.txt.fp32.onnx"
VISION_MODEL_PATH = "./output/deploy/vit-b-16.img.fp32.onnx"

print("正在加载 ONNX 模型...")
try:
    # 优先使用 CUDA，如果没有则使用 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    text_session = ort.InferenceSession(TEXT_MODEL_PATH, providers=providers)
    vision_session = ort.InferenceSession(VISION_MODEL_PATH, providers=providers)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise e

app = FastAPI(title="Chinese CLIP General Object Detection API", version="2.0.0")

# ================= 预置数据 =================

# 场景3：通用物品识别的预置类别列表 (可以根据实际场景扩充)
COMMON_CATEGORIES = [
    "人", "男人", "女人", "孩子",
    "鞋子", "运动鞋", "高跟鞋", "靴子", "拖鞋", "凉鞋",
    "上衣", "裤子", "裙子", "外套", "帽子", "眼镜", "手表", "包", "领带", "围巾", "手套", "袜子", "内衣", "睡衣", "泳衣", "运动服", "制服", "婚纱", "礼服",
    "猫", "狗", "鸟", "鱼", "马", "牛", "羊", "猪", "兔子", "老鼠", "蛇", "乌龟", "蜥蜴",
    "汽车", "自行车", "摩托车", "公交车", "火车", "飞机", "船", "卡车", "救护车", "消防车", "出租车",
    "桌子", "椅子", "沙发", "床", "柜子", "灯", "电视", "电脑", "手机", "平板电脑", "耳机", "音响", "相机", "键盘", "鼠标", "充电器", "路由器", "显示器", "U盘", "游戏机",
    "食物", "水果", "蔬菜", "饮料", "杯子", "碗", "瓶子", "面包", "牛奶", "鸡蛋", "苹果", "香蕉", "橙子", "西瓜", "土豆", "西红柿", "黄瓜", "胡萝卜", "洋葱", "大米", "面条", "饼干", "巧克力", "蛋糕", "咖啡", "茶", "啤酒", "红酒", "果汁", "矿泉水",
    "树", "花", "草", "天空", "大海", "山", "建筑物", "道路", "桥梁", "公园", "森林", "沙漠", "湖泊", "河流",
    "枕头", "被子", "地毯", "窗帘", "镜子", "梳子", "牙刷", "毛巾", "肥皂", "洗发水", "沐浴露", "洗衣液", "纸巾", "垃圾桶", "拖把", "扫帚", "衣架", "餐具", "筷子", "叉子", "勺子", "砧板", "锅", "平底锅", "电饭煲", "微波炉", "冰箱", "洗衣机", "空调", "风扇",
    "足球", "篮球", "排球", "乒乓球", "网球", "羽毛球", "跳绳", "哑铃", "瑜伽垫", "跑步机", "自行车头盔", "滑板", "冲浪板", "滑雪板", "登山包", "帐篷", "睡袋",
    "口罩", "体温计", "药瓶", "创可贴", "血压计", "听诊器", "注射器", "轮椅", "拐杖",
    "锤子", "螺丝刀", "扳手", "锯子", "尺子", "笔", "铅笔", "橡皮", "胶水", "剪刀", "订书机", "文件夹", "笔记本", "书", "报纸", "杂志", "钥匙", "锁", "钉子", "螺丝",
    "玩具", "积木", "洋娃娃", "摇篮", "奶瓶", "尿布", "婴儿车", "儿童座椅", "学步车",
    "钢琴", "吉他", "小提琴", "鼓", "萨克斯", "笛子", "扬琴", "古筝", "二胡"
]


# ================= 核心工具函数 =================

def preprocess_image(image_bytes, resolution=224):
    """预处理图像：Resize -> Normalize"""
    import torchvision.transforms as transforms

    preprocess = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        return image_tensor.numpy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")


def get_text_features(text_list: List[str]):
    """
    获取文本特征向量
    注意：这里会自动添加 '一张...的照片' 这种 prompt 可能会更好，
    但为了灵活性，我们在业务逻辑层处理 Prompt。
    """
    features = []
    # 逐个处理或小批次处理以防止内存溢出或维度错误
    for text in text_list:
        # Tokenize (注意：context_length 默认 52，根据模型调整)
        token = clip.tokenize([text], context_length=52).numpy().astype(np.int64)

        # Inference
        feature = text_session.run(None, {'text': token})[0]

        # L2 归一化 (非常重要，否则点积不是余弦相似度)
        feature = feature / np.linalg.norm(feature)
        features.append(feature)

    return np.vstack(features)  # Shape: (N, Feature_Dim)


def get_image_features(image_bytes):
    """获取图像特征向量"""
    image_data = preprocess_image(image_bytes)

    # Inference
    image_features = vision_session.run(None, {'image': image_data.astype(np.float32)})[0]

    # L2 归一化
    image_features = image_features / np.linalg.norm(image_features)
    return image_features


def softmax(x):
    """计算 softmax 概率"""
    e_x = np.exp(x - np.max(x))  # 减去max防止溢出
    return e_x / e_x.sum(axis=0)


# ================= API 接口定义 =================

@app.get("/")
def read_root():
    return {"message": "CN-CLIP Object Detection API Ready"}


# --- 场景 1: 判断是不是这个物品 ---
@app.post("/predict/is-object")
async def is_specific_object(
        target_name: str = Form(..., description="你想判断的物品名称，例如：'鞋子'"),
        image: UploadFile = File(...),
        threshold: float = Form(0.6, description="判定为'是'的概率阈值，建议 0.5-0.7")
):
    """
    功能：判断图片是不是指定的物品。
    原理：将图片与 ["一张{target}的照片", "一张其他物品的照片"] 进行对比。
    """
    image_bytes = await image.read()

    # 1. 构造对比文本 (Prompt Engineering)
    # 使用 "其他物品" 作为负样本，构建二分类
    positive_text = f"一张{target_name}的照片"
    negative_text = "一张其他物品的照片"
    texts = [positive_text, negative_text]

    # 2. 提取特征
    img_feat = get_image_features(image_bytes)  # Shape: (1, 512)
    txt_feats = get_text_features(texts)  # Shape: (2, 512)

    # 3. 计算相似度 (点积)
    # img_feat (1, D) dot txt_feats.T (D, 2) -> (1, 2)
    logits_per_image = np.dot(img_feat, txt_feats.T)[0]

    # 4. 计算概率 (Softmax)
    probs = softmax(logits_per_image * 100)  # *100 是 CLIP 的 logit scale 参数，通常需要放大差异

    pos_score = float(probs[0])
    is_match = pos_score > threshold

    return {
        "target": target_name,
        "is_match": is_match,
        "confidence": pos_score,  # 是这个物品的概率
        "raw_logits": logits_per_image.tolist(),
        "comparison_text": texts
    }


# --- 场景 2: 从列表中选出最相似的一个 ---
@app.post("/predict/select-from-list")
async def select_best_match(
        candidate_list: str = Form(..., description="物品名称列表，用逗号分隔，例如：'鞋子,香蕉,汽车'"),
        image: UploadFile = File(...)
):
    """
    功能：给定一张图和一组候选词，判断图片最像哪个。
    """
    image_bytes = await image.read()

    # 1. 解析列表
    candidates = [x.strip() for x in candidate_list.split(",") if x.strip()]
    if not candidates:
        raise HTTPException(status_code=400, detail="Candidate list cannot be empty")

    # 2. 构造 Prompt (加上前缀有助于提升准确率)
    prompts = [f"一张{c}的照片" for c in candidates]

    # 3. 提取特征
    img_feat = get_image_features(image_bytes)
    txt_feats = get_text_features(prompts)

    # 4. 计算相似度
    logits = np.dot(img_feat, txt_feats.T)[0]
    probs = softmax(logits * 100)

    # 5. 找出最大值
    best_idx = np.argmax(probs)
    sorted_indices = np.argsort(probs)[::-1]  # 降序排列

    # 返回前 3 个结果
    top_results = []
    for idx in sorted_indices[:3]:
        if idx < len(candidates):
            top_results.append({
                "label": candidates[idx],
                "score": float(probs[idx])
            })

    return {
        "best_match": candidates[best_idx],
        "best_score": float(probs[best_idx]),
        "top_3_results": top_results
    }


# --- 场景 3: 自动判断是什么物品 (基于通用库) ---
@app.post("/predict/general")
async def recognize_general_object(
        image: UploadFile = File(...)
):
    """
    功能：从预置的常见物品库中，猜测图片是什么。
    注意：CLIP 不是生成式模型，它只能从已知的列表中选。
    """
    image_bytes = await image.read()

    # 1. 使用全局定义的 COMMON_CATEGORIES
    candidates = COMMON_CATEGORIES
    prompts = [f"一张{c}的照片" for c in candidates]

    # 2. 提取特征
    img_feat = get_image_features(image_bytes)
    txt_feats = get_text_features(prompts)

    # 3. 计算相似度
    logits = np.dot(img_feat, txt_feats.T)[0]
    probs = softmax(logits * 100)

    # 4. 找出最佳匹配
    best_idx = np.argmax(probs)
    best_score = float(probs[best_idx])

    # 简单的置信度过滤
    result_label = candidates[best_idx]
    if best_score < 0.1:  # 如果在这么多类别里，最高的概率都很低
        result_label = "未知/不确定"

    return {
        "predicted_label": result_label,
        "confidence": best_score,
        "top_5_guesses": [
            {"label": candidates[i], "score": float(probs[i])}
            for i in np.argsort(probs)[::-1][:5]
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)