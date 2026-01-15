# 服务端API说明文档

### ✅ 核心选型：Electron 以【文件上传 (FormData)】的方式把图片传给本地 Web 服务器处理，是唯一最优解，绝对不要让 Electron 直接读取 / 处理图片文件；
### ✅ 存储结论：处理后的图片，让 Web 服务器处理完直接保存到本地磁盘指定路径，再把「本地文件路径 / 访问 URL」返回给 Electron 即可，Electron 端不需要再做二次保存，完全没必要


### 执行流程
#### Electron 选择本地图片文件 → 通过 FormData 封装二进制文件 → 调用本地 Web 服务器的图片处理 API → Web 服务器接收图片并完成裁剪 / 压缩 / 水印等处理 → Web 服务器把处理后的图片直接保存到本地磁盘 → 返回「处理后的图片本地绝对路径」给 Electron → Electron 拿到路径直接使用 / 展示

#### 优点（全是优势，完美适配你的场景）

##### 1.极致性能：你的 Web 服务器是本地部署，不是公网服务器！上传请求走的是本地localhost/127.0.0.1，没有网络带宽损耗，100M 的图片上传耗时都是毫秒级，比公网快 100 倍，完全不用担心之前说的「上传慢」问题；

##### 2.职责彻底分离（核心！）：Electron 只做「客户端 UI、文件选择、调用 API」，Web 服务器只做「图片处理、文件读写存储」，这是标准的前后端分离架构，代码解耦，后续维护、升级、修改图片处理逻辑（比如加压缩规则、加水印），只改 Web 服务器代码即可，不用重新打包 Electron 客户端；

##### 3.无内存风险：10M + 的图片，Electron 只是做「二进制转发」，不会把图片文件读入 Node 层 / 渲染层内存，不会出现 Electron 客户端内存溢出、卡顿、崩溃的问题；

##### 4.开发成本低：Electron 的文件上传代码和网页端几乎一模一样，复用性极高，Web 服务器的图片处理逻辑（Java/Node/PHP/Python）都是成熟生态，有大量现成工具库；

##### 5.无二次冗余：Web 服务器处理完图片直接存本地，返回路径即可，Electron 拿到就能用，没有任何多余的保存步骤。

#### 缺点

##### 几乎无缺点，硬说的话就是要写几行 FormData 上传代码（代码量极少，后面给你现成示例）。


### 示例
```ts
import axios from 'axios';
export const uploadFile = (file, bizType) => {
    const formData = new FormData();
    formData.append('uploadFile', file);
    formData.append('bizType', bizType);
    return axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    });
};
```

## **语义分类API:** 

### 接口地址
### POST /api/clip
### 请求方式
### POST
### 数据格式
### Content-Type: multipart/form-data

### 接口说明
### 支持任意大小图片 / 文件上传，推荐上传 1M 以上文件使用该接口，禁止 Base64 方式上传。

|参数名	                 |类型	        |是否必传	         |说明                                                     |
|-----------------------|--------------|-------------------|-------------------------------------------------------- |
|uploadFile	            |file	       |是	               |待上传的文件二进制流（图片）                                |
|bizType	            |string	       |是	               |业务类型标识，如：image - 图片、video - 视频、attach - 附件 |
|userId                 |string        |否                 | 操作人 ID，业务关联字段                                   |
|categories             |string[]      |是                 | 分类数组，如主题，副图，详细图等                           |
|model                  |string        |是                 | 模型名称                                                 |
|modelParams            |string        |是                 | 模型参数 (json 字符串)                                    |

### 返回参数

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| code | integer | 状态码，200表示成功 |
| msg | string | 提示信息 |
| data | object | 返回数据对象 |
| data.fileUrl | string | 图片访问URL (Web服务地址) |
| data.localUrl | string | 图片本地存储绝对路径 |
| data.fileName | string | 文件名称 |
| data.fileSize | integer | 文件大小(字节) |
| data.category | string | 识别出的分类结果 |

HTTP POST 200 返回
```json 
{
  "code": 200,
  "msg": "上传成功",
  "data": {
    "fileUrl": "https://127.0.0.1:8080/2026/01/15/abc.jpg",
    "localUrl": "c:\\upload\\2026\\01\\15\\abc.jpg",
    "fileName": "产品主图.jpg",
    "fileSize": 10485760,
    "category": "主图"
  }
}
```




HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件大小超过限制"
}
```

HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件格式有误"
}
```


## **智能修补API:** 

### 接口地址
### POST /api/magic
### 请求方式
### POST
### 数据格式
### Content-Type: multipart/form-data

### 接口说明
### 支持任意大小图片 / 文件上传，推荐上传 1M 以上文件使用该接口，禁止 Base64 方式上传。

|参数名	                 |类型	        |是否必传	         |说明                                                     |
|-----------------------|--------------|-------------------|-------------------------------------------------------- |
|uploadFile	            |file	       |是	               |待上传的文件二进制流（图片）                                |
|bizType	            |string	       |是	               |业务类型标识，如：image - 图片、video - 视频、attach - 附件 |
|userId                 |string        |否                 | 操作人 ID，业务关联字段                                   |
|model                  |string        |是                 | 模型名称                                                 |
|modelParams            |string        |是                 | 模型参数 (json 字符串)                                    |

### 返回参数

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| code | integer | 状态码，200表示成功 |
| msg | string | 提示信息 |
| data | object | 返回数据对象 |
| data.fileUrl | string | 图片访问URL (Web服务地址) |
| data.localUrl | string | 图片本地存储绝对路径 |
| data.fileName | string | 文件名称 |
| data.fileSize | integer | 文件大小(字节) |

HTTP POST 200 返回
```json 
{
  "code": 200,
  "msg": "上传成功",
  "data": {
    "fileUrl": "https://127.0.0.1:8080/2026/01/15/abcd.jpg",
    "localUrl": "c:\\upload\\2026\\01\\15\\abc.jpg",
    "fileName": "修补后图片.jpg",
    "fileSize": 20485760,
  }
}
```

HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件大小超过限制"
}
```

HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件格式有误"
}
```



## **高精抠图API:** 

### 接口地址
### POST /api/removebg
### 请求方式
### POST
### 数据格式
### Content-Type: multipart/form-data

### 接口说明
### 支持任意大小图片 / 文件上传，推荐上传 1M 以上文件使用该接口，禁止 Base64 方式上传。

|参数名	                 |类型	        |是否必传	         |说明                                                     |
|-----------------------|--------------|-------------------|-------------------------------------------------------- |
|uploadFile	            |file	       |是	               |待上传的文件二进制流（图片）                                |
|bizType	            |string	       |是	               |业务类型标识，如：image - 图片、video - 视频、attach - 附件 |
|userId                 |string        |否                 | 操作人 ID，业务关联字段                                   |
|model                  |string        |是                 | 模型名称                                                 |
|modelParams            |string        |是                 | 模型参数 (json 字符串)                                    |

### 返回参数

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| code | integer | 状态码，200表示成功 |
| msg | string | 提示信息 |
| data | object | 返回数据对象 |
| data.fileUrl | string | 图片访问URL (Web服务地址) |
| data.localUrl | string | 图片本地存储绝对路径 |
| data.fileName | string | 文件名称 |
| data.fileSize | integer | 文件大小(字节) |

HTTP POST 200 返回
```json 
{
  "code": 200,
  "msg": "上传成功",
  "data": {
    "fileUrl": "https://127.0.0.1:8080/2026/01/15/abcd.jpg",
    "localUrl": "c:\\upload\\2026\\01\\15\\abc.jpg",
    "fileName": "修补后图片.jpg",
    "fileSize": 20485760,
  }
}
```

HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件大小超过限制"
}
```

HTTP POST 500 返回
```json 
{
  "code": 500,
  "msg": "文件上传失败，文件格式有误"
}
```