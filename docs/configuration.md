# 服务端配置文件说明文档

## 概述

本项目采用 python3.10.5+框架，使用 python-dotenv框架的配置文件管理来管理应用的各项配置。配置文件分为三个层级：

- `.env`          - 主配置文件，包含所有通用配置
- `.env.development` - 开发环境专用配置
- `.env.production.yml` - 生产环境专用配置


安装依赖
```bash
pip install python-dotenv sqlalchemy
```

### 实例代码

```python
# 导入核心依赖
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.orm import sessionmaker

# ===================== 1. 读取 .env 配置文件 =====================
# 加载项目根目录的.env文件所有配置项
load_dotenv()

# 从环境变量中读取配置（严格对应你的.env文件key）
ENV = os.getenv("ENV")
DEBUG = os.getenv("DEBUG").lower() == "true"  # 转为布尔值
DB_TYPE = os.getenv("DataSource.Type")
DB_URL = os.getenv("DataSource.URL")
DB_USER = os.getenv("DataSource.UserName")
DB_PWD = os.getenv("DataSource.Password")

# ===================== 2. SQLAlchemy Core 数据库连接配置 =====================
# ✅ 适配你的sqlite配置：sqlite无需账号密码，拼接标准连接字符串
# 注：此处使用 相对路径的sqlite数据库文件 data.db，可自行修改为绝对路径如 sqlite:///D:/project/data.db
if DB_TYPE == "sqlite":
    DB_CONNECT_STR = f"{DB_TYPE}:///{DB_URL if DB_URL else 'data.db'}"
else:
    # 兼容其他数据库（mysql/postgresql），拓展性强，可不改
    DB_CONNECT_STR = f"{DB_TYPE}://{DB_USER}:{DB_PWD}@{DB_URL}"

# 创建数据库引擎（sqlite必须加 check_same_thread=False，否则多线程报错）
engine = create_engine(
    DB_CONNECT_STR,
    echo=DEBUG,  # DEBUG=True时，控制台打印SQL执行日志，方便调试
    connect_args={"check_same_thread": False} if DB_TYPE == "sqlite" else {}
)

# 创建元数据对象（核心！SQLAlchemy Core 用来映射数据库表结构）
metadata = MetaData()

# 绑定元数据到数据库引擎，自动反射加载数据库中的所有表
metadata.reflect(bind=engine)

# ===================== 3. 映射 image_clip 数据表 =====================
# ✅ 核心操作：反射加载 已存在的 image_clip 表
# 无需手动定义表字段，自动读取数据库中image_clip表的所有字段结构
image_clip_table = Table("image_clip", metadata, autoload_with=engine)

# ===================== 4. 创建数据库会话 =====================
# 创建会话工厂，用于执行SQL语句
Session = sessionmaker(bind=engine)
session = Session()

# ===================== 5. 核心业务：读取 image_clip 表数据（多场景示例） =====================
def read_image_clip_table():
    """读取image_clip表数据的核心方法"""
    try:
        # ✅ 场景1：查询 image_clip 表 所有数据（最常用）
        stmt_all = select(image_clip_table)
        result_all = session.execute(stmt_all).fetchall()
        print("===== image_clip表 所有数据 =====")
        for row in result_all:
            print(row)  # row是元组，可通过 row.字段名 取值，如 row.id, row.clip_content

        # ✅ 场景2：查询 单条数据（按条件，示例：id=1）
        stmt_single = select(image_clip_table).where(image_clip_table.c.id == 1)
        result_single = session.execute(stmt_single).first()
        print("\n===== image_clip表 id=1 的单条数据 =====")
        if result_single:
            print(f"id: {result_single.id}, 内容: {result_single.clip_content}")

        # ✅ 场景3：查询 部分字段（示例：只查 id 和 image_url 字段）
        stmt_part = select(image_clip_table.c.id, image_clip_table.c.image_url)
        result_part = session.execute(stmt_part).fetchall()
        print("\n===== image_clip表 部分字段数据 =====")
        for row in result_part:
            print(f"id: {row.id}, 图片地址: {row.image_url}")

        # ✅ 场景4：查询 带排序+分页（示例：按id倒序，取前10条）
        stmt_page = select(image_clip_table).order_by(image_clip_table.c.id.desc()).limit(10)
        result_page = session.execute(stmt_page).fetchall()
        print("\n===== image_clip表 分页+排序数据 =====")
        for row in result_page:
            print(row)

        return {
            "code": 200,
            "msg": "查询成功",
            "data": result_all
        }

    except Exception as e:
        print(f"查询失败：{str(e)}")
        session.rollback()  # 异常回滚
        return {
            "code": 500,
            "msg": f"查询失败：{str(e)}",
            "data": None
        }
    finally:
        # 关闭会话，释放连接
        session.close()

# ===================== 执行查询 =====================
if __name__ == "__main__":
    print(f"当前运行环境: {ENV}, Debug模式: {DEBUG}")
    print(f"数据库连接类型: {DB_TYPE}, 连接地址: {DB_CONNECT_STR}")
    read_image_clip_table()
```


## 配置文件结构

### 1. 应用基础配置

```text
ENV=development           #环境名称
DEBUG=True                #是否Debug
DataSource.Type=sqlite    #数据库类型
DataSource.URL=sqlite:///你的路径.db #数据库地址
DataSource.UserName=root  #数据库用户名
DataSource.Password=password #数据库密码

```