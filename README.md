# Vision Agent 

有些纯文本大语言模型（如 DeepSeek）原生不具备多模态视觉能力。本项目通过构建智能体（Agent）工作流，让大模型能够自主调度本地视觉模型（OCR、目标检测、图像描述）以及**外部生态工具（全网搜索、代码执行）**，从而赋予纯文本大模型强大的图片理解、现实数据获取与复杂逻辑计算能力，实现对极其复杂的真实世界问题的自然语言问答。

## 核心功能

* **智能任务编排**：基于 DeepSeek 的推理链（CoT），自主决定工具调用顺序。
* **多模态工具箱**：
  * **文字识别**：集成 PaddleOCR。
  * **目标检测**：集成 YOLO 进行物体定位与计数。
  * **场景理解**：集成 BLIP 生成图像描述。(如果使用具备多模态能力的大模型，那这三个功能是没有必要的)
  * **联网搜索**：集成 DuckDuckGo，赋予大模型突破知识库限制、实时获取外部世界数据（如最新价格、新闻）的能力。
  * **安全代码沙盒**：基于 Docker 容器技术，允许大模型动态编写并隔离执行 Python 代码，完成高难度的数学计算与数据清洗。
* **工业化设计**：支持多会话管理（JSON 持久化）、全链路日志追踪及显存常驻优化。

## 快速开始

### 1. 环境准备
建议使用 Python 3.10+ 环境：

```bash
git clone https://github.com/ZeroBot-ATRI/llm-vision-agent.git
cd vision-agent
conda create -n vision_agent python=3.10 -y
conda activate vision_agent
```

### 2. 安装依赖
请根据硬件环境优先单独安装 **PyTorch**，随后安装其他核心依赖：

```bash
# 以 NVIDIA 显卡 (CUDA 12.8) 为例
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装其他依赖
pip install -r requirements.txt
```

安装并启动 **Docker**

### 3. 配置与运行
1. 把 `config.backup.json` 文件复制一份，重命名为`config.json`并配置 API 密钥：

```json
{
  "deepseek_api_key": "your-api-key-here"
}
```

2. 启动 Web 服务：

```bash
streamlit run app.py
```

## 目录结构

* `app.py`: Streamlit 前端页面与 Agent 核心逻辑流转。
* `requirements.txt`: 核心依赖清单。
* `sessions/`: 自动生成的对话历史归档。
* `logs/`: 自动生成的底层运行完整日志。
* `workspace/`: 用户前端上传的图片缓存。