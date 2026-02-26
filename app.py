import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from paddleocr import PaddleOCR
import json
from openai import OpenAI
from PIL import Image
import logging
import streamlit as st
import json
import os
import re
from openai import OpenAI
from datetime import datetime

# ==========================================
# 全局日志系统配置
# ==========================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 每天生成一个独立的日志文件
log_filename = datetime.now().strftime("agent_%Y%m%d.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

# 1. 获取我们专属的独立 logger，不使用全局 basicConfig
logger = logging.getLogger("VisionAgent")
logger.setLevel(logging.INFO)

# 2. 核心：判断是否已经添加过（防止 Streamlit 刷新导致重复打印）
if not logger.handlers:
    # 配置文件写入通道
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    # 配置终端打印通道
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)
    
    # 切断与全局日志的联系，防止被其他库干扰或打印双份
    logger.propagate = False

# 静音 PaddleOCR 的废话
logging.getLogger("ppocr").setLevel(logging.ERROR)
# ==========================================
# 初始化 DeepSeek 客户端
# ==========================================

def load_api_key():
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f).get('deepseek_api_key')
client = OpenAI(api_key=load_api_key(), base_url="https://api.deepseek.com/v1")

# ==========================================
# 将模型永久缓存到显存中
# ==========================================
@st.cache_resource
def load_ai_models():
    """只有在第一次打开网页时才会执行，后续所有操作直接从内存读取"""
    print("[系统日志] 正在启动 AI 引擎并装载模型 (仅执行一次)...")
    
    # 屏蔽 PaddleOCR 烦人的红色调试日志
    logging.getLogger("ppocr").setLevel(logging.ERROR) 
    
    ocr = PaddleOCR(lang="ch")
    yolo = YOLO("yolov8n.pt")
    
    blip_p = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_m = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # 如果有 GPU，顺便把 BLIP 提前移动到 GPU 上，避免每次识别时反复搬运
    if torch.cuda.is_available():
        blip_m.to("cuda")
        
    print("[系统日志] 所有 AI 模型已锁定在内存中！")
    return ocr, yolo, blip_p, blip_m

with st.spinner("正在将视觉模型装载到显存中，请稍候（仅启动时加载一次）..."):
    ocr_engine, yolo_engine, blip_processor, blip_model = load_ai_models()

def real_run_ocr(image_path):
    print(f"\n[系统日志] 正在调用真实的 PaddleOCR 扫描图片: {image_path}...")
    try:
        result = ocr_engine.ocr(image_path)
        if not result or not result[0]:
            return f"未能在图片 '{image_path}' 中识别到任何文字。"

        extracted_texts = [line[1][0] for line in result[0]]
        final_text = "\n".join(extracted_texts)
        print(f"[系统日志] OCR 扫描完成。")
        return final_text
    except Exception as e:
        return f"OCR 代码执行出错，原因：{str(e)}"

def real_run_yolo(image_path):
    print(f"\n[系统日志] 正在调用 YOLO 检测图片: {image_path}...")
    try:
        results = yolo_engine(image_path, verbose=False)
        detected_objects = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = yolo_engine.names[class_id]
                detected_objects.append(class_name)
        
        if not detected_objects:
            return "未在图片中检测到任何常见物体。"
            
        from collections import Counter
        object_counts = Counter(detected_objects)
        result_json = json.dumps(dict(object_counts), ensure_ascii=False)
        print(f"[系统日志] YOLO 检测完成: {result_json}")
        return f"图片中包含以下物体及数量（英文标签）：{result_json}"
    except Exception as e:
        return f"YOLO 检测失败，原因：{str(e)}"
    
def real_run_blip(image_path):
    print(f"\n[系统日志] 正在调用 BLIP 描述图片: {image_path}...")
    try:
        raw_image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(raw_image, return_tensors="pt")
        # 将张量移动到正确的设备（如果用的是 GPU）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        print(f"[系统日志] BLIP 描述完成: {caption}")
        return f"图片的整体场景描述为（英文）：{caption}"
    except Exception as e:
        return f"BLIP 描述失败，原因：{str(e)}"  
    
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_ocr",
            "description": "光学字符识别（OCR）工具。当需要提取图片中的文字时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string", 
                        "description": "图片路径"
                    },
                    "lang": {
                        "type": "string", 
                        "description": "指定要识别的语言，默认为 'ch'（中英文混合），如果全是英文请传入 'en'"
                    }
                },
                "required": ["image_path"] # lang 是可选参数，大模型可以自己决定传不传
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_yolo",
            "description": "目标检测工具。用于统计图片里的物体和数量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图片路径"},
                    "target": {"type": "string", "description": "如果用户只想找特定物体（如'人'），在这里传入英文标签（如'person'）。如果要找所有物体，请忽略此参数。"}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_blip",
            "description": "图像场景描述工具。用于了解图片整体画了什么。",
            "parameters": {
                "type": "object",
                "properties": {"image_path": {"type": "string", "description": "图片路径"}},
                "required": ["image_path"]
            }
        }
    }
]

# ==========================================
# 多会话文件管理系统
# ==========================================
UPLOAD_DIR = "uploads"
SESSION_DIR = "sessions"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SESSION_DIR, exist_ok=True)

def start_new_chat():
    """初始化一个全新的会话"""
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") # 用时间戳做唯一ID
    st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages = []
    st.session_state.current_image_path = None

def save_session():
    """把当前会话连同图片路径，保存为独立的 JSON 文件"""
    if not st.session_state.messages: 
        return # 空对话不保存
    
    session_data = {
        "session_id": st.session_state.session_id,
        "timestamp": st.session_state.timestamp,
        "image_path": st.session_state.current_image_path,
        # 过滤掉底层繁杂的 tool 调用记录，只保存你和大模型的对话
        "messages": [m for m in st.session_state.messages if m.get("role") in ["user", "assistant"] and not m.get("tool_calls")],
        "raw_trace": st.session_state.messages
    }
    filepath = os.path.join(SESSION_DIR, f"{st.session_state.session_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

def get_all_sessions():
    """读取所有历史会话"""
    files = [f for f in os.listdir(SESSION_DIR) if f.endswith('.json')]
    sessions = []
    for file in files:
        with open(os.path.join(SESSION_DIR, file), "r", encoding="utf-8") as f:
            try:
                sessions.append(json.load(f))
            except:
                pass
    # 按时间倒序排列（最新的在最上面）
    sessions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return sessions

def load_chat(session_data):
    """加载选中的历史会话"""
    st.session_state.session_id = session_data["session_id"]
    st.session_state.timestamp = session_data.get("timestamp", "")
    st.session_state.messages = session_data.get("messages", [])
    st.session_state.current_image_path = session_data.get("image_path")

# ==========================================
# 网页前端 UI 与 核心逻辑
# ==========================================

st.set_page_config(page_title="Vision Agent", layout="wide")
st.title("llm-vision-agent")

# 系统初始化：如果是第一次打开，建立新会话
if "session_id" not in st.session_state:
    start_new_chat()

# --- 侧边栏：历史记录与图片工作区 ---
with st.sidebar:
    # 1. 重置/新建对话按钮
    if st.button("+ 新建对话 (重置)", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun() # 强制刷新页面
        
    st.divider()
    
    # 2. 图片拖拽区
    st.header("图片工作区")
    # 巧妙的魔法：给 uploader 加上 session_id 作为 key，这样新建对话时上传框会自动清空！
    uploaded_file = st.file_uploader("把图片拖到这里...", type=["png", "jpg", "jpeg", "webp"], key=f"uploader_{st.session_state.session_id}")
    
    if uploaded_file:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.current_image_path = file_path
        save_session() # 上传图片也触发一次保存
    
    # 展示当前会话绑定的图片
    if st.session_state.current_image_path and os.path.exists(st.session_state.current_image_path):
        st.image(st.session_state.current_image_path, caption="当前关联图片", use_container_width=True)
    
    st.divider()
    
    # 3. 历史会话漫游区
    st.header("历史会话")
    all_sessions = get_all_sessions()
    
    for sess in all_sessions:
        # 提取第一句话作为标题，如果没有则显示“新对话”
        msgs = sess.get("messages", [])
        time_str = sess.get("timestamp", "")[5:16] # 提取 月-日 时:分
        if msgs:
            # 截取用户说的第一句话作为按钮名字
            first_msg = msgs[0]["content"].replace("\n", " ")[:12]
            btn_label = f"{time_str} | {first_msg}..."
        else:
            btn_label = f"✨ {time_str} | 空白对话"
            
        # 渲染历史记录按钮，如果是当前选中的，高亮显示
        is_current = (sess["session_id"] == st.session_state.session_id)
        if st.button(btn_label, key=f"btn_{sess['session_id']}", disabled=is_current, use_container_width=True):
            load_chat(sess)
            st.rerun()

# --- 主聊天区域展示历史记录 ---
for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"]:
        with st.chat_message(msg["role"]):
            # 把系统塞小抄的尾巴去掉再展示，保持界面整洁
            display_text = re.sub(r'\[系统提示：用户当前聚焦的图片路径是.*?\]', '', msg["content"]).strip()
            st.markdown(display_text)

# --- 接收用户输入 ---
user_input = st.chat_input("发一张图片到左侧，或问我任何问题...")

if user_input:
    # 1. 网页上展示用户的输入
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. 后台塞小抄
    if st.session_state.current_image_path:
        prompt_for_llm = f"{user_input}\n\n[系统提示：用户当前聚焦的图片路径是 '{st.session_state.current_image_path}'，请使用此路径调用视觉工具]"
    else:
        prompt_for_llm = user_input
        
    st.session_state.messages.append({"role": "user", "content": prompt_for_llm})
    save_session() # 每说一句话就保存一次
    
    # 3. 召唤大模型
    with st.chat_message("assistant"):
        status_box = st.status("Agent 正在思考并调用视觉工具...", expanded=True)
        
        MAX_ROUNDS = 3
        current_round = 1
        current_messages = st.session_state.messages.copy() 
        
        while current_round <= MAX_ROUNDS:
            status_box.write(f"第 {current_round} 轮思考中...")
            logger.info(f"\n{'='*40}\n[Session: {st.session_state.session_id} | Round: {current_round}]\n🚀 发送给大模型的 Messages:\n{json.dumps(current_messages, ensure_ascii=False, indent=2)}\n{'='*40}")
            if current_round == MAX_ROUNDS:
                current_messages.append({
                    "role": "user",
                    "content": "【系统强制指令】：由于资源限制，无法再调用任何工具。请立刻基于已有的工具检测结果回答，绝对不要输出工具调用代码！"
                })
                response = client.chat.completions.create(model="deepseek-chat", messages=current_messages)
            else:
                response = client.chat.completions.create(model="deepseek-chat", messages=current_messages, tools=tools, tool_choice="auto")
            
            ai_message = response.choices[0].message
            logger.info(f"大模型原始回复:\n{ai_message.model_dump_json(indent=2)}")
            ai_message_dict = {"role": ai_message.role, "content": ai_message.content}
            
            if ai_message.tool_calls:
                ai_message_dict["tool_calls"] = [{"id": t.id, "type": t.type, "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in ai_message.tool_calls]
            
            if current_round < MAX_ROUNDS and ai_message.tool_calls:
                current_messages.append(ai_message_dict)
                for tool_call in ai_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    status_box.write(f"决定调用工具：{func_name}")
                    
                    if func_name == "run_ocr":
                        tool_result = real_run_ocr(func_args.get("image_path")) 
                    elif func_name == "run_yolo":
                        tool_result = real_run_yolo(func_args.get("image_path"))
                    elif func_name == "run_blip":
                        tool_result = real_run_blip(func_args.get("image_path"))
                    else:
                        tool_result = "未知的工具名称"
                    logger.info(f"工具 [{func_name}] 执行结果:\n{tool_result}")
                    status_box.write(f"{func_name} 执行完毕。")
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_result
                    })
                current_round += 1
            else:
                final_text = ai_message.content or ""
                final_text = re.sub(r'<｜DSML｜.*?(?:</｜DSML｜function_calls>|$)', '', final_text, flags=re.DOTALL).strip()
                logger.info(f"最终交付给用户的回复:\n{final_text}")
                status_box.update(label="思考完毕！", state="complete", expanded=False)
                st.markdown(final_text)
                
                st.session_state.messages.append({"role": "assistant", "content": final_text})
                save_session() # 拿到最终回答后再次保存
                break