from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import base64
import requests
from rag_main import agent_run

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------在这里填入你的智谱GLM4V API Key--------------------------
ZHIPU_API_KEY = "密钥"

# 主页
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# 对话接口 Form表单接收，解决422报错
@app.post("/api/chat")
async def chat(user_msg: str = Form(...), role: str = Form(...)):
    try:
        resp = agent_run(user_msg, role)
        return {"code": 0, "data": resp}
    except Exception as e:
        return {"code": -1, "msg": str(e)}

# 图片上传识别接口
@app.post("/api/upload_img")
async def upload_img(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {ZHIPU_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "glm-4v-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "识别图片中的商品，只返回商品名称，简短回答"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ]
        }
        resp = requests.post("https://open.bigmodel.cn/api/paas/v4/chat/completions", headers=headers, json=payload, timeout=30)
        res_json = resp.json()
        goods_name = res_json["choices"][0]["message"]["content"].strip()

        agent_msg = f"图片识别到商品：{goods_name}，帮我查一下这个商品库存"
        agent_result = agent_run(agent_msg, role="customer")

        return {"code": 0, "goods_name": goods_name, "agent_reply": agent_result}
    except Exception as e:
        return {"code": -1, "msg": f"图片识别失败：{str(e)}"}

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="127.0.0.1", port=8000, reload=False)
