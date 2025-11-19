import os
import json
import numpy as np
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import requests

# === 配置 ===
MODEL_NAME = "shibing624/text2vec-base-chinese"  # text2vec 官方模型（无需本地路径）
TOP_K_RETRIEVE = 3
TOP_K_CITE = 2
SIM_THRESHOLD = 0.5

# 加载嵌入模型（自动下载，Streamlit 支持）
print("正在加载 text2vec 模型...")
model = SentenceModel(MODEL_NAME)
print("✅ text2vec 模型加载完成")

# 加载知识库（JSON）
with open("knowledge_chunks.json", "r", encoding="utf-8") as f:
    ALL_CHUNKS = json.load(f)

# 转回 numpy 向量
for c in ALL_CHUNKS:
    c["embedding"] = np.array(c["embedding"], dtype=np.float32)
ALL_EMBEDDINGS = np.array([c["embedding"] for c in ALL_CHUNKS])


def retrieve_chunks(question: str, top_k=TOP_K_RETRIEVE):
    question_vec = model.encode([question])
    similarities = cosine_similarity(question_vec, ALL_EMBEDDINGS)[0]

    indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in indices:
        if similarities[idx] >= SIM_THRESHOLD:
            results.append(ALL_CHUNKS[idx])
    return results


def call_deepseek(prompt: str) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return "[错误] 未设置 DEEPSEEK_API_KEY"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个专业的金融知识助手，请严格基于用户提供的资料回答问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"[DeepSeek API 错误] {response.status_code}"
    except Exception as e:
        return f"[请求异常] {str(e)}"


def generate_answer(question: str):
    retrieved = retrieve_chunks(question)

    if not retrieved:
        return "根据现有资料无法确定。", []

    # 构造上下文
    context = "\n".join([f"- {chunk['text']}" for chunk in retrieved[:TOP_K_CITE]])
    prompt = f"""你是一个专业的金融知识助手，请严格基于以下资料回答问题。
- 必须忠实于资料内容，不得编造或推测。
- 回答应简洁、清晰、分点列出（如适用）。
- 如果资料中没有相关信息，请回答“根据现有资料无法确定”。

资料：
{context}

问题：{question}"""

    answer = call_deepseek(prompt)

    # 去重引用
    seen = set()
    citations = []
    for chunk in retrieved[:TOP_K_CITE]:
        key = (chunk["source_file"], chunk["page"])
        if key not in seen:
            citations.append({
                "file": chunk["source_file"],
                "page": chunk["page"]
            })
            seen.add(key)

    return answer, citations