import streamlit as st
import os
from rag_api import generate_answer

# 从 Streamlit Secrets 安全读取 API Key
os.environ["DEEPSEEK_API_KEY"] = st.secrets.get("DEEPSEEK_API_KEY", "")

st.set_page_config(page_title="智能问答系统", layout="centered")
st.title("📄 金融/政务智能问答")
st.caption("基于本地知识库 + DeepSeek 大模型")

question = st.text_input("请输入问题：", placeholder="例如：医保报销需要哪些材料？")

if st.button("提问") and question.strip():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        st.error("❌ 未配置 DEEPSEEK_API_KEY，请在 Streamlit Cloud 的 Settings → Secrets 中添加。")
    else:
        with st.spinner("检索知识库并生成答案..."):
            answer, citations = generate_answer(question.strip())

        st.success("✅ 答案")
        st.write(answer)

        if citations:
            st.info("📎 参考来源")
            for c in citations:
                st.write(f"- 📄 {c['file']} (第 {c['page']} 页)")
        else:
            st.info("📎 无相关文档匹配")