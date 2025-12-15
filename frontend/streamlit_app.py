import time
import requests
import streamlit as st


st.set_page_config(
    page_title="Project 0 Sentiment Demo",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Project 0: Sentiment Classification Demo")
st.caption("Streamlit → FastAPI → Transformers (fine-tuned DistilRoBERTa on IMDb)")

# ==== Sidebar: API 配置 ====
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "FastAPI base URL",
        value="http://127.0.0.1:8000",
        help="确保 FastAPI 已经运行在这个地址上（默认是 127.0.0.1:8000）。",
    )
    timeout = st.slider("Request timeout (seconds)", 1, 60, 15)

st.divider()

# ==== 主界面：输入区 ====
default_text = "This movie was absolutely fantastic, I loved every minute of it!"
text = st.text_area(
    "Enter text",
    value=default_text,
    height=160,
    placeholder="Type an English review here...",
)

col1, col2 = st.columns([1, 2])
with col1:
    predict_btn = st.button("Predict", type="primary")
with col2:
    st.write("Tip: keep FastAPI running: `uvicorn app.main:app --reload`")

# ==== 调用 FastAPI 的函数 ====
def call_api_predict(base_url: str, text: str, timeout_s: int):
    url = base_url.rstrip("/") + "/predict"
    payload = {"text": text}
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout_s)
    latency_ms = (time.time() - t0) * 1000.0
    return resp, latency_ms


# ==== 触发预测 ====
if predict_btn:
    if not text.strip():
        st.error("Text is empty. Please input something.")
        st.stop()

    with st.spinner("Calling FastAPI /predict ..."):
        try:
            resp, latency_ms = call_api_predict(api_url, text, timeout)
        except requests.exceptions.ConnectionError:
            st.error(
                "Connection error: cannot reach FastAPI.\n\n"
                "请确认你已经启动了服务：\n"
                "`uvicorn app.main:app --reload`\n"
                "并且 URL 设置正确。"
            )
            st.stop()
        except requests.exceptions.Timeout:
            st.error("Request timeout. Try increasing timeout or check server load.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    # ==== 处理响应 ====
    if resp.status_code != 200:
        st.error(f"FastAPI returned error (status={resp.status_code})")
        try:
            st.json(resp.json())
        except Exception:
            st.text(resp.text)
        st.stop()

    data = resp.json()

    label = data.get("label", "unknown")
    score = float(data.get("score", 0.0))
    probs = data.get("probs", {})

    st.success(f"Done! Latency: {latency_ms:.1f} ms")

    # ==== 展示结果 ====
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted label", label)
    with c2:
        st.metric("Confidence", f"{score:.4f}")

    st.subheader("Probabilities")
    if isinstance(probs, dict) and probs:
        # 按概率从高到低排序
        probs_sorted = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        st.bar_chart(probs_sorted)
        st.json(probs_sorted)
    else:
        st.info("No probability details returned.")

    with st.expander("Raw response JSON"):
        st.json(data)
