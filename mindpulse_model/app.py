import streamlit as st
from src.emoji_wrapper import predict_emotion, predict_sentiment
from src.emoji_utils import get_emoji_details
from src.database import save_result, get_history
import pandas as pd
import plotly.express as px
import re
from collections import Counter

# ===== PAGE CONFIG =====
st.set_page_config(page_title="MindPulse AI", layout="wide", page_icon="🧠")

# ===== GLOBAL FONT =====
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: "JetBrains Mono", monospace;
}
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.markdown("## 🧠 MindPulse AI")
page = st.sidebar.radio("Navigate", ["💬 Chat", "📊 Dashboard", "🕓 History"])

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ===== EMOTION CONFIG =====
EMOTION_CONFIG = {
    "joy":      {"color": "🟢", "score":  1.0},
    "happy":    {"color": "🟢", "score":  1.0},
    "love":     {"color": "❤️",  "score":  1.0},
    "surprise": {"color": "🟡", "score":  0.5},
    "neutral":  {"color": "⚪", "score":  0.0},
    "fear":     {"color": "🟠", "score": -0.5},
    "sad":      {"color": "🔵", "score": -1.0},
    "sadness":  {"color": "🔵", "score": -1.0},
    "anger":    {"color": "🔴", "score": -1.0},
}

emotion_colors = {k: v["color"] for k, v in EMOTION_CONFIG.items()}
score_map      = {k: v["score"] for k, v in EMOTION_CONFIG.items()}

# ===== HEADER =====
st.markdown("""
<h1 style='text-align: center;'>🧠 MindPulse AI</h1>
<p style='text-align: center; color: gray;'>Real-time Emotion Intelligence Platform</p>
""", unsafe_allow_html=True)

st.divider()

# ===== SENTENCE SPLITTER =====
def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

# ================= CHAT =================
if page == "💬 Chat":
    st.subheader("💬 Emotion Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Type your message...")

    if user_input:
        sentences = split_sentences(user_input)
        for s in sentences:
            try:
                emotion, conf     = predict_emotion(s)
                sentiment, s_conf = predict_sentiment(s)
                emoji_details     = get_emoji_details(s)

                save_result(s, emotion, sentiment, conf)

                st.session_state.messages.append({
                    "text":          s,
                    "emotion":       emotion,
                    "sentiment":     sentiment,
                    "confidence":    conf,
                    "emoji_details": emoji_details,
                })
            except Exception as e:
                st.warning(f"Could not analyze: '{s}' — {str(e)}")

    emotions_list = []

    for msg in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(msg["text"])

        with st.chat_message("assistant"):
            emoji_icon = emotion_colors.get(msg["emotion"].lower(), "⚪")
            st.markdown(f"""
**Emotion:** {emoji_icon} `{msg['emotion']}`
**Sentiment:** `{msg['sentiment']}`
""")
            conf = msg["confidence"]
            if conf > 0.8:
                st.progress(conf, text="🟢 High Confidence")
            elif conf > 0.5:
                st.progress(conf, text="🟡 Medium Confidence")
            else:
                st.progress(conf, text="🔴 Low Confidence")

            if msg.get("emoji_details"):
                with st.expander("🔍 Emoji Breakdown"):
                    for d in msg["emoji_details"]:
                        e_icon = emotion_colors.get(d["emotion"], "⚪")
                        st.markdown(
                            f"**{d['emoji']}** `{d['description']}` "
                            f"→ {e_icon} `{d['emotion']}` "
                            f"({d['confidence']:.0%})"
                        )

        emotions_list.append(msg["emotion"])

    if emotions_list:
        st.divider()
        most_common = Counter(emotions_list).most_common(1)[0][0]
        st.subheader("📝 Overall Summary")
        emoji_icon = emotion_colors.get(most_common.lower(), "⚪")
        st.success(f"Overall Emotion: {emoji_icon} **{most_common}**")

        if len(emotions_list) > 1:
            counts = Counter(emotions_list)
            cols   = st.columns(min(len(counts), 6))
            for i, (em, cnt) in enumerate(counts.most_common(6)):
                icon = emotion_colors.get(em.lower(), "⚪")
                cols[i].metric(f"{icon} {em}", cnt)

# ================= DASHBOARD =================
elif page == "📊 Dashboard":
    st.subheader("📊 Analytics Dashboard")
    data = get_history()

    if len(data) == 0:
        st.warning("No data yet — start chatting to see analytics!")
    else:
        df = pd.DataFrame(
            data,
            columns=["id", "text", "emotion", "sentiment", "confidence"]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Entries",   len(df))
        col2.metric("Unique Emotions", df["emotion"].nunique())
        col3.metric("Avg Confidence",  f"{df['confidence'].mean():.0%}")
        col4.metric("Positive Ratio",
                    f"{(df['sentiment'] == 'POSITIVE').mean():.0%}")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            ec = df["emotion"].value_counts().reset_index()
            ec.columns = ["emotion", "count"]
            fig = px.bar(ec, x="emotion", y="count",
                         title="Emotion Distribution", color="emotion")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.pie(df, names="sentiment",
                          title="Sentiment Distribution",
                          color="sentiment",
                          color_discrete_map={
                              "POSITIVE": "#22c55e",
                              "NEGATIVE": "#ef4444"
                          })
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        scores = [score_map.get(e.lower(), 0) for e in df["emotion"]]
        fig3 = px.line(y=scores, title="Emotional Trend")
        fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig3.update_layout(yaxis=dict(range=[-1.2, 1.2]))
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.histogram(df, x="confidence", nbins=20,
                            title="Confidence Distribution",
                            color_discrete_sequence=["#6366f1"])
        st.plotly_chart(fig4, use_container_width=True)

# ================= HISTORY =================
elif page == "🕓 History":
    st.subheader("🕓 History")
    data = get_history()

    if len(data) == 0:
        st.warning("No history available yet")
    else:
        df = pd.DataFrame(
            data,
            columns=["id", "text", "emotion", "sentiment", "confidence"]
        )

        col1, col2 = st.columns(2)
        with col1:
            ef = st.multiselect("Filter by emotion",
                                options=sorted(df["emotion"].unique()),
                                default=[])
        with col2:
            sf = st.multiselect("Filter by sentiment",
                                options=sorted(df["sentiment"].unique()),
                                default=[])

        if ef: df = df[df["emotion"].isin(ef)]
        if sf: df = df[df["sentiment"].isin(sf)]

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="mindpulse_history.csv",
                           mime="text/csv")