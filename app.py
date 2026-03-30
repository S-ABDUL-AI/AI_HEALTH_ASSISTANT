import os
import time
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Page config must be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@st.cache_data
def load_health_table(path: str = "Health_dataset.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {"symptom", "possible_condition", "recommended_action"}
    if not need.issubset(set(c.lower() for c in df.columns)):
        return None
    # normalize column names to expected keys
    colmap = {c.lower(): c for c in df.columns}
    return pd.DataFrame(
        {
            "symptom": df[colmap["symptom"]].astype(str),
            "possible_condition": df[colmap["possible_condition"]].astype(str),
            "recommended_action": df[colmap["recommended_action"]].astype(str),
        }
    )


@st.cache_data
def build_tfidf_corpus(symptoms: tuple):
    if not symptoms:
        return None, None
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(list(symptoms))
    return vec, mat


def match_rule(data: pd.DataFrame, user_input: str) -> Optional[str]:
    u = user_input.lower().strip()
    if not u:
        return None
    for _, row in data.iterrows():
        s = str(row["symptom"]).lower()
        if s and s in u:
            return (
                f"**Symptom:** {row['symptom']}\n\n"
                f"**Possible condition (informational):** {row['possible_condition']}\n\n"
                f"**Suggested next steps:** {row['recommended_action']}"
            )
    return None


def match_tfidf(data: pd.DataFrame, user_input: str, vec, mat) -> Optional[str]:
    if vec is None or mat is None:
        return None
    q = vec.transform([user_input])
    sims = cosine_similarity(q, mat).flatten()
    i = int(sims.argmax())
    if sims[i] < 0.12:
        return None
    row = data.iloc[i]
    return (
        f"**Closest match in knowledge base** (similarity {sims[i]:.2f})\n\n"
        f"**Symptom:** {row['symptom']}\n\n"
        f"**Possible condition (informational):** {row['possible_condition']}\n\n"
        f"**Suggested next steps:** {row['recommended_action']}"
    )


def health_chatbot_gpt(user_input: str) -> str:
    if client is None:
        return (
            "No API key found. Add **OPENAI_API_KEY** in Streamlit Cloud secrets (or set the env var locally) "
            "for richer answers. The sidebar still shows rule-based and text-matching results when possible."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a cautious health information assistant. Never diagnose. "
                "Give short, plain-language guidance and urge seeing a clinician for serious or worsening symptoms."
            ),
        },
        {
            "role": "user",
            "content": f"User describes: {user_input}. Respond in under 180 words.",
        },
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.5,
            max_tokens=280,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"AI service error: {e}"


def compose_reply(data: pd.DataFrame, vec, mat, user_input: str, use_llm: bool) -> str:
    r = match_rule(data, user_input)
    if r:
        return r
    r2 = match_tfidf(data, user_input, vec, mat)
    if r2:
        return r2
    if use_llm:
        return health_chatbot_gpt(user_input)
    return (
        "No close match in the local table. Try different wording or add **OPENAI_API_KEY** for AI-assisted guidance. "
        "If you have severe pain, trouble breathing, confusion, or other emergency signs, seek urgent care."
    )


data = load_health_table()
if data is None:
    st.error(
        "Missing or invalid **Health_dataset.csv** (need columns: symptom, possible_condition, recommended_action)."
    )
    st.stop()

symptoms_tuple = tuple(data["symptom"].tolist())
vec, mat = build_tfidf_corpus(symptoms_tuple)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #ecfdf5 0%, #f8fafc 100%); }
    .chat-bubble {
        background: #ecfdf5; border-left: 4px solid #059669; padding: 14px 16px;
        border-radius: 12px; margin: 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }
    .user-bubble {
        background: #fff; border: 1px solid #e2e8f0; padding: 12px 14px;
        border-radius: 12px; margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 AI Health Assistant")
st.caption(
    "Informational only — not medical advice, diagnosis, or treatment. "
    "For emergencies, call your local emergency number."
)

with st.sidebar:
    st.subheader("Options")
    use_llm = st.toggle("Use OpenAI when no local match", value=bool(client))
    st.caption(f"OpenAI: {'configured' if client else 'not configured'}")
    st.divider()
    st.markdown(
        "**Sherriff Abdul-Hamid**  \n"
        "[GitHub](https://github.com/S-ABDUL-AI) · "
        "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
    )

common = ["Fever", "Cough", "Headache", "Fatigue", "Stomach pain", "Shortness of breath"]
st.subheader("Quick prompts")
cols = st.columns(3)
choice = None
for i, s in enumerate(common):
    if cols[i % 3].button(s, key=f"q_{s}"):
        choice = s

user_input = st.text_area("Describe symptoms or concerns", value=choice or "", height=100, placeholder="e.g. dull headache for 2 days…")
run = st.button("Get guidance", type="primary")

if run and user_input.strip():
    st.markdown(f"<div class='user-bubble'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
    ph = st.empty()
    for j in range(3):
        ph.markdown(f"⏳ Preparing response{'.' * (j + 1)}")
        time.sleep(0.25)
    reply = compose_reply(data, vec, mat, user_input.strip(), use_llm)
    ph.markdown(f"<div class='chat-bubble'><b>Assistant:</b><br/><br/>{reply}</div>", unsafe_allow_html=True)

with st.expander("Preview of knowledge base (first rows)"):
    st.dataframe(data.head(15), use_container_width=True)
