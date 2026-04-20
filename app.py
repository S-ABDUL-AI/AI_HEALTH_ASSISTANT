import os
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

REQUIRED_COLUMNS = ["symptom", "possible_condition", "recommended_action"]
RED_FLAG_TERMS = [
    "chest pain",
    "trouble breathing",
    "shortness of breath",
    "fainting",
    "passed out",
    "seizure",
    "stroke",
    "slurred speech",
    "severe bleeding",
    "suicidal",
    "overdose",
]
SAMPLE_FALLBACK_ROWS = [
    {
        "symptom": "headache",
        "possible_condition": "Migraine, dehydration, stress",
        "recommended_action": "Hydrate, rest in a dark room, monitor symptoms, and seek care if severe or persistent.",
    },
    {
        "symptom": "fever",
        "possible_condition": "Viral infection, flu, or other illness",
        "recommended_action": "Drink fluids, rest, and seek medical care if fever is high, persistent, or worsening.",
    },
    {
        "symptom": "shortness of breath",
        "possible_condition": "Asthma, infection, anxiety, or other urgent causes",
        "recommended_action": "If severe or sudden, seek emergency care immediately.",
    },
]


@st.cache_data(show_spinner="Loading health knowledge base...")
def load_health_table(path: str = "Health_dataset.csv") -> Tuple[pd.DataFrame, bool, str]:
    try:
        if not os.path.exists(path):
            fallback = pd.DataFrame(SAMPLE_FALLBACK_ROWS)
            return fallback, True, "Using fallback sample data because Health_dataset.csv is missing."

        raw = pd.read_csv(path)
        colmap = {c.lower().strip(): c for c in raw.columns}
        if not set(REQUIRED_COLUMNS).issubset(set(colmap.keys())):
            fallback = pd.DataFrame(SAMPLE_FALLBACK_ROWS)
            return fallback, True, "Using fallback sample data because the dataset schema is invalid."

        out = pd.DataFrame(
            {
                "symptom": raw[colmap["symptom"]].astype(str).str.strip(),
                "possible_condition": raw[colmap["possible_condition"]].astype(str).str.strip(),
                "recommended_action": raw[colmap["recommended_action"]].astype(str).str.strip(),
            }
        )
        out = out.replace({"": pd.NA}).dropna().drop_duplicates(subset=["symptom"]).reset_index(drop=True)
        if out.empty:
            fallback = pd.DataFrame(SAMPLE_FALLBACK_ROWS)
            return fallback, True, "Using fallback sample data because the dataset became empty after cleaning."
        return out, False, "Dataset loaded successfully."
    except Exception as exc:
        fallback = pd.DataFrame(SAMPLE_FALLBACK_ROWS)
        return fallback, True, f"Using fallback sample data because of data loading error: {exc}"


@st.cache_data(show_spinner=False)
def build_tfidf_corpus(symptoms: tuple) -> Tuple[Optional[TfidfVectorizer], Optional[object]]:
    if not symptoms:
        return None, None
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vec.fit_transform(list(symptoms))
    return vec, matrix


def detect_red_flags(user_input: str) -> list[str]:
    text = user_input.lower()
    return [term for term in RED_FLAG_TERMS if term in text]


@st.cache_data(ttl=300, show_spinner=False)
def check_openai_connection(api_key_fingerprint: str) -> Tuple[str, str]:
    """Check whether the OpenAI key works and the API is reachable."""
    if not OPENAI_API_KEY:
        return "not_configured", "No API key configured."
    try:
        test_client = OpenAI(api_key=OPENAI_API_KEY)
        # Lightweight connectivity/auth test.
        test_client.models.list()
        return "connected", "Connected. API key is valid."
    except Exception as exc:
        return "error", f"Connection failed: {exc}"


def match_rule(data: pd.DataFrame, user_input: str) -> Optional[Dict]:
    text = user_input.lower().strip()
    if not text:
        return None
    for _, row in data.iterrows():
        symptom = str(row["symptom"]).lower()
        if symptom and symptom in text:
            return {
                "source": "Direct match",
                "confidence": 0.96,
                "symptom": row["symptom"],
                "possible_condition": row["possible_condition"],
                "recommended_action": row["recommended_action"],
            }
    return None


def match_tfidf(data: pd.DataFrame, user_input: str, vec, matrix) -> Optional[Dict]:
    if vec is None or matrix is None:
        return None
    query = vec.transform([user_input])
    similarities = cosine_similarity(query, matrix).flatten()
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    if best_score < 0.12:
        return None
    row = data.iloc[best_idx]
    return {
        "source": "Similarity match",
        "confidence": best_score,
        "symptom": row["symptom"],
        "possible_condition": row["possible_condition"],
        "recommended_action": row["recommended_action"],
    }


def health_chatbot_gpt(user_input: str) -> str:
    if client is None:
        return (
            "OpenAI is not configured right now. You can still use the local knowledge base guidance above. "
            "If symptoms are severe, worsening, or concerning, seek care from a licensed clinician."
        )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious health information assistant. "
                        "Never diagnose and never present certainty about disease. "
                        "Provide plain-language supportive guidance, mention warning signs, and encourage professional care."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Patient description: {user_input}\nRespond in <=160 words using bullet points.",
                },
            ],
            temperature=0.4,
            max_tokens=260,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"AI service error: {exc}"


def compose_local_guidance(data: pd.DataFrame, vec, matrix, user_input: str) -> Optional[Dict]:
    rule = match_rule(data, user_input)
    if rule:
        return rule
    return match_tfidf(data, user_input, vec, matrix)


def estimate_triage_level(severity: int, duration_days: int, red_flag_count: int) -> str:
    if red_flag_count > 0:
        return "Emergency"
    if severity >= 8 or duration_days >= 10:
        return "Urgent"
    if severity >= 5 or duration_days >= 4:
        return "Soon"
    return "Monitor"


def triage_color(level: str) -> str:
    return {
        "Emergency": "#dc2626",
        "Urgent": "#ea580c",
        "Soon": "#ca8a04",
        "Monitor": "#059669",
    }.get(level, "#334155")


data, using_fallback, load_msg = load_health_table()
symptoms_tuple = tuple(data["symptom"].tolist())
vec, matrix = build_tfidf_corpus(symptoms_tuple)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%); }
    div.block-container { padding-top: 1rem; padding-bottom: 1.2rem; }
    .hero {
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        background: linear-gradient(90deg, #dbeafe 0%, #f8fafc 100%);
        padding: 12px 14px;
        color: #0f172a;
    }
    .assistant-card {
        border-radius: 14px;
        border-left: 6px solid #2563eb;
        background: #ffffff;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 14px 16px;
    }
    .alert-card {
        border-radius: 14px;
        border-left: 6px solid #dc2626;
        background: #fef2f2;
        border: 1px solid #fecaca;
        padding: 12px 14px;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 AI Health Assistant")
st.caption(
    "For informational support only. This tool does not provide medical diagnosis or treatment. "
    "If symptoms are severe or sudden, contact emergency services."
)
st.markdown(
    "<div class='hero'><strong>Challenge / problem statement:</strong> People often need quick, understandable health guidance "
    "before they can reach a clinician. This assistant helps organize symptom information and recommend safer next steps.</div>",
    unsafe_allow_html=True,
)

if using_fallback:
    st.warning(load_msg)

with st.sidebar:
    st.subheader("Assistant settings")
    key_fingerprint = OPENAI_API_KEY[-8:] if OPENAI_API_KEY else "none"
    conn_status, conn_message = check_openai_connection(key_fingerprint)
    default_ai_toggle = conn_status == "connected"
    use_llm = st.toggle("Use AI for additional context", value=default_ai_toggle, disabled=not default_ai_toggle)
    if conn_status == "connected":
        st.success(f"OpenAI status: connected. {conn_message}")
    elif conn_status == "not_configured":
        st.info("OpenAI status: not configured. Add `OPENAI_API_KEY` in secrets to enable AI context.")
    else:
        st.error(f"OpenAI status: error. {conn_message}")

    if st.button("Re-check OpenAI connection"):
        check_openai_connection.clear()
        st.rerun()
    st.divider()
    with st.expander("How to use this app", expanded=True):
        st.markdown(
            "1. Describe symptoms in plain language.\n"
            "2. Review emergency warning banner if shown.\n"
            "3. Check suggested condition and recommended action.\n"
            "4. Use triage section to estimate urgency.\n"
            "5. Seek clinician care for persistent or severe symptoms."
        )
    st.divider()
    st.markdown(
        "**Designed by:** Sherriff Abdul-Hamid  \n"
        "[GitHub](https://github.com/S-ABDUL-AI) · "
        "[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)"
    )

m1, m2, m3 = st.columns(3)
m1.metric("Knowledge base rows", f"{len(data):,}")
m2.metric("Distinct symptom entries", f"{data['symptom'].nunique():,}")
m3.metric("Emergency keywords monitored", f"{len(RED_FLAG_TERMS):,}")

tab1, tab2, tab3 = st.tabs(["Symptom assistant", "Quick triage", "Knowledge base"])

with tab1:
    st.subheader("Describe symptoms")
    quick_prompts = [
        "Headache and fatigue for 2 days",
        "Sore throat and fever since yesterday",
        "Cough with chest discomfort",
        "Stomach pain after meals",
        "Shortness of breath while resting",
        "Back pain after lifting heavy items",
    ]
    c1, c2, c3 = st.columns(3)
    prompt_choice = None
    for idx, prompt in enumerate(quick_prompts):
        if [c1, c2, c3][idx % 3].button(prompt, key=f"prompt_{idx}"):
            prompt_choice = prompt

    user_input = st.text_area(
        "Symptom description",
        value=prompt_choice or "",
        height=120,
        placeholder="Example: I have had fever and cough for 3 days, with low appetite.",
    )
    analyze = st.button("Analyze symptoms", type="primary")

    if analyze and user_input.strip():
        red_flags = detect_red_flags(user_input)
        if red_flags:
            st.markdown(
                "<div class='alert-card'><strong>Emergency warning signs detected:</strong> "
                + ", ".join(red_flags)
                + ". If this is severe or worsening, seek urgent medical care now.</div>",
                unsafe_allow_html=True,
            )

        local = compose_local_guidance(data, vec, matrix, user_input.strip())
        if local:
            st.markdown(
                (
                    "<div class='assistant-card'>"
                    f"<strong>Match source:</strong> {local['source']}<br>"
                    f"<strong>Confidence:</strong> {local['confidence']:.0%}<br><br>"
                    f"<strong>Possible condition (informational):</strong> {local['possible_condition']}<br><br>"
                    f"<strong>Suggested next steps:</strong> {local['recommended_action']}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "No strong local match found. Try adding clearer symptom details "
                "(duration, severity, associated symptoms)."
            )

        if use_llm:
            st.markdown("### Additional AI context")
            st.markdown(health_chatbot_gpt(user_input.strip()))

        st.markdown("### Safety reminders")
        st.markdown(
            "- This is educational guidance and not a diagnosis.\n"
            "- Seek urgent care for severe chest pain, breathing difficulty, confusion, or heavy bleeding.\n"
            "- If symptoms persist or worsen, contact a licensed healthcare professional."
        )

with tab2:
    st.subheader("Quick triage (non-diagnostic)")
    st.caption("Use this as a rough urgency helper while arranging professional care.")

    t1, t2, t3 = st.columns(3)
    with t1:
        severity = st.slider("Symptom severity (0-10)", 0, 10, 4)
    with t2:
        duration_days = st.slider("Duration (days)", 0, 30, 2)
    with t3:
        concern = st.selectbox("General trend", ["Improving", "No change", "Worsening"])

    triage_red_flags = st.multiselect(
        "Any critical signs present?",
        ["Chest pain", "Trouble breathing", "Confusion", "Fainting", "Severe bleeding", "Seizure"],
    )
    if concern == "Worsening":
        severity = min(10, severity + 1)

    level = estimate_triage_level(severity, duration_days, len(triage_red_flags))
    color = triage_color(level)
    st.markdown(
        f"<div style='border-left:6px solid {color};background:#fff;border-radius:12px;padding:12px 14px;"
        f"border:1px solid #e2e8f0;'><strong>Estimated urgency:</strong> {level}</div>",
        unsafe_allow_html=True,
    )

    if level == "Emergency":
        st.error("Seek emergency care immediately.")
    elif level == "Urgent":
        st.warning("Arrange same-day medical evaluation.")
    elif level == "Soon":
        st.info("Book a clinician visit soon and monitor closely.")
    else:
        st.success("Continue monitoring and use self-care guidance. Escalate if symptoms worsen.")

with tab3:
    st.subheader("Knowledge base and transparency")
    st.caption("Source table used by direct and similarity matching.")
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.markdown(
        "You can improve this assistant by expanding `Health_dataset.csv` with more symptom phrases, "
        "clearer actions, and localized clinical guidance."
    )
