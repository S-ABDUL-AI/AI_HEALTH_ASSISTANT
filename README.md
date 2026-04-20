# AI Health Assistant

AI Health Assistant is a Streamlit app that provides clear, safety-focused health information for symptom guidance. It combines:

- local knowledge-base matching from `Health_dataset.csv`
- similarity search for close symptom phrases
- optional AI context with OpenAI (if configured)
- a non-diagnostic quick triage view

## Purpose

People often need simple, understandable guidance before they can access a clinician.  
This app helps users describe symptoms, review possible informational matches, and get safer next-step suggestions.

## Safety Notice

- This app is for informational support only.
- It does **not** provide diagnosis or treatment.
- If symptoms are severe, sudden, or worsening, seek urgent medical care.

## Features

- **Symptom assistant**
  - Direct symptom match from local dataset
  - TF-IDF similarity fallback with confidence score
  - Optional AI-generated context (when API key exists)
- **Quick triage**
  - Emergency / Urgent / Soon / Monitor urgency estimate
  - Red-flag checks for critical warning terms
- **Knowledge base**
  - Transparent view of source data used for guidance
- **Resilient data loading**
  - Validates dataset schema
  - Uses fallback sample rows if CSV is missing or invalid

## Project Files

- `app.py` - Streamlit app UI and core logic
- `Health_dataset.csv` - local health knowledge table
- `chatbt_model.py` - earlier semantic-search experiment script
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python runtime for deployment

## Expected Dataset Schema

`Health_dataset.csv` should include:

- `symptom`
- `possible_condition`
- `recommended_action`

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## OpenAI Setup (Optional)

Add `OPENAI_API_KEY` either:

- as environment variable locally, or
- in Streamlit Cloud secrets

Without this key, the app still works using local knowledge-base matching.

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Create a new Streamlit app linked to this repo.
3. Set the main file path to `app.py`.
4. Add `OPENAI_API_KEY` in app secrets (optional).
5. Deploy / reboot app.

## Recommended Next Improvements

- Add automated tests for triage and red-flag detection.
- Expand `Health_dataset.csv` with richer symptom variants.
- Add localization support for country-specific emergency guidance.
- Add analytics for top symptom queries and unresolved questions.

## Author

Sherriff Abdul-Hamid  
[GitHub](https://github.com/S-ABDUL-AI)  
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)
