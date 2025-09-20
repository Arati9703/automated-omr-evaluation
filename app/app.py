import streamlit as st
import pandas as pd
import numpy as np
import cv2, os, re, tempfile, sys

# --- Fix import path (so we can access src/ from app/)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.omr_utils import detect_bubbles

# --- Streamlit page setup
st.set_page_config(page_title="Automated OMR Evaluation", layout="wide")
st.title("📄 Automated OMR Evaluation & Scoring System")

# =====================
# Step 1: Upload Answer Key
# =====================
key_file = st.file_uploader("📑 Upload Answer Key (Excel)", type=["xlsx"])
answer_key = None

if key_file:
    raw_key = pd.read_excel(key_file)
    answer_key = []
    for subject in raw_key.columns:
        for cell in raw_key[subject].dropna():
            cell = str(cell).strip().lower()
            match = re.match(r"(\d+)\s*[-.\s]*\s*([a-d])", cell)
            if match:
                q_num = int(match.group(1))
                ans = match.group(2)
                answer_key.append((q_num, subject, ans))
    answer_key = pd.DataFrame(answer_key, columns=["QNo","Subject","Answer"]).sort_values("QNo").reset_index(drop=True)
    st.success(f"✅ Parsed {len(answer_key)} questions from key")

# =====================
# Step 2: Upload OMR Sheet Images
# =====================
uploaded_files = st.file_uploader("📷 Upload OMR Sheets", type=["jpg","jpeg","png"], accept_multiple_files=True)

results = []
if uploaded_files and answer_key is not None:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        # Run bubble detection
        marked = detect_bubbles(tmp_path, answer_key)
        student_id = os.path.splitext(file.name)[0]

        for _, row in answer_key.iterrows():
            q, subj, ans = row["QNo"], row["Subject"], row["Answer"]
            marked_ans = marked.get(q, None)   # safe get
            correct = 1 if marked_ans == ans else 0
            results.append([student_id, q, subj, marked_ans, ans, correct])

    # Convert to DataFrame
    results = pd.DataFrame(results, columns=["StudentID","QNo","Subject","Marked","Answer","Correct"])
    st.write("### 🔍 Sample Evaluations", results.head(10))

    # =====================
    # Step 3: Pivot Report
    # =====================
    report = results.groupby(["StudentID","Subject"])["Correct"].sum().reset_index()
    report = report.pivot(index="StudentID", columns="Subject", values="Correct").fillna(0).reset_index()
    report["TotalScore"] = report.drop(columns=["StudentID"]).sum(axis=1)

    st.write("### 📊 Final Report", report)

    # =====================
    # Step 4: Download Button
    # =====================
    st.download_button(
        "⬇️ Download CSV",
        report.to_csv(index=False).encode("utf-8"),
        "omr_results.csv",
        "text/csv"
    )

