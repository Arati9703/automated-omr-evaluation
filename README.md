# automated-omr-evaluation
%%writefile README.md
# Automated OMR Evaluation & Scoring System

## Problem
Manual OMR evaluation is slow, error-prone. We automate evaluation using Computer Vision + Python.

## Approach
- Preprocessing with OpenCV
- Bubble detection (CV / CNN)
- Answer key matching
- Score calculation per subject + total
- Streamlit web app for deployment

## How to Run
1. Clone repo
2. Install dependencies
   ```bash
   pip install -r requirements.txt
