import cv2
import numpy as np

def detect_bubbles(img_path, answer_key, subjects=5, questions_per_subject=20, choices=4):
    """
    Detect filled bubbles from an OMR sheet using grid-based segmentation.
    
    Args:
        img_path: Path to OMR image
        answer_key: DataFrame with columns ["QNo", "Subject", "Answer"]
        subjects: Number of subject columns
        questions_per_subject: Questions per subject
        choices: Number of options per question (default=4 for A–D)
    
    Returns:
        dict {QNo: marked_answer}
    """
    # --- Step 1: Read + preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1000, 1400))  # normalize size (tune as per sheet)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = thresh.shape
    cell_h = h // questions_per_subject
    cell_w = w // (subjects * choices)

    marked = {}
    
    for subj_idx in range(subjects):
        for q_idx in range(questions_per_subject):
            q_no = subj_idx * questions_per_subject + q_idx + 1

            # Extract 4 bubbles (choices A–D) for this question
            scores = []
            for choice_idx in range(choices):
                x1 = (subj_idx * choices + choice_idx) * cell_w
                y1 = q_idx * cell_h
                roi = thresh[y1:y1+cell_h, x1:x1+cell_w]
                fill_score = cv2.countNonZero(roi)
                scores.append(fill_score)

            # Pick the bubble with max filled pixels
            marked_choice = np.argmax(scores)
            marked[q_no] = "abcd"[marked_choice]

    return marked

