import cv2
import numpy as np

def detect_bubbles(img_path, answer_key, debug=False):
    """
    Detect bubbles using contour detection.
    If debug=True, returns (marked, debug_img) for visualization.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # Find contours
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if 20 <= w <= 50 and 20 <= h <= 50 and 0.8 <= ar <= 1.2:
            bubble_cnts.append(c)

    # Sort bubbles
    bubble_cnts = sorted(bubble_cnts, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    marked = {}
    questions = len(answer_key)
    debug_img = img.copy()

    for q_idx in range(questions):
        q_no = answer_key.iloc[q_idx]["QNo"]

        # Get 4 choices
        choices = bubble_cnts[q_idx*4:(q_idx+1)*4]
        scores = []
        for i, c in enumerate(choices):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            score = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            scores.append(score)

            # draw contours (blue outline for all)
            cv2.drawContours(debug_img, [c], -1, (255, 0, 0), 2)

        if scores:
            marked_choice = np.argmax(scores)
            marked[q_no] = "abcd"[marked_choice]

            # highlight selected choice (green)
            cv2.drawContours(debug_img, [choices[marked_choice]], -1, (0, 255, 0), 3)

    if debug:
        return marked, debug_img
    return marked
