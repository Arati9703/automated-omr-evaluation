
import numpy as np

def detect_bubbles(img_path, answer_key):
    """
    Dummy bubble detection for hackathon demo.
    Replace this with actual OpenCV/CNN detection pipeline.
    """
    np.random.seed(0)
    return {q: np.random.choice(list("abcd")) for q in range(1, len(answer_key)+1)}
