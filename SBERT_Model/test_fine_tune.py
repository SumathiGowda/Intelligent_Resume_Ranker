import os
import fitz
import re
from sentence_transformers import SentenceTransformer, util
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ---------------- Helper Functions ----------------
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)      # remove emails
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove links
    text = re.sub(r'\d+', ' ', text)          # remove numbers
    text = re.sub(r'[^a-z\s]', ' ', text)     # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------- Main Function ----------------
def check_resume_vs_jd(resume_pdf: str, jd_pdf: str, model_dir: str, threshold: float = 0.5):
    # Load fine-tuned model
    model = SentenceTransformer(model_dir)

    # Extract & clean text
    resume_text = clean_text(extract_text_from_pdf(resume_pdf))
    jd_text = clean_text(extract_text_from_pdf(jd_pdf))

    # Encode
    resume_emb = model.encode([resume_text], convert_to_tensor=True)
    jd_emb = model.encode([jd_text], convert_to_tensor=True)

    # Cosine similarity
    score = util.cos_sim(resume_emb, jd_emb).item()

    # Prediction
    match = "YES ✅ Resume fits JD" if score >= threshold else "NO ❌ Resume does not fit JD"

    print(f"Similarity Score: {score:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Prediction: {match}")

    return score, match


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    resume_path = r"C:\Users\sumat\OneDrive\Desktop\resume4.pdf"
    jd_path = r"C:\Users\sumat\OneDrive\Desktop\jd2.pdf"
    model_path = r"C:\Users\sumat\OneDrive\Desktop\Project\saved_model_v1"  # your fine-tuned model folder

    check_resume_vs_jd(resume_path, jd_path, model_path, threshold=0.5)
