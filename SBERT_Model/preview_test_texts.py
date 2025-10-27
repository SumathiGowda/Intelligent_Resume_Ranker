import os
import argparse
import pandas as pd
import fitz  # PyMuPDF
import re
from typing import Dict, List

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)  # remove emails
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove URLs
    text = re.sub(r'\d+', ' ', text)  # remove numbers
    text = re.sub(r'[^a-z\s]', ' ', text)  # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

def load_cleaned_jds(jds_dir: str) -> Dict[str, str]:
    jds: Dict[str, str] = {}
    for jd_file in os.listdir(jds_dir):
        if jd_file.lower().endswith('.pdf'):
            role = jd_file.rsplit('.', 1)[0]
            jds[role] = clean_text(extract_text_from_pdf(os.path.join(jds_dir, jd_file)))
    return jds

def load_cleaned_resumes(csv_path: str, resumes_dir: str) -> List[Dict]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")  # handles BOM
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    
    # Find the resume column
    resume_col = None
    for col in df.columns:
        if 'resume' in col.lower() or 'pdf' in col.lower():
            resume_col = col
            break
    if resume_col is None:
        raise ValueError(f"No resume column found in CSV. Columns: {df.columns.tolist()}")
    
    df.rename(columns={resume_col: 'RESUME'}, inplace=True)
    df['RESUME'] = df['RESUME'].astype(str).str.strip()

    # Label columns = everything except 'RESUME'
    label_cols = [c for c in df.columns if c != 'RESUME']
    df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    items = []
    for _, row in df.iterrows():
        resume_name = str(row['RESUME']).strip()
        file_name = resume_name if resume_name.lower().endswith(".pdf") else resume_name + ".pdf"
        file_path = os.path.join(resumes_dir, file_name)

        if os.path.exists(file_path):
            text = clean_text(extract_text_from_pdf(file_path))
            labels = {c: int(row[c]) for c in label_cols}
            items.append({'name': resume_name, 'text': text, 'labels': labels})
        else:
            print(f"⚠️ Resume not found: {file_name}")
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\test.csv")
    parser.add_argument('--test_resumes_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\Hundred_Resumes\test_resumes")
    parser.add_argument('--jds_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\JDs")
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--resume_chars', type=int, default=10000)
    parser.add_argument('--jd_chars', type=int, default=1000)
    args = parser.parse_args()

    print("Loading JDs and resumes...")
    jds = load_cleaned_jds(args.jds_dir)
    resumes = load_cleaned_resumes(args.test_csv, args.test_resumes_dir)

    if not resumes:
        print("⚠️ No resumes loaded. Check CSV and PDF folder paths.")
        return

    num_samples = args.num_samples or len(resumes)
    resume_chars = args.resume_chars
    jd_chars = args.jd_chars

    print(f"\nShowing {min(num_samples, len(resumes))} test samples (cleaned):\n")
    for i, item in enumerate(resumes[:num_samples], start=1):
        print(f"=== Resume {i}: {item['name']} ===")
        print(f"Resume text (first {resume_chars} chars):")
        print(item['text'][:resume_chars] or "<empty>")
        print()
        for jd_name, jd_text in jds.items():
            print(f"JD: {jd_name} (first {jd_chars} chars):")
            print(jd_text[:jd_chars] or "<empty>")
            print()
        print("-" * 80)

if __name__ == '__main__':
    main()
