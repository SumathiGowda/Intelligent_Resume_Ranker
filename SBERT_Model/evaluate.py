import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util

# Single source of truth for default threshold
THRESHOLD_DEFAULT: float = 0.5


def load_cleaned_jds(jds_dir: str) -> Dict[str, str]:
    import fitz, re

    def extract_text_from_pdf(file_path):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text")
        return text

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'http\S+|www.\S+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    jds: Dict[str, str] = {}
    for jd_file in os.listdir(jds_dir):
        if jd_file.endswith('.pdf'):
            role = jd_file.replace('.pdf', '')
            jds[role] = clean_text(extract_text_from_pdf(os.path.join(jds_dir, jd_file)))
    return jds


def load_cleaned_resumes(csv_path: str, resumes_dir: str) -> List[Dict]:
    import fitz, re

    def extract_text_from_pdf(file_path):
        text = ""
        try:
            if os.path.getsize(file_path) == 0:
                print(f"[WARNING] Skipping empty file: {os.path.basename(file_path)}")
                return ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text("text")
        except Exception as e:
            print(f"[WARNING] Could not read {os.path.basename(file_path)}: {e}")
            return ""
        return text

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'http\S+|www.\S+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df['PDF Names'] = df['PDF Names'].astype(str).str.strip()

    label_cols = [c for c in df.columns if c != 'PDF Names']
    df[label_cols] = df[label_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    items = []
    for _, row in df.iterrows():
        resume_name = str(row['PDF Names']).strip()
        file_name = resume_name if resume_name.lower().endswith(".pdf") else resume_name + ".pdf"
        file_path = os.path.join(resumes_dir, file_name)

        if not os.path.exists(file_path):
            print(f"[WARNING] Resume not found: {file_name}")
            continue

        text = extract_text_from_pdf(file_path)
        if not text:
            continue

        labels = {c: int(row[c]) for c in label_cols}
        items.append({'name': resume_name, 'text': clean_text(text), 'labels': labels})

    return items


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    accuracy = float((labels == preds).mean()) if labels.size else 0.0
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max((precision + recall), 1e-8))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def evaluate_pairwise(model: SentenceTransformer, resumes: List[Dict], jds: Dict[str, str], threshold: float = THRESHOLD_DEFAULT):
    jd_names = list(jds.keys())
    jd_texts = [jds[name] for name in jd_names]
    jd_embeddings = model.encode(jd_texts, convert_to_tensor=True, batch_size=16, show_progress_bar=False)

    preds, labels_all, rows = [], [], []

    for item in resumes:
        resume_name = item['name']
        resume_emb = model.encode([item['text']], convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(resume_emb, jd_embeddings)[0].cpu().numpy()

        label_map = {k: int(v) for k, v in item['labels'].items()}
        gt_vec = [label_map.get(jd_name, 0) for jd_name in jd_names]

        max_idx = np.argmax(scores)
        pred_vec = np.zeros_like(scores, dtype=int)
        if scores[max_idx] >= threshold:
            pred_vec[max_idx] = 1

        labels_all.extend(gt_vec)
        preds.extend(list(pred_vec))

        for idx, jd_name in enumerate(jd_names):
            rows.append({
                "resume": resume_name,
                "jd_compared_with": jd_name,
                "ground_truth": int(gt_vec[idx]),
                "predicted": int(pred_vec[idx]),
                "similarity_score": float(scores[idx])
            })

    labels_arr = np.asarray(labels_all, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    overall_metrics = compute_metrics(labels_arr, preds_arr)

    per_jd_metrics = {}
    for jd in jd_names:
        jd_rows = [r for r in rows if r['jd_compared_with'] == jd]
        jd_labels = np.asarray([r['ground_truth'] for r in jd_rows], dtype=np.int64)
        jd_preds = np.asarray([r['predicted'] for r in jd_rows], dtype=np.int64)
        per_jd_metrics[jd] = compute_metrics(jd_labels, jd_preds)

    return rows, overall_metrics, per_jd_metrics


def rows_to_wide_df(rows: List[Dict], jd_order: List[str]) -> pd.DataFrame:
    df_pairs = pd.DataFrame(rows)
    if df_pairs.empty:
        return pd.DataFrame(columns=["resume"] + [c for jd in jd_order for c in (f"{jd}_gt", f"{jd}_pred", f"{jd}_score")])
    resumes = list(df_pairs['resume'].drop_duplicates().tolist())
    wide_rows = []
    for res in resumes:
        row_dict: Dict[str, object] = {"resume": res}
        res_slice = df_pairs[df_pairs['resume'] == res]
        for jd in jd_order:
            cell = res_slice[res_slice['jd_compared_with'] == jd]
            if not cell.empty:
                gt = int(cell.iloc[0]['ground_truth'])
                pred = int(cell.iloc[0]['predicted'])
                score = float(cell.iloc[0]['similarity_score'])
            else:
                gt = 0
                pred = 0
                score = float('nan')
            row_dict[f"{jd}_gt"] = gt
            row_dict[f"{jd}_pred"] = pred
            row_dict[f"{jd}_score"] = score
        wide_rows.append(row_dict)
    result = pd.DataFrame(wide_rows)
    ordered_cols = ["resume"]
    for jd in jd_order:
        ordered_cols.extend([f"{jd}_gt", f"{jd}_pred", f"{jd}_score"])
    return result[ordered_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\sbert_model")
    parser.add_argument('--csv', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\all_data_label.csv")
    parser.add_argument('--resumes_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\Hundred_Resumes\all")
    parser.add_argument('--jds_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\JDs")
    parser.add_argument('--export_xlsx', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\evaluation_all_data.xlsx")
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT)
    args = parser.parse_args()

    print(f"Using threshold: {args.threshold}")
    model = SentenceTransformer(args.model_dir)
    jds = load_cleaned_jds(args.jds_dir)
    jd_order = list(jds.keys())

    # All data evaluation
    all_resumes = load_cleaned_resumes(args.csv, args.resumes_dir)
    rows, metrics, per_jd = evaluate_pairwise(model, all_resumes, jds, threshold=args.threshold)
    print(f"All data metrics @ threshold {args.threshold}: {metrics}")

    wide_df = rows_to_wide_df(rows, jd_order)

    # Build metrics tables
    per_jd_df = pd.DataFrame([{"jd": jd, "threshold": args.threshold, **m} for jd, m in per_jd.items()])
    overall_df = pd.DataFrame([{"split": "all_data", "threshold": args.threshold, **metrics}])

    if args.export_xlsx:
        print(f"Exporting results to {args.export_xlsx} ...")
        with pd.ExcelWriter(args.export_xlsx, engine="openpyxl") as writer:
            wide_df.to_excel(writer, index=False, sheet_name="all_data_wide")

            startrow = 0
            pd.DataFrame(["Per JD Metrics"]).to_excel(writer, index=False, header=False, sheet_name="metrics", startrow=startrow)
            startrow += 1
            per_jd_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

            startrow += len(per_jd_df) + 3
            pd.DataFrame(["Overall Metrics"]).to_excel(writer, index=False, header=False, sheet_name="metrics", startrow=startrow)
            startrow += 1
            overall_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

        print("Export complete.")


if __name__ == '__main__':
    main()
