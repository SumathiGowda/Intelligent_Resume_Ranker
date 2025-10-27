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


from typing import List, Dict
import os
import pandas as pd
def load_cleaned_resumes(test_csv_path: str, resumes_dir: str) -> List[Dict]:
    import fitz, re

    def extract_text_from_pdf(file_path):
        text = ""
        try:
            if os.path.getsize(file_path) == 0:  # check for empty file
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
        text = re.sub(r'\S+@\S+', ' ', text)        # remove emails
        text = re.sub(r'http\S+|www.\S+', ' ', text) # remove urls
        text = re.sub(r'\d+', ' ', text)            # remove numbers
        text = re.sub(r'[^a-z\s]', ' ', text)       # keep only alphabets
        text = re.sub(r'\s+', ' ', text).strip()    # remove extra spaces
        return text

    # Read CSV
    df = pd.read_csv(test_csv_path, encoding="utf-8-sig")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df['PDF Names'] = df['PDF Names'].astype(str).str.strip()

    # Label columns = everything except 'PDF Names'
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
        if not text:  # skip empty / unreadable
            continue

        labels = {c: int(row[c]) for c in label_cols}
        items.append({'name': resume_name, 'text': clean_text(text), 'labels': labels})

    return items


def normalize(name: str) -> str:
    return name.lower().replace(' ', '_')


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

    preds = []
    labels_all = []
    rows = []
    
    
    for item in resumes:
        resume_name = item['name']
        resume_emb = model.encode([item['text']], convert_to_tensor=True, show_progress_bar=False)
        scores = util.cos_sim(resume_emb, jd_embeddings)[0].cpu().numpy()

        # Ground truth
        label_map = {k: int(v) for k, v in item['labels'].items()}
        gt_vec = [label_map.get(jd_name, 0) for jd_name in jd_names]

        # Prediction: only the JD with max score above threshold
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



    # Overall metrics
    labels_arr = np.asarray(labels_all, dtype=np.int64)
    preds_arr = np.asarray(preds, dtype=np.int64)
    overall_metrics = compute_metrics(labels_arr, preds_arr)

    # Per-JD metrics
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
    parser.add_argument('--train_csv', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\all_data_label.csv")
    parser.add_argument('--train_resumes_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\Hundred_Resumes\all")
    #parser.add_argument('--test_csv', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\test.csv")
    #parser.add_argument('--test_resumes_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\Hundred_Resumes\test_resumes")
    parser.add_argument('--jds_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\JDs")
    parser.add_argument('--export_xlsx', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\evaluation_detailed.xlsx")
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT)
    args = parser.parse_args()

    print(f"Using threshold: {args.threshold}")
    model = SentenceTransformer(args.model_dir)
    jds = load_cleaned_jds(args.jds_dir)
    jd_order = list(jds.keys())

    # Train split
    train_rows: List[Dict] = []
    train_metrics: Dict[str, float] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    train_per_jd: Dict[str, Dict[str, float]] = {}
    if args.train_csv and args.train_resumes_dir:
        train_resumes = load_cleaned_resumes(args.train_csv, args.train_resumes_dir)
        train_rows, train_metrics, train_per_jd = evaluate_pairwise(model, train_resumes, jds, threshold=args.threshold)
        print(f"Train metrics @ threshold {args.threshold}: {train_metrics}")
    train_wide = rows_to_wide_df(train_rows, jd_order)

    # Test split
    test_rows: List[Dict] = []
    test_metrics: Dict[str, float] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    test_per_jd: Dict[str, Dict[str, float]] = {}
    if args.test_csv and args.test_resumes_dir:
        test_resumes = load_cleaned_resumes(args.test_csv, args.test_resumes_dir)
        test_rows, test_metrics, test_per_jd = evaluate_pairwise(model, test_resumes, jds, threshold=args.threshold)
        print(f"Test metrics @ threshold {args.threshold}: {test_metrics}")
    test_wide = rows_to_wide_df(test_rows, jd_order)

    # Build metrics DataFrame (overall + per-JD)
       # Build train metrics table (drop ALL row)
    train_metrics_rows = []
    for jd, m in train_per_jd.items():
        train_metrics_rows.append({"jd": jd, "threshold": args.threshold, **m})
    train_metrics_df = pd.DataFrame(train_metrics_rows)

    # Build test metrics table (drop ALL row)
    test_metrics_rows = []
    for jd, m in test_per_jd.items():
        test_metrics_rows.append({"jd": jd, "threshold": args.threshold, **m})
    test_metrics_df = pd.DataFrame(test_metrics_rows)

    # Overall train vs test
    overall_metrics_df = pd.DataFrame([
        {"split": "train", "threshold": args.threshold, **train_metrics},
        {"split": "test", "threshold": args.threshold, **test_metrics},
    ])

    if args.export_xlsx:
        print(f"Exporting train/test wide and metrics to {args.export_xlsx} ...")
        with pd.ExcelWriter(args.export_xlsx, engine="openpyxl") as writer:
            train_wide.to_excel(writer, index=False, sheet_name="train_wide")
            test_wide.to_excel(writer, index=False, sheet_name="test_wide")

            # Write labelled blocks in metrics sheet
            startrow = 0
            pd.DataFrame(["Train JD Metrics"]).to_excel(writer, index=False, header=False,
                                                       sheet_name="metrics", startrow=startrow)
            startrow += 1
            train_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

            startrow += len(train_metrics_df) + 3
            pd.DataFrame(["Test JD Metrics"]).to_excel(writer, index=False, header=False,
                                                      sheet_name="metrics", startrow=startrow)
            startrow += 1
            test_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

            startrow += len(test_metrics_df) + 3
            pd.DataFrame(["Overall Metrics"]).to_excel(writer, index=False, header=False,
                                                      sheet_name="metrics", startrow=startrow)
            startrow += 1
            overall_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

        print("Export complete.")



if __name__ == '__main__':
    main()
