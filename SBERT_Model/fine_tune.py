import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


THRESHOLD_DEFAULT: float = 0.5

# ----------------------- Data Loading -----------------------
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

    jds = {}
    for jd_file in os.listdir(jds_dir):
        if jd_file.endswith('.pdf'):
            role = jd_file.replace('.pdf', '')
            jds[role] = clean_text(extract_text_from_pdf(os.path.join(jds_dir, jd_file)))
    return jds

def load_cleaned_resumes(csv_path: str, resumes_dir: str) -> List[Dict]:
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
        if os.path.exists(file_path):
            text = clean_text(extract_text_from_pdf(file_path))
            labels = {c: int(row[c]) for c in label_cols}
            items.append({'name': resume_name, 'text': text, 'labels': labels})
        else:
            print(f"⚠️ Resume not found: {file_name}")
    return items

# ----------------------- Prepare Training Examples -----------------------
def prepare_training_examples(resumes: List[Dict], jds: Dict[str, str]) -> List[InputExample]:
    examples = []
    for item in resumes:
        resume_text = item['text']
        for jd_name, jd_text in jds.items():
            label = float(item['labels'].get(jd_name, 0))
            examples.append(InputExample(texts=[resume_text, jd_text], label=label))
    return examples

# ----------------------- Evaluation -----------------------
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

        label_map = {k: int(v) for k, v in item['labels'].items()}
        gt_vec = [label_map.get(jd_name, 0) for jd_name in jd_names]
        pred_vec = (scores >= threshold).astype(int)

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
    overall_metrics = {
        "accuracy": float((labels_arr == preds_arr).mean()),
        "precision": float(((preds_arr & labels_arr).sum()) / max(preds_arr.sum(), 1)),
        "recall": float(((preds_arr & labels_arr).sum()) / max(labels_arr.sum(), 1)),
        "f1": float(2 * ((preds_arr & labels_arr).sum()) / max((preds_arr.sum() + labels_arr.sum()), 1e-8))
    }

    # Per-JD metrics
    per_jd_metrics = {}
    for jd in jd_names:
        jd_rows = [r for r in rows if r['jd_compared_with'] == jd]
        jd_labels = np.asarray([r['ground_truth'] for r in jd_rows], dtype=np.int64)
        jd_preds = np.asarray([r['predicted'] for r in jd_rows], dtype=np.int64)
        tp = int(((jd_preds == 1) & (jd_labels == 1)).sum())
        fp = int(((jd_preds == 1) & (jd_labels == 0)).sum())
        fn = int(((jd_preds == 0) & (jd_labels == 1)).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
        per_jd_metrics[jd] = {"accuracy": float((jd_labels == jd_preds).mean()), "precision": precision, "recall": recall, "f1": f1}

    return rows, overall_metrics, per_jd_metrics

def rows_to_wide_df(rows: List[Dict], jd_order: List[str]) -> pd.DataFrame:
    df_pairs = pd.DataFrame(rows)
    if df_pairs.empty:
        return pd.DataFrame(columns=["resume"] + [c for jd in jd_order for c in (f"{jd}_gt", f"{jd}_pred", f"{jd}_score")])
    resumes = list(df_pairs['resume'].drop_duplicates().tolist())
    wide_rows = []
    for res in resumes:
        row_dict = {"resume": res}
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
    ordered_cols = ["resume"]
    for jd in jd_order:
        ordered_cols.extend([f"{jd}_gt", f"{jd}_pred", f"{jd}_score"])
    return pd.DataFrame(wide_rows)[ordered_cols]
from sklearn.model_selection import train_test_split
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\sbert_model")
    parser.add_argument('--csv', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\all_data_label.csv")
    parser.add_argument('--resumes_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\Hundred_Resumes\all")
    parser.add_argument('--jds_dir', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\Small_dataset\JDs")
    parser.add_argument('--export_xlsx', type=str, default=r"C:\Users\sumat\OneDrive\Desktop\Project\evaluation_finetuned.xlsx")
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_size', type=float, default=0.2)  
    parser.add_argument('--skip_train', action='store_true', help="Skip training and only run evaluation")  
    args = parser.parse_args()

    # Load resumes
    all_resumes = load_cleaned_resumes(args.csv, args.resumes_dir)
    train_resumes, test_resumes = train_test_split(all_resumes, test_size=args.test_size, random_state=42)
    print(f"Train size: {len(train_resumes)}, Test size: {len(test_resumes)}")

    # Load model + JDs
    model = SentenceTransformer(args.model_dir)
    jds = load_cleaned_jds(args.jds_dir)
    jd_order = list(jds.keys())

    # ---------- Training (if not skipped) ----------
    if not args.skip_train:
        print("🚀 Training started...")
        train_examples = prepare_training_examples(train_resumes, jds)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  epochs=args.epochs,
                  warmup_steps=10)

        # Save updated model after training
        model.save("saved_model_v1")
        print("✅ Training complete, model saved.")
    else:
        print("⚡ Skipping training, using saved model directly.")

    # ---------- Evaluation ----------
    print("📊 Running evaluation...")
    train_rows, train_metrics, train_per_jd = evaluate_pairwise(model, train_resumes, jds, threshold=args.threshold)
    test_rows, test_metrics, test_per_jd = evaluate_pairwise(model, test_resumes, jds, threshold=args.threshold)

    train_wide = rows_to_wide_df(train_rows, jd_order)
    test_wide = rows_to_wide_df(test_rows, jd_order)

    # Metrics
    train_metrics_df = pd.DataFrame([{"jd": jd, "threshold": args.threshold, **m} for jd, m in train_per_jd.items()])
    test_metrics_df = pd.DataFrame([{"jd": jd, "threshold": args.threshold, **m} for jd, m in test_per_jd.items()])
    overall_metrics_df = pd.DataFrame([
        {"split": "train", "threshold": args.threshold, **train_metrics},
        {"split": "test", "threshold": args.threshold, **test_metrics}
    ])

    # Export results
    with pd.ExcelWriter(args.export_xlsx, engine="openpyxl") as writer:
        train_wide.to_excel(writer, index=False, sheet_name="train_wide")
        test_wide.to_excel(writer, index=False, sheet_name="test_wide")

        startrow = 0
        pd.DataFrame(["Train JD Metrics"]).to_excel(writer, index=False, header=False, sheet_name="metrics", startrow=startrow)
        startrow += 1
        train_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

        startrow += len(train_metrics_df) + 3
        pd.DataFrame(["Test JD Metrics"]).to_excel(writer, index=False, header=False, sheet_name="metrics", startrow=startrow)
        startrow += 1
        test_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

        startrow += len(test_metrics_df) + 3
        pd.DataFrame(["Overall Metrics"]).to_excel(writer, index=False, header=False, sheet_name="metrics", startrow=startrow)
        startrow += 1
        overall_metrics_df.to_excel(writer, index=False, sheet_name="metrics", startrow=startrow)

    print(f"✅ Evaluation complete. Results saved to {args.export_xlsx}")

if __name__ == '__main__':
    main()
