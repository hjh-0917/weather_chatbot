# train_model.py
import json
import (pathlib as _pathlib)
from pathlib import Path
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib
from model import OutfitNet

JSON_ROOT = Path(__file__).parent / "JSON"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def collect_samples(json_root: Path):
    """
    JSON 폴더 구조에서 (input_features dict, output dict) 샘플을 수집.
    반환: list of (features_dict, outputs_dict)
    """
    samples = []
    # Walk through persona dirs etc.
    for age_dir in json_root.iterdir():
        if not age_dir.is_dir(): continue
        for persona_file in age_dir.iterdir():
            if persona_file.suffix != ".json": continue
            persona = persona_file.stem  # e.g. 'trendy'
            try:
                data = json.loads(persona_file.read_text(encoding='utf-8'))
            except Exception:
                continue
            # structure: persona -> gender -> age_label -> weather_cat -> skin -> body -> outfit(dict/list)
            node_persona = data.get(persona, {})
            for gender, node_gender in node_persona.items():
                for age_label, node_age in node_gender.items():
                    for weather_cat, node_weather in node_age.items():
                        for skin, node_skin in node_weather.items():
                            for body, outfit in node_skin.items():
                                # outfit may be dict or list
                                if isinstance(outfit, list):
                                    for item in outfit:
                                        samples.append((
                                            {"persona": persona, "gender": gender, "age_label": age_label,
                                             "weather_cat": weather_cat, "skin": skin, "body": body},
                                            item
                                        ))
                                elif isinstance(outfit, dict):
                                    samples.append((
                                        {"persona": persona, "gender": gender, "age_label": age_label,
                                         "weather_cat": weather_cat, "skin": skin, "body": body},
                                        outfit
                                    ))
    return samples

class OutfitDataset(Dataset):
    def __init__(self, df, feature_cols, encoders, target_cols):
        """
        df: pandas DataFrame containing feature cols and target cols (strings)
        encoders: dict with "onehot" or label encoders
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.encoders = encoders
        self.target_cols = target_cols

        # Build numeric X using OneHotEncoder for categorical feature set
        # We'll use sklearn OneHotEncoder fitted outside
        self.X = encoders['onehot'].transform(self.df[feature_cols]).toarray().astype('float32')
        # targets: label encoded arrays per target
        self.y = {}
        for t in target_cols:
            le = encoders['le_'+t]
            self.y[t] = le.transform(self.df[t].fillna("")).astype('int64')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {t: int(self.y[t][idx]) for t in self.target_cols}
        return x, y

def prepare_encoders(df, feature_cols, target_cols):
    # OneHotEncoder for features
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
    ohe.fit(df[feature_cols])
    encoders = {'onehot': ohe}
    # LabelEncoders for each target
    for t in target_cols:
        le = LabelEncoder()
        le.fit(df[t].fillna(""))
        encoders['le_'+t] = le
    return encoders

def build_dataframe(samples):
    rows = []
    for feat, out in samples:
        row = {}
        row.update(feat)
        # Ensure target columns exist
        row['상의'] = out.get('상의', '') if isinstance(out, dict) else ''
        row['하의'] = out.get('하의', '') if isinstance(out, dict) else ''
        row['신발'] = out.get('신발', '') if isinstance(out, dict) else ''
        row['액세서리'] = out.get('액세서리', '') if isinstance(out, dict) else ''
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def train(args):
    samples = collect_samples(JSON_ROOT)
    if not samples:
        print("No samples found in JSON folder. Exiting.")
        return
    df = build_dataframe(samples)
    feature_cols = ["persona", "gender", "age_label", "weather_cat", "skin", "body"]
    target_cols = ["상의", "하의", "신발", "액세서리"]

    encoders = prepare_encoders(df, feature_cols, target_cols)
    # Save encoders
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")

    ds = OutfitDataset(df, feature_cols, encoders, target_cols)
    batch_size = args.batch_size
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    input_dim = ds.X.shape[1]
    # head sizes:
    head_sizes = { "상의": len(encoders['le_상의'].classes_),
                  "하의": len(encoders['le_하의'].classes_),
                  "신발": len(encoders['le_신발'].classes_),
                  "액세서리": len(encoders['le_액세서리'].classes_) }
    model = OutfitNet(input_dim=input_dim, hidden_dim=args.hidden_dim, head_sizes=head_sizes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterions = {t: nn.CrossEntropyLoss() for t in target_cols}

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            X_batch = torch.tensor(X_batch, dtype=torch.float32, device=device)
            # y_batch is dict of ints
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = 0.0
            for t in target_cols:
                y_t = torch.tensor([yb[t] for yb in y_batch], dtype=torch.long, device=device)
                logits = outputs[t]
                loss += criterions[t](logits, y_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

    # Save model state dict and head class arrays
    torch.save(model.state_dict(), MODELS_DIR / "outfit_net.pt")
    # Save head sizes/classes info
    meta = {
        "head_classes": {
            t: list(encoders['le_'+t].classes_) for t in target_cols
        },
        "onehot_feature_names": encoders['onehot'].get_feature_names_out(feature_cols).tolist()
    }
    joblib.dump(meta, MODELS_DIR / "meta.joblib")
    print("Training finished. Models saved to models/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256, dest="hidden_dim")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
