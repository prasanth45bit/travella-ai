import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
import json

class TripMind(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super(TripMind, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3)  # morning, afternoon, evening
        )   

    def forward(self, x):
        return self.net(x)

class Preprocessor:
    def __init__(self, embed_model='all-MiniLM-L6-v2'):
        self.embed_model = SentenceTransformer(embed_model)
        self.dest_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.ready = False

    def fit(self, df):
        df = df.fillna("")
        self.dest_encoder.fit(df["destination"].astype(str))
        self.scaler.fit(df[["budget", "cost", "rating", "duration_hr"]].fillna(0))
        self.ready = True

    def transform(self, df):
        df = df.fillna("")
        dest_labels = self.dest_encoder.transform(df["destination"].astype(str))
        dest_onehot = np.eye(len(self.dest_encoder.classes_))[dest_labels]
        embeddings = self.embed_model.encode(df["activity_name"].astype(str).tolist(), show_progress_bar=False)
        nums = self.scaler.transform(df[["budget", "cost", "rating", "duration_hr"]].fillna(0))
        coords = df[["latitude", "longitude"]].fillna(0).to_numpy()
        return np.hstack([dest_onehot, embeddings, nums, coords])

    def save(self, path):
        meta = {
            "classes": self.dest_encoder.classes_.tolist(),
            "scaler_params": {
                "min": self.scaler.min_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "data_min": self.scaler.data_min_.tolist(),
                "data_max": self.scaler.data_max_.tolist(),
                "data_range": self.scaler.data_range_.tolist(),
            },
        }
        with open(path, "w") as f:
            json.dump(meta, f)

    def load(self, path):
        with open(path, "r") as f:
            meta = json.load(f)
        self.dest_encoder.classes_ = np.array(meta["classes"])
        s = meta["scaler_params"]
        self.scaler.min_ = np.array(s["min"])
        self.scaler.scale_ = np.array(s["scale"])
        self.scaler.data_min_ = np.array(s["data_min"])
        self.scaler.data_max_ = np.array(s["data_max"])
        self.scaler.data_range_ = np.array(s["data_range"])
        self.ready = True

def prepare_training(df, pre):
    slot_map = {"morning": 0, "afternoon": 1, "evening": 2}
    df["slot"] = df["time_slot"].map(slot_map).fillna(1).astype(int)
    X = pre.transform(df)
    y = df["slot"].values.astype(int)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_model(df, epochs=20, batch=32, lr=1e-3, model_path="./models/tripmind_v3.pt"):
    pre = Preprocessor()
    pre.fit(df)
    X, y = prepare_training(df, pre)
    model = TripMind(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(dataset):.4f}")

    torch.save(model.state_dict(), model_path)
    pre.save(model_path + ".meta.json")
    print(f"✅ Model saved at {model_path}")
    return model, pre

def load_model_safe(model_path, input_dim):
    model = TripMind(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model
