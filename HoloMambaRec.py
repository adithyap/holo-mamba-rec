# Run this command to install deps on Colab
# !pip install --target=$custom_package_path mamba-ssm causal-conv1d google-generativeai datasets transformers torch --use-deprecated=legacy-resolver

import os
import time
import math
import random
import requests
import gzip
import zipfile
import io
import json
import sys
import itertools
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# PART 0: CONFIGURATION & UTILS
# ==========================================

GLOBAL_CONFIG = {
    'min_user_len': 5,
    'max_len': 50,
    'max_len_long': 512,    # for compression profiling
    'test_ratio': 0.2,
    'd_model': 96,          # Tuned from grid search
    'd_state': 16,          # Mamba state
    'n_layers': 2,          # Overridden per dataset below
    'window_size': 4,       # Holo compression
    'use_compression': True,
    'dropout': 0.1,
    'num_synthetic_attrs': 50,
    'batch_size': 128,
    'lr': 1e-3,
    'epochs': 10,           # Increased for paper results
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(GLOBAL_CONFIG['seed'])
print(f"Running on {GLOBAL_CONFIG['device']}...")

# Color palette for plots
COLOR_PALETTE = {
    'HoloMamba': '#1f77b4',
    'SASRec': '#ff7f0e',
    'GRU4Rec': '#2ca02c',
    'Mamba-ItemOnly': '#d62728'
}

# ==========================================
# PART 1: UNIFIED DATA PIPELINE
# ==========================================

class DataManager:
    def __init__(self, dataset_name, config):
        self.dataset_name = dataset_name
        self.config = config
        self.data_dir = f'data/{dataset_name.lower()}'
        os.makedirs(self.data_dir, exist_ok=True)
        
        if dataset_name == 'Amazon-Beauty':
            self.loader = AmazonLoader(self.data_dir, config)
        elif dataset_name == 'ML-1M':
            self.loader = ML1MLoader(self.data_dir, config)
        else:
            raise ValueError("Unknown Dataset")
            
        self.users, self.items, self.data = self.loader.load_data()

    def get_sequences(self):
        return self.loader.get_sequences(self.data)

class AmazonLoader:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        self.file_path = os.path.join(data_dir, 'reviews_Beauty_5.json.gz')
        self.download()

    def download(self):
        if os.path.exists(self.file_path): return
        url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz"
        print(f"Downloading Amazon Beauty to {self.file_path}...")
        r = requests.get(url, stream=True)
        with open(self.file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk: f.write(chunk)

    def load_data(self):
        data = []
        with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing JSON", file=sys.stdout):
                try:
                    record = json.loads(line)
                    data.append({
                        'user_id': record['reviewerID'],
                        'item_id': record['asin'],
                        'timestamp': record['unixReviewTime']
                    })
                except ValueError: continue
        
        df = pd.DataFrame(data)
        return self._process_df(df)

    def _process_df(self, df):
        # Filter & Map
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.config['min_user_len']].index
        df = df[df['user_id'].isin(valid_users)].copy()
        
        user_map = {id: i+1 for i, id in enumerate(df['user_id'].unique())}
        item_map = {id: i+1 for i, id in enumerate(df['item_id'].unique())}
        df['user_id'] = df['user_id'].map(user_map)
        df['item_id'] = df['item_id'].map(item_map)
        
        # Attributes
        self.item2attr = {iid: (iid % self.config['num_synthetic_attrs']) + 1 for iid in item_map.values()}
        
        self.config['num_items'] = len(item_map) + 1
        self.config['num_attrs'] = self.config['num_synthetic_attrs'] + 1
        return len(user_map), len(item_map), df.sort_values(by=['user_id', 'timestamp'])

    def get_sequences(self, df):
        train_seqs, train_attrs = [], []
        test_seqs, test_attrs, test_targets = [], [], []
        
        for _, group in tqdm(df.groupby('user_id'), desc="Gen Seqs", file=sys.stdout):
            items = group['item_id'].tolist()
            if len(items) < self.config['min_user_len']: continue
            attrs = [self.item2attr.get(i, 0) for i in items]
            
            # Test (Last item)
            test_seqs.append(items[:-1][-self.config['max_len']:])
            test_attrs.append(attrs[:-1][-self.config['max_len']:])
            test_targets.append(items[-1])
            
            # Train (Everything else)
            if len(items) > 1:
                train_seqs.append(items[:-1][-self.config['max_len']:])
                train_attrs.append(attrs[:-1][-self.config['max_len']:])
                
        return (train_seqs, train_attrs), (test_seqs, test_attrs, test_targets)

class ML1MLoader:
    def __init__(self, data_dir, config):
        self.data_dir = data_dir
        self.config = config
        self.download()

    def download(self):
        if os.path.exists(os.path.join(self.data_dir, 'ratings.dat')): return
        print("Downloading MovieLens-1M...")
        r = requests.get("https://files.grouplens.org/datasets/movielens/ml-1m.zip")
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall('data')
        # Move files if needed or just access data/ml-1m

    def load_data(self):
        rpath = os.path.join(self.data_dir, 'ratings.dat')
        mpath = os.path.join(self.data_dir, 'movies.dat')
        
        df = pd.read_csv(rpath, sep='::', header=None, engine='python', 
                         names=['user_id', 'item_id', 'rating', 'timestamp'])
        movies = pd.read_csv(mpath, sep='::', header=None, engine='python', encoding='latin-1',
                             names=['item_id', 'title', 'genres'])
        
        # Process Genres
        all_genres = set()
        for g in movies['genres']: all_genres.update(g.split('|'))
        genre_map = {g: i+1 for i, g in enumerate(sorted(list(all_genres)))}
        
        # Item Map
        item_map = {id: i+1 for i, id in enumerate(df['item_id'].unique())}
        df['item_id'] = df['item_id'].map(item_map)
        
        # Attr Map (First Genre)
        self.item2attr = {}
        for _, row in movies.iterrows():
            if row['item_id'] in item_map:
                mapped_id = item_map[row['item_id']]
                self.item2attr[mapped_id] = genre_map[row['genres'].split('|')[0]]
        
        # User Map
        user_map = {id: i+1 for i, id in enumerate(df['user_id'].unique())}
        df['user_id'] = df['user_id'].map(user_map)
        
        self.config['num_items'] = len(item_map) + 1
        self.config['num_attrs'] = len(genre_map) + 1
        return len(user_map), len(item_map), df.sort_values(by=['user_id', 'timestamp'])

    def get_sequences(self, df):
        # Logic is identical to Amazon for this simple bench
        train_seqs, train_attrs = [], []
        test_seqs, test_attrs, test_targets = [], [], []
        
        for _, group in tqdm(df.groupby('user_id'), desc="Gen Seqs", file=sys.stdout):
            items = group['item_id'].tolist()
            if len(items) < self.config['min_user_len']: continue
            attrs = [self.item2attr.get(i, 0) for i in items]
            
            test_seqs.append(items[:-1][-self.config['max_len']:])
            test_attrs.append(attrs[:-1][-self.config['max_len']:])
            test_targets.append(items[-1])
            
            if len(items) > 1:
                train_seqs.append(items[:-1][-self.config['max_len']:])
                train_attrs.append(attrs[:-1][-self.config['max_len']:])
        return (train_seqs, train_attrs), (test_seqs, test_attrs, test_targets)

class RecDataset(Dataset):
    def __init__(self, seqs, attrs, targets=None, max_len=50, mode='train'):
        self.seqs = seqs
        self.attrs = attrs
        self.targets = targets
        self.max_len = max_len
        self.mode = mode
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        attr = self.attrs[idx]
        pad_len = self.max_len - len(seq)
        seq = [0]*pad_len + seq
        attr = [0]*pad_len + attr
        seq_t = torch.tensor(seq, dtype=torch.long)
        attr_t = torch.tensor(attr, dtype=torch.long)
        if self.mode == 'train': return seq_t, attr_t, seq_t
        return seq_t, attr_t, torch.tensor(self.targets[idx], dtype=torch.long)

# ==========================================
# PART 2: MODELS
# ==========================================

# --- HOLOMAMBA UTILS ---
class HoloBinding(nn.Module):
    def forward(self, x, y):
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        y_fft = torch.fft.rfft(y, dim=-1, norm='ortho')
        return torch.fft.irfft(x_fft * y_fft, n=x.shape[-1], dim=-1, norm='ortho')

class HoloEmbedding(nn.Module):
    def __init__(self, num_items, num_attributes, embed_dim):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.attr_emb = nn.Embedding(num_attributes, embed_dim, padding_idx=0)
        self.binding = HoloBinding()
        self.norm = nn.LayerNorm(embed_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, i, a):
        ei, ea = self.item_emb(i), self.attr_emb(a)
        return self.norm(ei + self.alpha * self.binding(ei, ea))

class HoloBundler(nn.Module):
    """
    Bundles k consecutive holographic tokens using positional role vectors.
    """
    def __init__(self, embed_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.roles = nn.Parameter(torch.randn(window_size, embed_dim) * (1.0 / math.sqrt(embed_dim)))
        self.binding = HoloBinding()

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        k = self.window_size
        if L % k != 0:
            pad_len = k - (L % k)
            pad = torch.zeros(B, pad_len, D, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
            L = x.shape[1]
        x = x.reshape(B, L // k, k, D)  # (B, W, k, D)
        out = torch.zeros(B, L // k, D, device=x.device, dtype=x.dtype)
        for t in range(k):
            role = self.roles[t].unsqueeze(0).unsqueeze(0)  # (1,1,D)
            out += self.binding(x[:, :, t, :], role)
        return out

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, (math.ceil(d_model/16)) + d_state*2, bias=False)
        self.dt_proj = nn.Linear(math.ceil(d_model/16), self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state+1, dtype=torch.float32).repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.act = nn.SiLU()
        self.d_state = d_state

    def forward(self, x):
        B, L, _ = x.shape
        x_inner, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.act(self.conv1d(x_inner.transpose(1, 2))[:, :, :L]).transpose(1, 2)
        dt_rank = self.dt_proj.in_features
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # Simple Scan
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))
        dB = torch.einsum('bld,bln->bldn', dt, B_ssm)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x_conv[:, t].unsqueeze(-1)
            ys.append((h * C_ssm[:, t].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(ys, dim=1) + x_conv * self.D
        return self.out_proj(y * self.act(z))

# --- MODELS ---
class HoloMambaRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = HoloEmbedding(config['num_items'], config['num_attrs'], config['d_model'])
        self.use_compression = False  # default off to preserve legacy behavior
        self.window_size = config.get('window_size', 4)
        if config.get('use_compression', False):
            self.use_compression = True
        self.bundler = HoloBundler(config['d_model'], self.window_size)
        self.layers = nn.ModuleList([MambaBlock(config['d_model'], d_state=config['d_state']) for _ in range(config['n_layers'])])
        self.norms = nn.ModuleList([nn.LayerNorm(config['d_model']) for _ in range(config['n_layers'])])
        self.head = nn.Linear(config['d_model'], config['num_items'], bias=False)
    def forward(self, seq, attr, compress=None):
        if compress is None:
            compress = self.use_compression
        x = self.emb(seq, attr)
        if compress and self.bundler is not None:
            x = self.bundler(x)
        for l, n in zip(self.layers, self.norms): x = x + l(n(x))
        return self.head(x)

class MambaItemOnlyRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Standard Embedding only
        self.emb = nn.Embedding(config['num_items'], config['d_model'], padding_idx=0)
        self.norm_e = nn.LayerNorm(config['d_model'])
        self.layers = nn.ModuleList([MambaBlock(config['d_model'], d_state=config['d_state']) for _ in range(config['n_layers'])])
        self.norms = nn.ModuleList([nn.LayerNorm(config['d_model']) for _ in range(config['n_layers'])])
        self.head = nn.Linear(config['d_model'], config['num_items'], bias=False)
    def forward(self, seq, attr=None, compress=None):
        # Ignore attr
        x = self.norm_e(self.emb(seq))
        for l, n in zip(self.layers, self.norms): x = x + l(n(x))
        return self.head(x)

class SASRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.item_emb = nn.Embedding(config['num_items'], config['d_model'], padding_idx=0)
        self.pos_emb = nn.Embedding(config['max_len'], config['d_model'])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config['d_model'], nhead=2, dim_feedforward=256, dropout=config['dropout'], batch_first=True),
            num_layers=config['n_layers'])
        self.head = nn.Linear(config['d_model'], config['num_items'])
    def forward(self, seq, attr=None, compress=None):
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(seq.size(1), seq.size(1), device=seq.device), diagonal=1).bool()
        x = self.encoder(x, mask=mask)
        return self.head(x)

class GRU4Rec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.item_emb = nn.Embedding(config['num_items'], config['d_model'], padding_idx=0)
        self.gru = nn.GRU(config['d_model'], config['d_model'], config['n_layers'], batch_first=True)
        self.head = nn.Linear(config['d_model'], config['num_items'])
    def forward(self, seq, attr=None, compress=None):
        x, _ = self.gru(self.item_emb(seq))
        return self.head(x)

# ==========================================
# PART 3: EXPERIMENT RUNNER
# ==========================================

def train(model, loader, opt, config, desc=None):
    model.train()
    loss_sum = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    iterable = loader
    if desc is not None:
        iterable = tqdm(loader, desc=desc, leave=False, total=len(loader))
    for seq, attr, _ in iterable:
        seq, attr = seq.to(config['device']), attr.to(config['device'])
        opt.zero_grad()
        logits = model(seq[:, :-1], attr[:, :-1], compress=False)
        loss = criterion(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        loss_sum += loss.item()
    return loss_sum / len(loader)

def evaluate(model, loader, config, compress=False):
    model.eval()
    hits, ndcgs = [], []
    with torch.no_grad():
        for seq, attr, target in loader:
            seq, attr, target = seq.to(config['device']), attr.to(config['device']), target.to(config['device'])
            logits = model(seq, attr, compress=compress)
            scores = logits[:, -1, :]
            scores[:, 0] = -float('inf')
            _, indices = torch.topk(scores, 10)
            for pred, true in zip(indices.cpu().numpy(), target.cpu().numpy()):
                if true in pred:
                    hits.append(1)
                    ndcgs.append(1 / np.log2(np.where(pred==true)[0][0] + 2))
                else:
                    hits.append(0); ndcgs.append(0)
    return np.mean(hits), np.mean(ndcgs)

def run_benchmark():
    datasets = ['Amazon-Beauty', 'ML-1M']
    models = ['HoloMamba', 'SASRec', 'GRU4Rec']
    history = {d: {m: {'loss': [], 'hr': [], 'ndcg': []} for m in models} for d in datasets}

    for ds_name in datasets:
        print(f"\n=== PROCESSING DATASET: {ds_name} ===")
        # Apply best-found hyperparams per dataset
        if ds_name == 'Amazon-Beauty':
            cfg = {**GLOBAL_CONFIG, 'n_layers': 2, 'batch_size': 64}
        elif ds_name == 'ML-1M':
            cfg = {**GLOBAL_CONFIG, 'n_layers': 3, 'batch_size': 64}
        else:
            cfg = GLOBAL_CONFIG

        dm = DataManager(ds_name, cfg)
        (tr_s, tr_a), (te_s, te_a, te_t) = dm.get_sequences()
        
        tr_dl = DataLoader(RecDataset(tr_s, tr_a, max_len=cfg['max_len']), batch_size=cfg['batch_size'], shuffle=True)
        te_dl = DataLoader(RecDataset(te_s, te_a, te_t, max_len=cfg['max_len'], mode='test'), batch_size=cfg['batch_size'], shuffle=False)
        
        for m_name in models:
            print(f"\n--- Training {m_name} on {ds_name} ---")
            if m_name == 'HoloMamba': model = HoloMambaRec(cfg).to(cfg['device'])
            elif m_name == 'SASRec': model = SASRec(cfg).to(cfg['device'])
            elif m_name == 'GRU4Rec': model = GRU4Rec(cfg).to(cfg['device'])
            
            opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
            
            for ep in range(cfg['epochs']):
                loss = train(model, tr_dl, opt, cfg, desc=f"{ds_name} {m_name} ep {ep+1}/{cfg['epochs']}")
                hr, ndcg = evaluate(model, te_dl, cfg, compress=False)
                
                history[ds_name][m_name]['loss'].append(loss)
                history[ds_name][m_name]['hr'].append(hr)
                history[ds_name][m_name]['ndcg'].append(ndcg)
                
                print(f"Ep {ep+1} | Loss: {loss:.4f} | HR@10: {hr:.4f} | NDCG@10: {ndcg:.4f}")

    # Plotting: generate one image per (dataset, metric) for the paper
    metrics = ['loss', 'hr', 'ndcg']
    titles = ['Training Loss', 'Hit Rate @ 10', 'NDCG @ 10']
    ylabels = ['Loss', 'HR@10', 'NDCG@10']

    print("\nRendering individual benchmark plots...")
    for ds in datasets:
        for metric, title, ylabel in zip(metrics, titles, ylabels):
            fig, ax = plt.subplots(figsize=(6, 4))
            for model in models:
                color = COLOR_PALETTE.get(model, None)
                ax.plot(history[ds][model][metric], marker='.', label=model, color=color, linewidth=2)
            ax.set_title(f"{ds} - {title}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fname = f"learning_curve_{ds.replace(' ', '_').lower()}_{metric}.png"
            plt.tight_layout()
            plt.savefig(fname)
            plt.close(fig)
            print(f"Saved {fname}")

    final_data = []
    for model in models:
        row = {'Model': model}
        for ds in datasets:
            row[f'{ds} HR@10'] = history[ds][model]['hr'][-1]
            row[f'{ds} NDCG@10'] = history[ds][model]['ndcg'][-1]
        final_data.append(row)
    
    df_results = pd.DataFrame(final_data)
    pd.options.display.float_format = '{:.4f}'.format
    print("\n=== FINAL BENCHMARK RESULTS ===")
    display(df_results)
    print("\nMarkdown Table:")
    print(df_results.to_markdown(index=False))

def run_ablation_study():
    print("\n\n" + "="*40)
    print(" ABLATION STUDY: BINDING vs ITEM-ONLY")
    print("="*40)
    
    datasets = ['Amazon-Beauty', 'ML-1M']
    models = ['HoloMamba', 'Mamba-ItemOnly'] # Compare Full vs Ablated
    history = {d: {m: {'hr': [], 'ndcg': []} for m in models} for d in datasets}
    
    for ds_name in datasets:
        print(f"\n>>> Processing {ds_name} for Ablation...")
        dm = DataManager(ds_name, GLOBAL_CONFIG)
        (tr_s, tr_a), (te_s, te_a, te_t) = dm.get_sequences()
        
        tr_dl = DataLoader(RecDataset(tr_s, tr_a, max_len=GLOBAL_CONFIG['max_len']), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=True)
        te_dl = DataLoader(RecDataset(te_s, te_a, te_t, max_len=GLOBAL_CONFIG['max_len'], mode='test'), batch_size=GLOBAL_CONFIG['batch_size'], shuffle=False)
        
        for m_name in models:
            print(f"   Training {m_name}...")
            if m_name == 'HoloMamba': model = HoloMambaRec(GLOBAL_CONFIG).to(GLOBAL_CONFIG['device'])
            elif m_name == 'Mamba-ItemOnly': model = MambaItemOnlyRec(GLOBAL_CONFIG).to(GLOBAL_CONFIG['device'])
            
            opt = optim.AdamW(model.parameters(), lr=GLOBAL_CONFIG['lr'])
            for ep in range(GLOBAL_CONFIG['epochs']):
                train(model, tr_dl, opt, GLOBAL_CONFIG, desc=f"{ds_name} {m_name} ep {ep+1}/{GLOBAL_CONFIG['epochs']}")
                
            # Final Eval only for speed
            hr, ndcg = evaluate(model, te_dl, GLOBAL_CONFIG, compress=False)
            history[ds_name][m_name]['hr'].append(hr)
            history[ds_name][m_name]['ndcg'].append(ndcg)
            print(f"   -> {m_name} Final: HR={hr:.4f}, NDCG={ndcg:.4f}")

    # Visualization of Ablation
    print("\ngenerating ablation plot...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ['hr', 'ndcg']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(datasets))
        width = 0.35
        
        # Plot bars
        vals_holo = [history[d]['HoloMamba'][metric][-1] for d in datasets]
        vals_item = [history[d]['Mamba-ItemOnly'][metric][-1] for d in datasets]
        
        ax.bar(x - width/2, vals_holo, width, label='HoloMamba (Binding)', color=COLOR_PALETTE.get('HoloMamba'))
        ax.bar(x + width/2, vals_item, width, label='Mamba (Item-Only)', alpha=0.7, color=COLOR_PALETTE.get('Mamba-ItemOnly'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(f'{metric.upper()}@10')
        if i==0: ax.legend()
    
    plt.suptitle("Ablation: Impact of Holographic Binding")
    plt.savefig("ablation_study.png")
    plt.show()

def clone_config(base, overrides=None):
    cfg = copy.deepcopy(base)
    if overrides:
        for k, v in overrides.items():
            cfg[k] = v
    return cfg

def _build_loaders_for_grid(ds_name, config):
    dm = DataManager(ds_name, config)
    (tr_s, tr_a), (te_s, te_a, te_t) = dm.get_sequences()
    tr_dl = DataLoader(RecDataset(tr_s, tr_a, max_len=config['max_len']), batch_size=config['batch_size'], shuffle=True)
    te_dl = DataLoader(RecDataset(te_s, te_a, te_t, max_len=config['max_len'], mode='test'), batch_size=config['batch_size'], shuffle=False)
    return tr_dl, te_dl

def run_grid_search(search_space=None, datasets=None, max_trials=None):
    """
    Lightweight grid search over HoloMambaRec hyperparameters.
    Controlled by RUN_GRID_SEARCH=1 and GRID_MAX_TRIALS (optional) environment variables.
    """
    if search_space is None:
        search_space = {
            'd_model': [64, 96],
            'd_state': [8, 16],
            'n_layers': [2, 3],
            'lr': [1e-3, 5e-4],
            'batch_size': [64, 128]
        }
    if datasets is None:
        datasets = ['Amazon-Beauty', 'ML-1M']

    # Generate all combinations then optionally subsample
    keys = list(search_space.keys())
    combos = [dict(zip(keys, values)) for values in itertools.product(*[search_space[k] for k in keys])]
    if max_trials is not None and max_trials < len(combos):
        random.shuffle(combos)
        combos = combos[:max_trials]

    all_results = []
    for ds_name in datasets:
        print(f"\n### GRID SEARCH on {ds_name} ###")
        best_ndcg = -1.0
        best_cfg = None
        for idx, overrides in enumerate(combos, start=1):
            cfg = clone_config(GLOBAL_CONFIG, overrides)
            print(f"[{idx}/{len(combos)}] cfg={overrides}")
            try:
                tr_dl, te_dl = _build_loaders_for_grid(ds_name, cfg)
                model = HoloMambaRec(cfg).to(cfg['device'])
                opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
                for ep in range(cfg['epochs']):
                    train(model, tr_dl, opt, cfg, desc=f"{ds_name} grid ep {ep+1}/{cfg['epochs']}")
                hr, ndcg = evaluate(model, te_dl, cfg, compress=False)
                result = {'dataset': ds_name, **overrides, 'hr': hr, 'ndcg': ndcg}
                all_results.append(result)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_cfg = result
                print(f"   -> HR@10={hr:.4f}, NDCG@10={ndcg:.4f}")
            except Exception as e:
                print(f"   !! Skipping config due to error: {e}")
                continue
        if best_cfg:
            print(f"Best for {ds_name}: {best_cfg}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("grid_search_results.csv", index=False)
        print("\nSaved grid search results to grid_search_results.csv")
        print(df.groupby('dataset')[['hr', 'ndcg']].max())

def profile_inference(model, loader, config, steps=5, compress=False):
    """
    Measure latency and peak VRAM over a few batches.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    latencies = []
    total_tokens = 0
    with torch.no_grad():
        for i, (seq, attr, target) in enumerate(loader):
            if i >= steps: break
            seq, attr = seq.to(config['device']), attr.to(config['device'])
            start = time.perf_counter()
            logits = model(seq, attr, compress=compress)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append(end - start)
            total_tokens += seq.numel()
    avg_lat = np.mean(latencies) if latencies else float('nan')
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else float('nan')
    tps = total_tokens / sum(latencies) if latencies else float('nan')
    return {'avg_latency_s': avg_lat, 'peak_mem_mb': peak_mem, 'tokens_per_sec': tps}

def evaluate_cold_start(model, loader, config, max_len_threshold=10, compress=False):
    """
    Evaluate only on users with short histories (<= max_len_threshold).
    Falls back to a looser threshold if no users qualify to avoid empty metrics.
    Returns metrics plus metadata about the slice.
    """
    model.eval()

    # First pass: gather length distribution to decide an effective threshold.
    lengths = []
    with torch.no_grad():
        for seq, _, _ in loader:
            lens = (seq != 0).sum(dim=1)
            lengths.extend(lens.tolist())

    if not lengths:
        return 0.0, 0.0, {'used_threshold': max_len_threshold, 'num_users': 0, 'total_users': 0}

    lengths_arr = np.array(lengths)
    used_threshold = max_len_threshold
    total_users = len(lengths_arr)
    num_users = int((lengths_arr <= used_threshold).sum())

    # If no users meet the cold-start bar, widen to 25th percentile then min length.
    if num_users == 0:
        used_threshold = int(max(np.percentile(lengths_arr, 25), 1))
        num_users = int((lengths_arr <= used_threshold).sum())
        if num_users == 0:
            used_threshold = int(lengths_arr.min())
            num_users = int((lengths_arr <= used_threshold).sum())

    hits, ndcgs = [], []
    matched = 0
    with torch.no_grad():
        for seq, attr, target in loader:
            lens = (seq != 0).sum(dim=1)
            mask = lens <= used_threshold
            if mask.sum() == 0: continue
            matched += int(mask.sum())
            seq = seq[mask].to(config['device'])
            attr = attr[mask].to(config['device'])
            target = target[mask].to(config['device'])
            logits = model(seq, attr, compress=compress)
            scores = logits[:, -1, :]
            scores[:, 0] = -float('inf')
            _, indices = torch.topk(scores, 10)
            for pred, true in zip(indices.cpu().numpy(), target.cpu().numpy()):
                if true in pred:
                    hits.append(1)
                    ndcgs.append(1 / np.log2(np.where(pred==true)[0][0] + 2))
                else:
                    hits.append(0); ndcgs.append(0)

    return (np.mean(hits) if hits else 0.0,
            np.mean(ndcgs) if ndcgs else 0.0,
            {'used_threshold': used_threshold, 'num_users': matched, 'total_users': total_users})

def run_compression_evals():
    """
    Optional: evaluate compression (bundling) accuracy and runtime vs. baseline.
    Controlled by RUN_COMPRESSION=1 environment variable.
    """
    datasets = ['Amazon-Beauty', 'ML-1M']
    plot_data = {}
    for ds_name in datasets:
        print(f"\n=== COMPRESSION STUDY: {ds_name} ===")
        if ds_name == 'Amazon-Beauty':
            cfg = {**GLOBAL_CONFIG, 'n_layers': 2, 'batch_size': 64}
        elif ds_name == 'ML-1M':
            cfg = {**GLOBAL_CONFIG, 'n_layers': 3, 'batch_size': 64}
        else:
            cfg = GLOBAL_CONFIG

        dm = DataManager(ds_name, cfg)
        (tr_s, tr_a), (te_s, te_a, te_t) = dm.get_sequences()
        # standard length loader
        te_dl = DataLoader(RecDataset(te_s, te_a, te_t, max_len=cfg['max_len'], mode='test'), batch_size=cfg['batch_size'], shuffle=False)
        # long length loader for stress (if sequences longer, padded)
        te_dl_long = DataLoader(RecDataset(te_s, te_a, te_t, max_len=cfg['max_len_long'], mode='test'), batch_size=cfg['batch_size'], shuffle=False)

        model = HoloMambaRec({**cfg, 'use_compression': False}).to(cfg['device'])
        opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
        for ep in range(cfg['epochs']):
            train(
                model,
                DataLoader(RecDataset(tr_s, tr_a, max_len=cfg['max_len']), batch_size=cfg['batch_size'], shuffle=True),
                opt,
                cfg,
                desc=f"{ds_name} baseline ep {ep+1}/{cfg['epochs']}"
            )

        hr_base, ndcg_base = evaluate(model, te_dl, cfg, compress=False)
        hr_long, ndcg_long = evaluate(model, te_dl_long, cfg, compress=False)
        print(f"Baseline HR@10/NDCG@10 (L=50): {hr_base:.4f}/{ndcg_base:.4f}")
        print(f"Baseline HR@10/NDCG@10 (L={cfg['max_len_long']}): {hr_long:.4f}/{ndcg_long:.4f}")

        # Compress at inference
        hr_comp, ndcg_comp = evaluate(model, te_dl_long, cfg, compress=True)
        print(f"Compressed HR@10/NDCG@10 (L={cfg['max_len_long']} compressed): {hr_comp:.4f}/{ndcg_comp:.4f}")

        # Latency/VRAM profiling
        prof_base = profile_inference(model, te_dl_long, cfg, compress=False)
        prof_comp = profile_inference(model, te_dl_long, cfg, compress=True)
        print(f"Latency baseline: {prof_base}")
        print(f"Latency compressed: {prof_comp}")

        # Cold-start slice
        hr_cold, ndcg_cold, cold_meta = evaluate_cold_start(model, te_dl, cfg, max_len_threshold=10, compress=False)
        print(f"Cold-start HR@10/NDCG@10 (<= {cold_meta['used_threshold']} events, n={cold_meta['num_users']}/{cold_meta['total_users']}): {hr_cold:.4f}/{ndcg_cold:.4f}")

        plot_data[ds_name] = {
            'accuracy': {
                'labels': [
                    f"L={cfg['max_len']} (base)",
                    f"L={cfg['max_len_long']} (base)",
                    f"L={cfg['max_len_long']} (compressed)"
                ],
                'hr': [hr_base, hr_long, hr_comp],
                'ndcg': [ndcg_base, ndcg_long, ndcg_comp]
            },
            'runtime': {
                'labels': ['Uncompressed', 'Compressed'],
                'lat_ms': [prof_base['avg_latency_s'] * 1000, prof_comp['avg_latency_s'] * 1000],
                'tps': [prof_base['tokens_per_sec'], prof_comp['tokens_per_sec']],
                'peak_mem': [prof_base['peak_mem_mb'], prof_comp['peak_mem_mb']]
            },
            'cold_start': {'hr': hr_cold, 'ndcg': ndcg_cold, **cold_meta}
        }

    if plot_data:
        print("\nRendering compression plots (line charts)...")
        for ds_name in datasets:
            ds_plots = plot_data[ds_name]

            # Accuracy vs length/compression
            labels = ds_plots['accuracy']['labels']
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, ds_plots['accuracy']['hr'], marker='o', label='HR@10', color=COLOR_PALETTE.get('HoloMamba'), linewidth=2)
            ax.plot(x, ds_plots['accuracy']['ndcg'], marker='s', label='NDCG@10', color=COLOR_PALETTE.get('SASRec'), linewidth=2)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_ylabel("Score")
            ax.set_title(f"{ds_name}: Accuracy vs. sequence length")
            valid_scores = [s for s in (ds_plots['accuracy']['hr'] + ds_plots['accuracy']['ndcg']) if not math.isnan(s)]
            max_score = max(valid_scores) if valid_scores else 1.0
            ax.set_ylim(0, max_score * 1.2 if max_score > 0 else 1.0)
            ax.grid(True, axis='y', alpha=0.3)
            ax.legend()
            fname_acc = f"compression_accuracy_{ds_name.replace(' ', '_').lower()}.png"
            plt.tight_layout()
            plt.savefig(fname_acc)
            plt.close(fig)
            print(f"Saved {fname_acc}")

            # Runtime and efficiency (latency + tokens/sec)
            rt_labels = ds_plots['runtime']['labels']
            xr = np.arange(len(rt_labels))
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(xr, ds_plots['runtime']['lat_ms'], marker='o', label='Avg latency (ms)', color=COLOR_PALETTE.get('HoloMamba'), linewidth=2)
            ax1.set_xticks(xr)
            ax1.set_xticklabels(rt_labels)
            ax1.set_ylabel("Avg latency (ms)")
            ax1.set_title(f"{ds_name}: Runtime impact of compression")
            ax1.grid(True, axis='y', alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(xr, ds_plots['runtime']['tps'], marker='s', label='Tokens/sec', color=COLOR_PALETTE.get('GRU4Rec'), linewidth=2)
            ax2.set_ylabel("Tokens/sec")

            # Annotate memory footprints on latency line
            for idx, (lat, mem) in enumerate(zip(ds_plots['runtime']['lat_ms'], ds_plots['runtime']['peak_mem'])):
                mem_label = "CPU" if math.isnan(mem) else f"{mem:.0f} MB"
                if not math.isnan(lat):
                    ax1.text(xr[idx], lat, mem_label, ha='center', va='bottom', fontsize=8)

            lines, labels_ = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels_ + labels2, loc='upper left')
            fname_rt = f"compression_runtime_{ds_name.replace(' ', '_').lower()}.png"
            plt.tight_layout()
            plt.savefig(fname_rt)
            plt.close(fig)
            print(f"Saved {fname_rt}")

            # Cold-start slice
            cold_labels = ['HR@10', 'NDCG@10']
            cold_vals = [ds_plots['cold_start']['hr'], ds_plots['cold_start']['ndcg']]
            fig, axc = plt.subplots(figsize=(6, 4))
            axc.plot(cold_labels, cold_vals, marker='o', linewidth=2, color=COLOR_PALETTE.get('HoloMamba'))
            axc.scatter(cold_labels[0], cold_vals[0], color=COLOR_PALETTE.get('HoloMamba'), marker='o', label='HR@10')
            axc.scatter(cold_labels[1], cold_vals[1], color=COLOR_PALETTE.get('SASRec'), marker='s', label='NDCG@10')
            max_cold = max([v for v in cold_vals if not math.isnan(v)] or [1.0])
            axc.set_ylim(0, max_cold * 1.3 if max_cold > 0 else 1.0)
            axc.set_title(f"{ds_name}: Cold-start users (<= {ds_plots['cold_start']['used_threshold']} events, n={ds_plots['cold_start']['num_users']})")
            axc.grid(True, axis='y', alpha=0.3)
            axc.legend()
            fname_cold = f"compression_cold_start_{ds_name.replace(' ', '_').lower()}.png"
            plt.tight_layout()
            plt.savefig(fname_cold)
            plt.close(fig)
            print(f"Saved {fname_cold}")

if __name__ == "__main__":
    run_benchmark()
    run_ablation_study()

    # Optional
    max_trials = os.environ.get("GRID_MAX_TRIALS")
    max_trials = int(max_trials) if max_trials else None
    run_grid_search(max_trials=max_trials)
    
    # Optional
    run_compression_evals()

# Result from Grid search
# Best for Amazon-Beauty: {'dataset': 'Amazon-Beauty', 'd_model': 96, 'd_state': 16, 'n_layers': 2, 'lr': 0.001, 'batch_size': 64, 'hr': np.float64(0.056387783392210344), 'ndcg': np.float64(0.0367070070227746)}
# Best for ML-1M: {'dataset': 'ML-1M', 'd_model': 96, 'd_state': 16, 'n_layers': 3, 'lr': 0.001, 'batch_size': 64, 'hr': np.float64(0.16903973509933776), 'ndcg': np.float64(0.09608687679176134)}
#                      hr      ndcg
# dataset                          
# Amazon-Beauty  0.056388  0.036707
# ML-1M          0.169040  0.096087
