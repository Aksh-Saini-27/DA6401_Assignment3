import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler

class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2)) 
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        mask = torch.nonzero(target == self.pad_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return self.criterion(F.log_softmax(logits, dim=-1), true_dist)

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0.0
    total_tokens = 0
    
    for batch in tqdm(data_iter, desc=f"Epoch {epoch_num} {'Train' if is_train else 'Eval'}"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_expected = tgt[:, 1:]
        
        src_mask = make_src_mask(src, pad_idx=1).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=1).to(device)
        
        if is_train:
            optimizer.zero_grad()
            
        logits = model(src, tgt_input, src_mask, tgt_mask)
        
        loss = loss_fn(logits.contiguous().view(-1, logits.size(-1)), tgt_expected.contiguous().view(-1))
        
        if is_train:
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                
        ntokens = (tgt_expected != 1).data.sum().item()
        total_loss += loss.item()
        total_tokens += ntokens
        
    return total_loss / total_tokens

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = F.softmax(out[:, -1], dim=-1)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)
        if next_word == end_symbol:
            break
            
    return ys

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    # Requires standard bleu package or torchmetrics
    from bleu import list_bleu
    
    model.eval()
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Calculating BLEU"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            for i in range(src.size(0)):
                src_row = src[i:i+1]
                tgt_row = tgt[i]
                
                src_mask = make_src_mask(src_row, pad_idx=1).to(device)
                
                # decode indices
                pred_indices = greedy_decode(model, src_row, src_mask, max_len, start_symbol=2, end_symbol=3, device=device)
                pred_indices = pred_indices.squeeze().tolist()
                
                pred_tokens = [tgt_vocab.tgt_itos[idx] for idx in pred_indices if idx not in [0, 1, 2, 3]]
                ref_tokens = [tgt_vocab.tgt_itos[idx.item()] for idx in tgt_row if idx.item() not in [0, 1, 2, 3]]
                
                hypotheses.append(" ".join(pred_tokens))
                references.append(" ".join(ref_tokens))
                
    # Format and list_bleu execution
    return list_bleu([references], hypotheses)

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    model_config: dict = None
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': model_config
    }, path)

def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint['epoch']

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(item['src'])
        tgt_batch.append(item['tgt'])
        
    src_padded = pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    
    return {'src': src_padded, 'tgt': tgt_padded}

# def run_training_experiment() -> None:
#     wandb.init(project="da6401-a3", config={
#         "d_model": 512, "num_heads": 8, "d_ff": 2048, "N": 6, "dropout": 0.1, "batch_size": 128, "epochs": 10
#     })
#     config = wandb.config
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     dataset = Multi30kDataset()
#     dataset.build_vocab()
    
#     train_data = dataset.process_data()
#     val_data = Multi30kDataset(split='validation')
#     val_data.src_vocab, val_data.tgt_vocab = dataset.src_vocab, dataset.tgt_vocab
#     val_processed = val_data.process_data()
    
#     train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_processed, batch_size=config.batch_size, collate_fn=collate_fn)
    
#     model_config = {
#         'src_vocab_size': len(dataset.src_vocab),
#         'tgt_vocab_size': len(dataset.tgt_vocab),
#         'd_model': config.d_model,
#         'N': config.N,
#         'num_heads': config.num_heads,
#         'd_ff': config.d_ff,
#         'dropout': config.dropout
#     }
    
#     model = Transformer(**model_config).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
#     scheduler = NoamScheduler(optimizer, config.d_model, warmup_steps=4000)
#     loss_fn = LabelSmoothingLoss(model_config['tgt_vocab_size'], pad_idx=1, smoothing=0.1).to(device)
    
#     for epoch in range(config.epochs):
#         train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
#         val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        
#         wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
#         save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_ep{epoch}.pt", model_config=model_config)

def run_training_experiment() -> None:
    wandb.init(project="da6401-a3", config={
        "d_model": 512, "num_heads": 8, "d_ff": 2048, "N": 6, "dropout": 0.1, "batch_size": 128, "epochs": 10
    })
    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = Multi30kDataset()
    dataset.build_vocab()
    
    train_data = dataset.process_data()
    val_data = Multi30kDataset(split='validation')
    val_data.src_vocab, val_data.tgt_vocab = dataset.src_vocab, dataset.tgt_vocab
    val_processed = val_data.process_data()
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_processed, batch_size=config.batch_size, collate_fn=collate_fn)
    
    model_config = {
        'src_vocab_size': len(dataset.src_vocab),
        'tgt_vocab_size': len(dataset.tgt_vocab),
        'd_model': config.d_model,
        'N': config.N,
        'num_heads': config.num_heads,
        'd_ff': config.d_ff,
        'dropout': config.dropout
    }
    
    model = Transformer(**model_config).to(device)
    # NOTE: lr is set to 1.0 here so the Noam Scheduler can scale it properly!
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, config.d_model, warmup_steps=4000)
    loss_fn = LabelSmoothingLoss(model_config['tgt_vocab_size'], pad_idx=1, smoothing=0.1).to(device)
    
    # --- NEW ADDITION: Initialize best validation loss tracker ---
    best_val_loss = float('inf') 
    
    for epoch in range(config.epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        
        # --- NEW ADDITION: Print the losses for this epoch ---
        print(f"\nEpoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # --- NEW ADDITION: Check if current model is the best, then save/overwrite ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"--> Validation loss improved! Overwriting best checkpoint...")
            save_checkpoint(
                model, 
                optimizer, 
                scheduler, 
                epoch, 
                path="transformer_best.pt",  # Fixed filename means it will overwrite the old one
                model_config=model_config
            )
        else:
            print(f"--> No improvement. Best was {best_val_loss:.4f}.")

if __name__ == "__main__":
    run_training_experiment()