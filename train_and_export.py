import torch
import torch.nn as nn
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

sys.path.append('/content/gasy-llm/GASAI-PYTHON')
from train.model import SmallMalagasyLLM
from export.export_weights import GMLExporter

print("✅ Train script loaded successfully!")

# Tokenizer malagasy simple
class MalagasyTokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3
        
    def encode(self, text, max_length=512):
        words = text.split()[:max_length-2]
        tokens = [self.bos_token] + [hash(w) % (self.vocab_size-4) + 4 for w in words] + [self.eos_token]
        if len(tokens) < max_length:
            tokens = tokens + [self.pad_token] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
        return torch.tensor(tokens)
    
    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True, max_length=512):
        batch = [self.encode(text, max_length) for text in texts]
        return {'input_ids': torch.stack(batch)}

tokenizer = MalagasyTokenizer()

class MalagasyDataset(Dataset):
    def __init__(self, jsonl_path, max_length=512):
        self.examples = []
        self.max_length = max_length
        
        print(f"📖 Mamaky dataset: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    if text and len(text) > 20:
                        self.examples.append(text)
                except:
                    continue
                if i > 0 and i % 10000 == 0:
                    print(f"   {i} lines traités...")
        print(f"   ✅ {len(self.examples)} ohatra hita")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    texts = list(batch)
    encoded = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    return encoded['input_ids']

def train_one_epoch(model, dataloader, optimizer, epoch, device, accumulation_steps=2, logging_steps=100):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    optimizer.zero_grad()
    
    print("")
    print("="*60)
    print(f"🚀 EPOCH {epoch + 1}/5")
    print("="*60)
    
    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # Shift for next token prediction
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Loss computation
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Logging
        if batch_idx % logging_steps == 0:
            current_loss = loss.item() * accumulation_steps
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
            print(f"   Batch {batch_idx}/{num_batches} | Loss: {current_loss:.4f} | GPU: {gpu_mem:.2f}GB")
    
    avg_loss = total_loss / num_batches
    print("")
    print(f"✅ Epoch {epoch+1} vita! Loss: {avg_loss:.4f}")
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint: {path}")

def train_model(model, dataset, epochs=5, batch_size=4, lr=1e-4, 
                device='cuda', accumulation_steps=2, logging_steps=100, save_steps=1000, start_epoch=0):
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=2,
        pin_memory=True if device=='cuda' else False
    )
    
    # Try to use 8-bit optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
        print("✅ 8-bit optimizer (bitsandbytes) ampiasaina")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        print("⚠️ bitsandbytes tsy hita, AdamW standard no ampiasaina")
    
    print("")
    print(f"📊 Total batches: {len(dataloader)}")
    print(f"📊 Effective batch size: {batch_size * accumulation_steps}")
    
    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, epoch, device, 
            accumulation_steps, logging_steps
        )
        
        # Save checkpoint
        save_checkpoint(
            model, optimizer, epoch, avg_loss, 
            f"checkpoint_epoch_{epoch+1}.pt"
        )
    
    return model

def export_final_model(model, config, output_path, quant_type):
    print("")
    print(f"📦 Export mankany {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    exporter = GMLExporter(model, config)
    exporter.export_gml(output_path, quant_type=quant_type)
    
    if Path(output_path).exists():
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   ✅ Vita! {size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=str, default="Q4_0", choices=["Q4_0", "Q2_K", "Q8_0"])
    parser.add_argument("--output", type=str, default="./gasyAI-500m.gml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--resume", type=str, help="Checkpoint hialana")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--no_checkpointing", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--auto_resume", action="store_true", help="Haka ho azy ny checkpoint farany")
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 GASYML - TRAINING 500M (GITHUB AUTO-SAVE)")
    print("="*60)
    print(f"📁 Data: {args.data}")
    print(f"🔢 Epochs: {args.epochs}")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Accumulation steps: {args.accumulation_steps}")
    print(f"📊 Effective batch: {args.batch_size * args.accumulation_steps}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"💻 Device: {device}")
    
    if device.type == 'cuda':
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 VRAM: {vram:.1f}GB")
    
    # Load dataset
    dataset = MalagasyDataset(args.data)
    print(f"📊 Total ohatra: {len(dataset)}")
    
    # Create model
    print("")
    print("🔧 Mamorona modèle 500M...")
    model = SmallMalagasyLLM(
        vocab_size=16000,
        hidden_size=768,
        num_layers=20,
        num_heads=12,
        num_kv_heads=4,
        use_checkpointing=not args.no_checkpointing
    )
    model = model.to(device)
    
    total_params = model.get_num_params()
    print(f"   ✅ Modèle: {total_params:,} paramètres (~500M)")
    
    # ⚠️ ZAVA-DEHIBE: Auto-resume raha misy checkpoint
    start_epoch = 0
    if args.auto_resume:
        checkpoint_files = sorted([f for f in os.listdir('.') if f.startswith('checkpoint_epoch_')])
        if checkpoint_files:
            args.resume = checkpoint_files[-1]
            print(f"🔄 Auto-resume: {args.resume} no hita")
    
    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"🔄 Namerina tamin'ny epoch {start_epoch}")
    
    # Train
    print("")
    print("🏋️ Manomboka training 5 epochs...")
    model = train_model(
        model, dataset, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        accumulation_steps=args.accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        start_epoch=start_epoch
    )
    
    # Export
    config = {
        'vocab_size': 16000,
        'hidden_size': 768,
        'num_heads': 12,
        'num_kv_heads': 4,
        'num_layers': 20,
        'max_seq_len': 512,
    }
    
    export_final_model(model, config, args.output, args.quant)
    print("")
    print("✅ VITA! 5 epochs vita")
    print(f"📦 Modèle: {args.output}")

if __name__ == '__main__':
    main()
