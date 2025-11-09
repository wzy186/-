import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import yaml
import argparse

from model import TransformerLM
from data_loader import create_data_loader


class Trainer:
    """Transformer model trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create data loaders and model
        self.train_loader, self.val_loader, self.vocab_info = create_data_loader(
            batch_size=config['training']['batch_size'],
            seq_len=config['training']['seq_len']
        )
        
        self.model = TransformerLM(
            vocab_size=self.vocab_info['size'],
            d_model=config['model']['d_model'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            d_ff=config['model']['d_ff'],
            dropout=config['model']['dropout']
        ).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training on device: {self.device}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(data)
            
            loss = self.criterion(
                logits.view(-1, self.vocab_info['size']),
                targets.view(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.config['training']['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item() * data.numel()
            total_tokens += data.numel()
            
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / total_tokens
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                logits, _ = self.model(data)
                
                loss = self.criterion(
                    logits.view(-1, self.vocab_info['size']),
                    targets.view(-1)
                )
                
                total_loss += loss.item() * data.numel()
                total_tokens += data.numel()
        
        return total_loss / total_tokens
    
    def compute_perplexity(self, loss):
        """Compute perplexity from loss"""
        return np.exp(loss)
    
    def train(self):
        """Complete training process"""
        print("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            train_perplexity = self.compute_perplexity(train_loss)
            
            # Validation
            val_loss = self.validate()
            val_perplexity = self.compute_perplexity(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_perplexities.append(train_perplexity)
            self.val_perplexities.append(val_perplexity)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{self.config["training"]["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.2f}')
            print(f'  Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.2f}')
            print(f'  Time: {epoch_time:.2f}s, LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save training results"""
        os.makedirs('../results', exist_ok=True)
        
        # Save training curves
        self.plot_training_curves()
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'vocab_info': self.vocab_info
        }, '../results/transformer_model.pth')
        
        # Save training history
        history = {
            'epochs': list(range(1, len(self.train_losses) + 1)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_perplexity': self.train_perplexities,
            'val_perplexity': self.val_perplexities
        }
        
        import json
        with open('../results/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Perplexity curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_perplexities, label='Train Perplexity', marker='o')
        plt.plot(self.val_perplexities, label='Val Perplexity', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training and Validation Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('../results/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    trainer = Trainer(config)
    trainer.train()
    
    print("Training completed! Check results/ directory for outputs.")


if __name__ == '__main__':
    main()
