"""
BERT Text Classification Training Script
Fine-tunes bert-base-uncased for binary sentiment classification
and saves model weights and tokenizer to ./saved_model and ./tokenizer
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')



class Config:
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 64
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    NUM_CLASSES = 2
    SAVE_MODEL_PATH = './saved_model'
    TOKENIZER_PATH = './tokenizer'


config = Config()


def create_sample_dataset():
    """Create a sample dataset of movie reviews for binary sentiment classification"""
    positive_reviews = [
        "This movie is absolutely fantastic! Great acting and storyline.",
        "I loved every minute of this film. Highly recommended!",
        "An amazing masterpiece with incredible performances!",
        "Brilliant direction and exceptional screenplay. A must-watch!",
        "Truly heartwarming and emotionally resonant. One of the best movies this year.",
        "Spectacular visuals and a thrilling musical score kept me on the edge of my seat.",
        "Outstanding performance by the entire cast. Thoroughly entertaining from start to finish.",
        "A compelling and thoughtful narrative that stays with you long after the credits roll.",
        "Hilarious, charming, and wonderfully acted. A feel-good classic.",
        "Flawless cinematography and powerhouse performances. Five stars all the way!"
    ] * 6  # 60 positive samples

    negative_reviews = [
        "This movie was terrible. Waste of time and money.",
        "Boring storyline with poor acting throughout.",
        "The plot was confusing, slow-paced, and utterly predictable.",
        "Dreadful script and lifeless performances made this unwatchable.",
        "Completely disappointed by this overhyped and tedious film.",
        "Horrible dialogue and zero chemistry between the leads.",
        "A total disaster. Poor directing and sloppy editing from start to finish.",
        "I could barely stay awake. Extremely dull and repetitive.",
        "One of the worst movies I have ever seen. Avoid at all costs.",
        "Cringeworthy dialogue, cheap effects, and a completely pointless plot."
    ] * 6  # 60 negative samples

    data = []
    for review in positive_reviews:
        data.append({'text': review, 'label': 1})
    for review in negative_reviews:
        data.append({'text': review, 'label': 0})

    df = pd.DataFrame(data)
    return df


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    use_cuda = (device.type == 'cuda')
    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        if use_cuda:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_predictions / total_predictions:.4f}'
        })

    return total_loss / len(data_loader), correct_predictions / total_predictions



def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_predictions
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, all_predictions, all_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    print("[INFO] Creating dataset...")
    df = create_sample_dataset()
    print(f"[INFO] Dataset size: {len(df)} samples")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    print("[INFO] Loading BERT tokenizer...")
    # Use existing local tokenizer if present or download bert-base-uncased
    if os.path.exists(config.TOKENIZER_PATH) and os.path.exists(os.path.join(config.TOKENIZER_PATH, 'vocab.txt')):
        tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_PATH)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = TextDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    test_dataset = TextDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"[INFO] Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    print("[INFO] Initializing BERT sequence classification model...")
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_CLASSES
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print("[INFO] Starting training...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0.0

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        val_loss, val_acc, val_predictions, val_labels = evaluate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Epoch {epoch + 1} - Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        if val_acc >= best_accuracy:
            best_accuracy = val_acc
            print(f"[SAVED] Saving best model with validation accuracy: {best_accuracy:.4f}")

            # Save to root and subfolder
            save_dirs = [config.SAVE_MODEL_PATH, os.path.join('BERT Sentiment Analysis 1', 'saved_model')]
            tok_dirs = [config.TOKENIZER_PATH, os.path.join('BERT Sentiment Analysis 1', 'tokenizer')]

            torch.cuda.empty_cache()
            for s_dir in save_dirs:
                os.makedirs(s_dir, exist_ok=True)
                model.save_pretrained(s_dir, safe_serialization=False)

            for t_dir in tok_dirs:
                os.makedirs(t_dir, exist_ok=True)
                tokenizer.save_pretrained(t_dir)

    print(f"\n[DONE] Training completed! Best accuracy: {best_accuracy:.4f}")

    print("\n[EVAL] Final Evaluation on Test Set:")
    test_loss, test_acc, test_predictions, test_labels = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['Negative', 'Positive']))


    # Generate training results visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    cm = confusion_matrix(test_labels, test_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved training results visualization to 'training_results.png'")

    # Save training history
    training_history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_accuracy': best_accuracy,
        'test_accuracy': test_acc
    }

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)

    # Also save to subfolder
    sub_hist = os.path.join('BERT Sentiment Analysis 1', 'training_history.pkl')
    with open(sub_hist, 'wb') as f:
        pickle.dump(training_history, f)

    print("Saved training history to 'training_history.pkl'")


if __name__ == '__main__':
    main()
