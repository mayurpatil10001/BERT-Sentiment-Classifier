"""
BERT Text Classification Inference Script
This script demonstrates how to load and use the trained model for inference
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json
import os
import argparse
from datetime import datetime

class BERTTextClassifier:
    def __init__(self, model_path='./saved_model', tokenizer_path='./tokenizer', max_length=128):
        """
        Initialize the BERT Text Classifier
        
        Args:
            model_path (str): Path to saved model
            tokenizer_path (str): Path to saved tokenizer
            max_length (int): Maximum sequence length
        """
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model on device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Class labels
        self.labels = {0: 'Negative', 1: 'Positive'}
        
        print("Model loaded successfully!")
    
    def tokenize_text(self, text, return_details=False):
        """
        Tokenize input text using BERT tokenizer
        
        Args:
            text (str): Input text to tokenize
            return_details (bool): Whether to return detailed tokenization info
            
        Returns:
            dict: Tokenization results
        """
        # Basic tokenization
        tokens = self.tokenizer.tokenize(text)
        
        # Full encoding with special tokens
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        if return_details:
            # Get detailed information
            input_ids = encoding['input_ids'][0].numpy()
            attention_mask = encoding['attention_mask'][0].numpy()
            tokens_with_special = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            return {
                'original_text': text,
                'basic_tokens': tokens,
                'tokens_with_special': tokens_with_special,
                'input_ids': input_ids.tolist(),
                'attention_mask': attention_mask.tolist(),
                'num_tokens': len(tokens),
                'sequence_length': len(tokens_with_special),
                'encoding': encoding
            }
        else:
            return {
                'tokens': tokens,
                'encoding': encoding
            }
    
    def predict(self, text, return_probabilities=True):
        """
        Predict sentiment for input text
        
        Args:
            text (str): Input text for prediction
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Tokenize and encode
        tokenization_result = self.tokenize_text(text)
        encoding = tokenization_result['encoding']
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
        
        # Prepare results
        pred_class_idx = predicted_class.item()
        pred_label = self.labels[pred_class_idx]
        
        result = {
            'text': text,
            'predicted_class': pred_class_idx,
            'predicted_label': pred_label,
            'confidence': probabilities[0][pred_class_idx].item()
        }
        
        if return_probabilities:
            result['probabilities'] = {
                'negative': probabilities[0][0].item(),
                'positive': probabilities[0][1].item()
            }
        
        return result
    
    def batch_predict(self, texts, batch_size=8):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def analyze_text_detailed(self, text):
        """
        Perform detailed analysis of text including tokenization and prediction
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Detailed analysis results
        """
        # Get detailed tokenization
        tokenization = self.tokenize_text(text, return_details=True)
        
        # Get prediction
        prediction = self.predict(text)
        
        # Combine results
        analysis = {
            'input_text': text,
            'tokenization': {
                'basic_tokens': tokenization['basic_tokens'],
                'tokens_with_special': tokenization['tokens_with_special'],
                'num_basic_tokens': tokenization['num_tokens'],
                'sequence_length': tokenization['sequence_length'],
                'input_ids': tokenization['input_ids'],
                'attention_mask': tokenization['attention_mask']
            },
            'prediction': prediction,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description='BERT Text Classification Inference')
    parser.add_argument('--text', type=str, help='Text to classify')
    parser.add_argument('--file', type=str, help='File containing texts to classify (one per line)')
    parser.add_argument('--model_path', type=str, default='./saved_model', help='Path to saved model')
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer', help='Path to saved tokenizer')
    parser.add_argument('--output', type=str, help='Output file for results (JSON format)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed tokenization analysis')
    
    args = parser.parse_args()
    
    # Initialize classifier
    try:
        classifier = BERTTextClassifier(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run the training notebook first to create the saved model.")
        return
    
    results = []
    
    if args.text:
        # Single text prediction
        if args.detailed:
            result = classifier.analyze_text_detailed(args.text)
            print("\n" + "="*60)
            print("DETAILED TEXT ANALYSIS")
            print("="*60)
            print(f"Input Text: {result['input_text']}")
            print(f"\nBasic Tokens: {result['tokenization']['basic_tokens']}")
            print(f"Tokens with Special: {result['tokenization']['tokens_with_special'][:10]}...")  # Show first 10
            print(f"Number of Tokens: {result['tokenization']['num_basic_tokens']}")
            print(f"Sequence Length: {result['tokenization']['sequence_length']}")
            print(f"\nPrediction: {result['prediction']['predicted_label']}")
            print(f"Confidence: {result['prediction']['confidence']:.4f}")
            print(f"Probabilities: {result['prediction']['probabilities']}")
        else:
            result = classifier.predict(args.text)
            print(f"\nText: {result['text']}")
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.4f}")
        
        results.append(result)
    
    elif args.file:
        # Batch prediction from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(texts)} texts from {args.file}...")
            results = classifier.batch_predict(texts)
            
            print("\nResults:")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                print(f"{i}. Text: {result['text'][:50]}...")
                print(f"   Prediction: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
        
        except FileNotFoundError:
            print(f"File {args.file} not found.")
            return
    
    else:
        # Interactive mode
        print("\n" + "="*60)
        print("INTERACTIVE BERT TEXT CLASSIFIER")
        print("="*60)
        print("Enter text to classify (type 'quit' to exit):")
        
        while True:
            text = input("\n> ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if text:
                result = classifier.predict(text)
                print(f"Prediction: {result['predicted_label']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"Probabilities: Negative={result['probabilities']['negative']:.3f}, "
                      f"Positive={result['probabilities']['positive']:.3f}")
    
    # Save results to file if specified
    if args.output and results:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        # Demo mode
        print("BERT Text Classification Demo")
        print("="*40)
        
        try:
            classifier = BERTTextClassifier()
            
            # Test examples
            test_texts = [
                "This movie is absolutely fantastic! Great acting and storyline.",
                "I loved every minute of this film. Highly recommended!",
                "This movie was terrible. Waste of time and money.",
                "Boring storyline with poor acting throughout.",
                "The acting was decent but the plot was confusing."
            ]
            
            print("Testing with sample texts:")
            print("-" * 40)
            
            for text in test_texts:
                result = classifier.predict(text)
                print(f"Text: {text}")
                print(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
                print()
            
            # Detailed analysis example
            print("\nDetailed Analysis Example:")
            print("-" * 40)
            detailed_result = classifier.analyze_text_detailed(test_texts[0])
            print(f"Basic Tokens: {detailed_result['tokenization']['basic_tokens']}")
            print(f"Sequence Length: {detailed_result['tokenization']['sequence_length']}")
            print(f"Prediction: {detailed_result['prediction']['predicted_label']}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please run the training notebook first to create the saved model.")
    
    else:
        main()