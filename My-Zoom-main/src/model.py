"""
Model module for the My Zoom project.
This module implements the transformer-based model for feedback validation.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class FeedbackDataset(Dataset):
    """Dataset class for feedback data"""
    
    def __init__(self, encodings):
        """
        Initialize the dataset with encodings.
        
        Args:
            encodings (dict): Tokenized data with input_ids, attention_mask, and labels
        """
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = encodings.get('labels', None)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class ZoomFeedbackClassifier(nn.Module):
    """BERT-based classifier for Zoom feedback validation"""
    
    def __init__(self, pretrained_model_name='bert-base-uncased', num_labels=2, dropout_prob=0.1):
        """
        Initialize the feedback classifier.
        
        Args:
            pretrained_model_name (str): Name of the pretrained BERT model
            num_labels (int): Number of output labels
            dropout_prob (float): Dropout probability
        """
        super(ZoomFeedbackClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs (unused)
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            tuple: Loss (if labels provided) and logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, 2), labels)
            
        return (loss, logits) if loss is not None else logits

class ModelTrainer:
    """Trainer class for the Zoom feedback classifier model"""
    
    def __init__(self, model, device, lr=2e-5, eps=1e-8):
        """
        Initialize the model trainer.
        
        Args:
            model (ZoomFeedbackClassifier): The model to train
            device (torch.device): Device to train on (CPU or GPU)
            lr (float): Learning rate
            eps (float): Epsilon for Adam optimizer
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
        self.scheduler = None
        
    def create_dataloaders(self, train_encodings, eval_encodings, batch_size=16):
        """
        Create training and evaluation dataloaders.
        
        Args:
            train_encodings (dict): Training data encodings
            eval_encodings (dict): Evaluation data encodings
            batch_size (int): Batch size
            
        Returns:
            tuple: Training and evaluation dataloaders
        """
        train_dataset = FeedbackDataset(train_encodings)
        eval_dataset = FeedbackDataset(eval_encodings)
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_dataloader, eval_dataloader
    
    def create_scheduler(self, train_dataloader, epochs=4):
        """
        Create a learning rate scheduler.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            epochs (int): Number of training epochs
            
        Returns:
            LRScheduler: Learning rate scheduler
        """
        total_steps = len(train_dataloader) * epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        return self.scheduler
    
    def train_epoch(self, train_dataloader):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, _ = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate schedule
            if self.scheduler:
                self.scheduler.step()
                
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_dataloader)
        return avg_loss
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model.
        
        Args:
            eval_dataloader (DataLoader): Evaluation dataloader
            
        Returns:
            tuple: Evaluation metrics (accuracy, precision, recall, f1)
        """
        self.model.eval()
        
        true_labels = []
        predictions = []
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                loss, logits = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_eval_loss += loss.item()
                
                # Get predictions
                batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                batch_labels = batch['labels'].detach().cpu().numpy()
                
                true_labels.extend(batch_labels.tolist())
                predictions.extend(batch_preds.tolist())
                
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            average='binary'
        )
        
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        
        metrics = {
            'loss': avg_eval_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def train(self, train_dataloader, eval_dataloader, epochs=4, save_path=None):
        """
        Train the model for multiple epochs.
        
        Args:
            train_dataloader (DataLoader): Training dataloader
            eval_dataloader (DataLoader): Evaluation dataloader
            epochs (int): Number of training epochs
            save_path (str): Path to save the model (optional)
            
        Returns:
            dict: Training history
        """
        # Create scheduler if not already created
        if not self.scheduler:
            self.create_scheduler(train_dataloader, epochs)
            
        # Initialize best metrics
        best_f1 = 0.0
        history = {
            'train_loss': [],
            'eval_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(train_dataloader)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            metrics = self.evaluate(eval_dataloader)
            history['eval_loss'].append(metrics['loss'])
            history['accuracy'].append(metrics['accuracy'])
            history['precision'].append(metrics['precision'])
            history['recall'].append(metrics['recall'])
            history['f1'].append(metrics['f1'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Eval Loss: {metrics['loss']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            
            # Save model if it's the best so far
            if save_path and metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
                
        return history

# Example usage
if __name__ == "__main__":
    # This is a test to check if the model works
    from transformers import BertTokenizer
    
    # Create a toy dataset
    texts = [
        "this app is great for online classes",
        "i can't download this app",
        "very practical and easy to use",
        "app crashes when trying to join a meeting"
    ]
    labels = [1, 1, 1, 0]
    
    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(
        texts, 
        padding='max_length', 
        truncation=True, 
        max_length=128,
        return_tensors='pt'
    )
    encodings['labels'] = labels
    
    # Create model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZoomFeedbackClassifier()
    trainer = ModelTrainer(model, device)
    
    # Test dataset creation
    dataset = FeedbackDataset(encodings)
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
