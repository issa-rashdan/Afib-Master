import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle

from data_loader import AFDBDataLoader


class AFDataset(Dataset):
    
    def __init__(self, windows, labels):
        self.windows = torch.FloatTensor(windows).unsqueeze(1)  # Add channel dim
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


def create_windows(signal, labels, window_size=1000, stride=500):
    """Create sliding windows from signal"""
    windows, window_labels = [], []
    
    for start in range(0, len(signal) - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end])
        # Label is majority vote in window
        window_labels.append(1 if np.mean(labels[start:end]) >= 0.5 else 0)
    
    return np.array(windows), np.array(window_labels)


def build_datasets(data_path, window_size=1000, stride=500, test_size=0.2, save_path=None):
  
    loader = AFDBDataLoader(data_path)
    
    # Get usable records
    records = [r for r in loader.records 
               if (Path(data_path) / f"{r}.dat").exists()]
    
    # Split by patient (80/20)
    train_records, test_records = train_test_split(
        records, test_size=test_size, random_state=42
    )
    
    print(f"Train records: {len(train_records)}, Test records: {len(test_records)}")
    
    # Process each split
    def process_records(record_list, split_name):
        all_windows, all_labels = [], []
        
        for record in record_list:
            try:
                data = loader.load_record(record, channels=[0])
                signal = data['signal'].squeeze()
                rhythm_labels = data['rhythm_labels']
                
                if rhythm_labels is None:
                    continue
                
                # Convert to binary (AF=1, other=0)
                binary_labels = (rhythm_labels == 'atrial_fibrillation').astype(int)
                
                # Create windows
                windows, labels = create_windows(signal, binary_labels, window_size, stride)
                all_windows.append(windows)
                all_labels.append(labels)
                
                print(f"{record}: {len(windows)} windows ({labels.sum()} AF)")
                
            except Exception as e:
                print(f"Error with {record}: {e}")
        
        if all_windows:
            return np.vstack(all_windows), np.concatenate(all_labels)
        return np.array([]), np.array([])
    
    print("\nProcessing train records...")
    train_X, train_y = process_records(train_records, "train")
    
    print("\nProcessing test records...")
    test_X, test_y = process_records(test_records, "test")
    
    # Create datasets
    train_dataset = AFDataset(train_X, train_y)
    test_dataset = AFDataset(test_X, test_y)
    
    # Print stats
    print(f"\nTrain: {len(train_dataset)} windows, {train_y.sum()} AF ({train_y.mean()*100:.1f}%)")
    print(f"Test: {len(test_dataset)} windows, {test_y.sum()} AF ({test_y.mean()*100:.1f}%)")
    
    # Save if requested
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        with open(Path(save_path) / 'datasets.pkl', 'wb') as f:
            pickle.dump({
                'train': train_dataset,
                'test': test_dataset,
                'train_records': train_records,
                'test_records': test_records
            }, f)
        print(f"Saved to {save_path}")
    
    return train_dataset, test_dataset


def main():
    # Build datasets
    train_ds, test_ds = build_datasets(
        data_path="data/MIT-BIH AFDB/files",
        save_path="data/processed"
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # Test
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch shape: {batch_x.shape}, Labels: {batch_y.shape}")


if __name__ == "__main__":
    main()