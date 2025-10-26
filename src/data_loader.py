import numpy as np
import wfdb
from pathlib import Path
import pandas as pd


class AFDBDataLoader:
    """Load ECG signals and AF annotations from MIT-BIH AFDB"""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.records = self._get_records()
        print(f"Found {len(self.records)} records in {data_path}")
    
    def _get_records(self):
        """Find all valid records"""
        hea_files = self.data_path.glob('*.hea')
        records = sorted([f.stem for f in hea_files if not f.stem.endswith('-')])
        return records
    
    def load_record(self, record_name, channels=None):
        """
        Load ECG signal and rhythm annotations
        
        Returns dict with:
            - signal: ECG signal array
            - fs: sampling frequency
            - rhythm_labels: array of rhythm labels for each sample
        """
        record_path = str(self.data_path / record_name)
        
        # Load signal
        if channels:
            record = wfdb.rdrecord(record_path, channels=channels)
        else:
            record = wfdb.rdrecord(record_path)
        
        # Load annotations
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            rhythm_labels = self._get_rhythm_labels(annotation, record.sig_len)
        except:
            rhythm_labels = None
        
        return {
            'record_name': record_name,
            'signal': record.p_signal,
            'fs': record.fs,
            'sig_len': record.sig_len,
            'duration': record.sig_len / record.fs,
            'rhythm_labels': rhythm_labels
        }
    
    def _get_rhythm_labels(self, annotation, sig_len):
        """Extract rhythm labels from annotations"""
        labels = np.full(sig_len, 'unknown', dtype=object)
        
        # Rhythm info is in aux_note field
        rhythm_changes = [(sample, aux) for sample, aux in 
                         zip(annotation.sample, annotation.aux_note)
                         if aux and aux.startswith('(')]
        
        # Map rhythm codes to names
        rhythm_map = {
            '(AFIB': 'atrial_fibrillation',
            '(N': 'normal',
            '(AFL': 'atrial_flutter',
        }
        
        # Apply labels to segments
        for i, (sample, aux) in enumerate(rhythm_changes):
            next_sample = rhythm_changes[i+1][0] if i+1 < len(rhythm_changes) else sig_len
            rhythm_name = rhythm_map.get(aux, 'other')
            labels[sample:next_sample] = rhythm_name
        
        return labels
    
    def get_stats(self):
        """Get dataset statistics"""
        stats = []
        
        for record in self.records:
            try:
                data = self.load_record(record, channels=[0])
                if data['rhythm_labels'] is None:
                    continue
                
                labels = data['rhythm_labels']
                af_mask = labels == 'atrial_fibrillation'
                normal_mask = labels == 'normal'
                
                af_duration = af_mask.sum() / data['fs']
                normal_duration = normal_mask.sum() / data['fs']
                total = af_duration + normal_duration
                
                if total > 0:
                    stats.append({
                        'record': record,
                        'af_duration_h': af_duration / 3600,
                        'normal_duration_h': normal_duration / 3600,
                        'af_percent': af_duration / total * 100
                    })
            except Exception as e:
                print(f"Error with {record}: {e}")
        
        df = pd.DataFrame(stats)
        print(f"\nTotal AF: {df['af_duration_h'].sum():.1f}h")
        print(f"Total Normal: {df['normal_duration_h'].sum():.1f}h")
        print(f"Overall AF%: {df['af_duration_h'].sum() / (df['af_duration_h'].sum() + df['normal_duration_h'].sum()) * 100:.1f}%")
        
        return df


def main():
    loader = AFDBDataLoader("data/MIT-BIH AFDB/files")
    
    # Load one record
    print("\nLoading sample record...")
    for record in loader.records[:5]:
        try:
            data = loader.load_record(record, channels=[0])
            print(f"\n{record}:")
            print(f"  Duration: {data['duration']/3600:.2f}h")
            print(f"  Signal shape: {data['signal'].shape}")
            
            if data['rhythm_labels'] is not None:
                unique = np.unique(data['rhythm_labels'])
                for rhythm in unique:
                    count = (data['rhythm_labels'] == rhythm).sum()
                    pct = count / len(data['rhythm_labels']) * 100
                    print(f"  {rhythm}: {pct:.1f}%")
            break
        except:
            continue
    
    # Get dataset statistics
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    df = loader.get_stats()
    print(f"\n{df.to_string()}")


if __name__ == "__main__":
    main()