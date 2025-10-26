"""
Deep annotation analysis for MIT-BIH AFDB
Check all annotation attributes to find rhythm labels

This was with the goal of finding rhythm information in the .atr or .qrs files, was created by 
AI to help debug why rhythm labels were not being found in the annotations.

"""
import wfdb
from pathlib import Path
import numpy as np

data_path = Path("data/MIT-BIH AFDB/files")

# Test one complete record
record_name = "04015"
record_path = str(data_path / record_name)

print("="*70)
print(f"Deep Analysis of Record: {record_name}")
print("="*70)

# Load record
record = wfdb.rdrecord(record_path)
print(f"\nSignal Info:")
print(f"  - Duration: {record.sig_len/record.fs:.2f} seconds ({record.sig_len/record.fs/3600:.2f} hours)")
print(f"  - Sampling rate: {record.fs} Hz")
print(f"  - Channels: {record.sig_name}")

# Load .atr annotations
print(f"\n{'='*70}")
print("Analyzing .atr file:")
print("="*70)
annotation = wfdb.rdann(record_path, 'atr')

print(f"\nAnnotation object attributes:")
for attr in dir(annotation):
    if not attr.startswith('_'):
        print(f"  - {attr}: {getattr(annotation, attr) if not attr == 'aux_note' else '...'}")

print(f"\n.atr Annotation Details:")
print(f"  Total annotations: {len(annotation.sample)}")
print(f"  Unique symbols: {set(annotation.symbol)}")

# Check aux_note for rhythm information
print(f"\n  Auxiliary notes (first 20):")
for i, (sample, symbol, aux) in enumerate(zip(annotation.sample[:20], 
                                                annotation.symbol[:20], 
                                                annotation.aux_note[:20])):
    time_sec = sample / record.fs
    print(f"    [{i}] Time: {time_sec:.2f}s, Symbol: '{symbol}', Aux: '{aux}'")

# Check if any aux_note contains rhythm info
rhythm_indicators = ['AFIB', 'AFL', 'N', 'AB', 'SVTA', 'VT', 'IVR']
print(f"\n  Checking aux_note for rhythm keywords...")
for indicator in rhythm_indicators:
    matches = [i for i, aux in enumerate(annotation.aux_note) if indicator in str(aux)]
    if matches:
        print(f"    Found '{indicator}' in {len(matches)} annotations")
        # Show first match
        idx = matches[0]
        print(f"      Example: idx={idx}, sample={annotation.sample[idx]}, " +
              f"symbol='{annotation.symbol[idx]}', aux='{annotation.aux_note[idx]}'")

# Try loading .qrs file
print(f"\n{'='*70}")
print("Analyzing .qrs file:")
print("="*70)
qrs_annotation = wfdb.rdann(record_path, 'qrs')

print(f"\n.qrs Annotation Details:")
print(f"  Total annotations: {len(qrs_annotation.sample)}")
print(f"  Unique symbols: {set(qrs_annotation.symbol)}")
print(f"  First 20 symbols: {qrs_annotation.symbol[:20]}")

# Check for rhythm changes in aux_note
print(f"\n  Checking .qrs aux_note (first 20):")
for i in range(min(20, len(qrs_annotation.aux_note))):
    if qrs_annotation.aux_note[i]:
        time_sec = qrs_annotation.sample[i] / record.fs
        print(f"    [{i}] Time: {time_sec:.2f}s, Symbol: '{qrs_annotation.symbol[i]}', " +
              f"Aux: '{qrs_annotation.aux_note[i]}'")

# Alternative: Check if rhythm is encoded in subtype or chan
if hasattr(annotation, 'subtype'):
    print(f"\n  Subtype info: {set(annotation.subtype)}")
if hasattr(annotation, 'chan'):
    print(f"  Chan info: {set(annotation.chan)}")
if hasattr(annotation, 'num'):
    print(f"  Num info: {set(annotation.num)}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)

# Try alternative approach: Check header for annotation info
print("\nChecking header for annotation information...")
header = wfdb.rdheader(record_path)
print(f"Header comments: {header.comments if hasattr(header, 'comments') else 'None'}")

# List all files for this record
print(f"\nAll files for record {record_name}:")
for file in sorted(data_path.glob(f"{record_name}*")):
    print(f"  - {file.name} ({file.stat().st_size:,} bytes)")