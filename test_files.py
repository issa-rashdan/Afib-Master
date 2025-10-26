"""
Diagnostic script to test individual MIT-BIH AFDB files

This script iterates through the MIT-BIH AFDB dataset files, checking for the presence
of expected files (.hea, .dat, .atr, .qrs) for each record, and attempts to read
the header, signal, and annotations to ensure data integrity. It reports any issues found
with specific records to help identify problematic files in the dataset. 

This coded was created with the help of AI to assist in debugging dataset issues.

"""
import wfdb
import os
from pathlib import Path

data_path = Path("data/MIT-BIH AFDB/files")

# Get all .hea files
hea_files = sorted(data_path.glob("*.hea"))
print(f"Found {len(hea_files)} header files\n")

# Check what file types we have
print("="*60)
print("Checking available file types:")
print("="*60)
all_files = list(data_path.glob("*"))
extensions = set([f.suffix for f in all_files])
print(f"File extensions found: {sorted(extensions)}")

for ext in sorted(extensions):
    files = list(data_path.glob(f"*{ext}"))
    print(f"  {ext}: {len(files)} files")
print()

# Test first 5 records
for hea_file in hea_files[:5]:
    record_name = hea_file.stem
    record_path = str(data_path / record_name)
    
    print(f"{'='*60}")
    print(f"Testing: {record_name}")
    print(f"{'='*60}")
    
    # Check file existence
    dat_file = data_path / f"{record_name}.dat"
    atr_file = data_path / f"{record_name}.atr"
    qrs_file = data_path / f"{record_name}.qrs"
    hea_file_check = data_path / f"{record_name}.hea"
    
    print(f"Files for {record_name}:")
    print(f"  ✓ .hea: {hea_file_check.exists()}")
    print(f"  {'✓' if dat_file.exists() else '✗'} .dat: {dat_file.exists()} " + 
          (f"({dat_file.stat().st_size:,} bytes)" if dat_file.exists() else "(MISSING)"))
    print(f"  {'✓' if atr_file.exists() else '✗'} .atr: {atr_file.exists()} " +
          (f"({atr_file.stat().st_size:,} bytes)" if atr_file.exists() else "(missing)"))
    print(f"  {'✓' if qrs_file.exists() else '✗'} .qrs: {qrs_file.exists()} " +
          (f"({qrs_file.stat().st_size:,} bytes)" if qrs_file.exists() else "(missing)"))
    
    # Skip if no .dat file
    if not dat_file.exists():
        print("  ⚠️ Skipping - no .dat file (signal data missing)\n")
        continue
    
    # Try to read header
    try:
        header = wfdb.rdheader(record_path)
        print(f"\n✓ Header Info:")
        print(f"  - Sampling frequency: {header.fs} Hz")
        print(f"  - Duration: {header.sig_len/header.fs:.2f} seconds ({header.sig_len/header.fs/3600:.2f} hours)")
        print(f"  - Signals: {header.n_sig} channels - {header.sig_name}")
    except Exception as e:
        print(f"✗ Header error: {e}")
        continue
    
    # Try to read signal
    try:
        record = wfdb.rdrecord(record_path)
        print(f"\n✓ Signal Data:")
        print(f"  - Shape: {record.p_signal.shape}")
        print(f"  - Range: [{record.p_signal.min():.2f}, {record.p_signal.max():.2f}]")
    except Exception as e:
        print(f"✗ Signal error: {e}")
        continue
    
    # Try to read annotations (.atr)
    print(f"\n✓ Annotations (.atr):")
    try:
        annotation = wfdb.rdann(record_path, 'atr')
        print(f"  - Total annotations: {len(annotation.sample)}")
        
        # Show first 10 annotation symbols
        print(f"  - First 10 symbols: {annotation.symbol[:10]}")
        
        # Count rhythm annotations (start with '(')
        rhythm_anns = [s for s in annotation.symbol if s.startswith('(')]
        print(f"  - Rhythm annotations (starting with '('): {len(rhythm_anns)}")
        if rhythm_anns:
            print(f"  - Rhythm types found: {set(rhythm_anns)}")
        else:
            print(f"  ⚠️ No rhythm annotations found in .atr file!")
    except Exception as e:
        print(f"  ✗ Cannot read .atr: {e}")
    
    # Try to read .qrs annotations if they exist
    if qrs_file.exists():
        print(f"\n✓ QRS Annotations (.qrs):")
        try:
            qrs_ann = wfdb.rdann(record_path, 'qrs')
            print(f"  - Total annotations: {len(qrs_ann.sample)}")
            print(f"  - First 10 symbols: {qrs_ann.symbol[:10]}")
        except Exception as e:
            print(f"  ✗ Cannot read .qrs: {e}")
    
    print()

print("\n" + "="*60)
print("Summary:")
print("="*60)

# Count records with complete data
complete_records = []
for hea_file in hea_files:
    record_name = hea_file.stem
    dat_file = data_path / f"{record_name}.dat"
    if dat_file.exists() and dat_file.stat().st_size > 0:
        complete_records.append(record_name)

print(f"Records with complete data (.dat file present): {len(complete_records)}/{len(hea_files)}")
print(f"Complete records: {complete_records[:10]}..." if len(complete_records) > 10 else f"Complete records: {complete_records}")