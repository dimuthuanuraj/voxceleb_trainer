#!/usr/bin/env python3
"""
Analyze NaN Debug Files
This script analyzes the debug files saved when NaN is detected during training.
"""

import torch
import numpy as np
import sys
import os
import glob

def analyze_debug_file(file_path):
    """Analyze a single debug file saved during NaN detection."""
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {file_path}")
    print(f"{'='*80}")
    
    try:
        debug_data = torch.load(file_path, map_location='cpu')
        
        print(f"\n📊 BASIC INFO:")
        print(f"   Batch Index: {debug_data['batch_idx']}")
        print(f"   Epoch: {debug_data['epoch']}")
        print(f"   Loss: {debug_data['loss']}")
        print(f"   Precision: {debug_data['prec1']}")
        
        data = debug_data['data']
        label = debug_data['label']
        
        print(f"\n📈 DATA TENSOR ANALYSIS:")
        print(f"   Shape: {data.shape}")
        print(f"   Dtype: {data.dtype}")
        print(f"   Min: {data.min().item():.6f}")
        print(f"   Max: {data.max().item():.6f}")
        print(f"   Mean: {data.mean().item():.6f}")
        print(f"   Std: {data.std().item():.6f}")
        print(f"   Contains NaN: {torch.isnan(data).any().item()}")
        print(f"   Contains Inf: {torch.isinf(data).any().item()}")
        print(f"   Number of NaN: {torch.isnan(data).sum().item()}")
        print(f"   Number of Inf: {torch.isinf(data).sum().item()}")
        
        # Check for extreme values
        data_flat = data.flatten()
        sorted_values, _ = torch.sort(data_flat)
        print(f"\n   Top 10 largest values: {sorted_values[-10:].tolist()}")
        print(f"   Top 10 smallest values: {sorted_values[:10].tolist()}")
        
        # Distribution analysis
        finite_data = data[torch.isfinite(data)]
        if len(finite_data) > 0:
            print(f"\n   Statistics (excluding NaN/Inf):")
            print(f"      Min: {finite_data.min().item():.6f}")
            print(f"      Max: {finite_data.max().item():.6f}")
            print(f"      Mean: {finite_data.mean().item():.6f}")
            print(f"      Std: {finite_data.std().item():.6f}")
            print(f"      Median: {finite_data.median().item():.6f}")
            
            # Percentiles
            percentiles = [1, 5, 25, 50, 75, 95, 99]
            quantiles = [finite_data.quantile(p/100.0).item() for p in percentiles]
            print(f"\n   Percentiles:")
            for p, q in zip(percentiles, quantiles):
                print(f"      {p:3d}%: {q:.6f}")
        
        print(f"\n🏷️  LABEL TENSOR ANALYSIS:")
        print(f"   Shape: {label.shape}")
        print(f"   Dtype: {label.dtype}")
        print(f"   Unique labels: {torch.unique(label).tolist()}")
        print(f"   Min: {label.min().item()}")
        print(f"   Max: {label.max().item()}")
        print(f"   Label distribution:")
        for lbl in torch.unique(label):
            count = (label == lbl).sum().item()
            print(f"      Label {lbl}: {count} samples")
        
        # Check for any patterns
        print(f"\n🔍 POTENTIAL ISSUES:")
        issues = []
        
        if torch.isnan(data).any():
            issues.append(f"❌ Input data contains {torch.isnan(data).sum().item()} NaN values")
        
        if torch.isinf(data).any():
            issues.append(f"❌ Input data contains {torch.isinf(data).sum().item()} Inf values")
        
        if finite_data.max().item() > 1e6:
            issues.append(f"⚠️  Very large values detected (max: {finite_data.max().item():.2e})")
        
        if finite_data.min().item() < -1e6:
            issues.append(f"⚠️  Very small values detected (min: {finite_data.min().item():.2e})")
        
        if finite_data.std().item() > 1e3:
            issues.append(f"⚠️  Very high variance (std: {finite_data.std().item():.2e})")
        
        if label.max().item() >= 140:  # Assuming nClasses=140
            issues.append(f"❌ Label out of range: max label {label.max().item()} >= 140")
        
        if len(issues) == 0:
            print("   ✅ No obvious data issues detected (NaN likely from model/loss)")
        else:
            for issue in issues:
                print(f"   {issue}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("   • Check data augmentation - may be producing invalid values")
            print("   • Check audio loading - files may be corrupted")
        elif finite_data.max().item() > 1e3 or finite_data.std().item() > 1e2:
            print("   • Consider adding input normalization")
            print("   • Reduce learning rate")
        else:
            print("   • Input data looks normal - issue likely in model/loss")
            print("   • Check AAMSoftmax scale/margin parameters")
            print("   • Try reducing learning rate")
            print("   • Consider disabling mixed precision (mixedprec: false)")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) > 1:
        # Analyze specific file
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            analyze_debug_file(file_path)
        else:
            print(f"File not found: {file_path}")
    else:
        # Find all debug files in current experiment
        debug_dirs = glob.glob("exps/*/debug_nan")
        
        if not debug_dirs:
            print("No debug_nan directories found in exps/")
            print("\nUsage:")
            print(f"  {sys.argv[0]} <path_to_debug_file.pt>")
            print(f"  {sys.argv[0]}  # Analyze all debug files")
            return
        
        all_files = []
        for debug_dir in debug_dirs:
            files = glob.glob(os.path.join(debug_dir, "*.pt"))
            all_files.extend(files)
        
        if not all_files:
            print("No debug files found")
            return
        
        print(f"Found {len(all_files)} debug file(s)")
        
        # Analyze all files
        for file_path in sorted(all_files):
            analyze_debug_file(file_path)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete! Analyzed {len(all_files)} file(s)")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()
