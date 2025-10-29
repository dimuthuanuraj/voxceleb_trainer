# NaN Debugging Guide

## What Was Added

### 1. Training NaN Detection (SpeakerNet_performance_updated.py)
When NaN/Inf is detected during training, the system will:
- **Print detailed statistics** about the problematic batch
- **Save debug data** to `exps/<experiment_name>/debug_nan/`
- **Skip the batch** and continue training (instead of crashing)

Information captured:
- Data tensor statistics (min, max, mean, NaN/Inf counts)
- Label statistics (unique values, distribution)
- Model gradient statistics (which parameters have NaN gradients)
- Full batch data saved to disk for offline analysis

### 2. Validation NaN Detection (SpeakerNet_performance_updated.py)
During validation, when NaN scores are computed:
- **Prints which file pair** caused the NaN
- **Shows feature statistics** for both files
- **Continues evaluation** with filtered scores (via tuneThreshold.py)

### 3. Automatic NaN Filtering (tuneThreshold.py)
- **Automatically filters** NaN scores before ROC curve calculation
- **Prevents crashes** in scikit-learn
- **Reports** how many NaN scores were filtered

## How to Use

### When Training Encounters NaN:

1. **Watch the output** - you'll see a detailed report like:
   ```
   ================================================================================
   🚨 NaN/Inf LOSS DETECTED AT BATCH 1234
   ================================================================================
   Loss value: nan
   Precision: 0.0
   
   --- DATA STATISTICS ---
   Data shape: torch.Size([64, 200, 64])
   Data min: -15.234567, max: 12.345678, mean: 0.123456
   ...
   
   💾 Saved debug data to: exps/mini_voxceleb1_experiment_9/debug_nan/nan_batch_1234_epoch_18.pt
   ```

2. **Training will continue** - the problematic batch is skipped

3. **Analyze the saved data**:
   ```bash
   # Analyze all NaN debug files
   python3 analyze_nan_debug.py
   
   # Or analyze a specific file
   python3 analyze_nan_debug.py exps/mini_voxceleb1_experiment_9/debug_nan/nan_batch_1234_epoch_18.pt
   ```

### Analysis Script Output

The `analyze_nan_debug.py` script will show:
- **Basic info**: batch index, epoch, loss value
- **Data tensor analysis**: 
  - Shape, dtype, statistics
  - Top 10 largest/smallest values
  - Percentile distribution
  - NaN/Inf counts and locations
- **Label tensor analysis**:
  - Unique labels and distribution
  - Out-of-range detection
- **Potential issues**:
  - Identifies specific problems
  - Provides targeted recommendations

### Common NaN Causes & Solutions

#### 1. Input Data Contains NaN/Inf
**Symptoms**: `Input data contains X NaN values`
**Solutions**:
- Check audio files for corruption
- Inspect data augmentation (MUSAN/RIR paths)
- Verify audio loading in DatasetLoader

#### 2. Very Large/Small Values
**Symptoms**: `Very large values detected (max: 1.23e+08)`
**Solutions**:
- Add input normalization
- Reduce learning rate
- Check augmentation parameters

#### 3. Model/Loss Instability
**Symptoms**: Input looks normal but NaN in loss
**Solutions**:
- **Reduce learning rate**: `lr: 0.0005` (currently 0.001)
- **Reduce AAMSoftmax scale**: `scale: 20` (currently 30)
- **Disable mixed precision**: `mixedprec: false`
- **Check margin**: Try `margin: 0.1` instead of 0.2

#### 4. Label Issues
**Symptoms**: `Label out of range: max label X >= 140`
**Solutions**:
- Check nClasses matches your dataset
- Verify data_label indexing in DatasetLoader

## Quick Fixes to Try

### Option 1: Reduce Learning Rate
```yaml
# In configs/mini_voxceleb1_config.yaml
lr: 0.0005  # Was 0.001
```

### Option 2: Reduce AAMSoftmax Scale
```yaml
# In configs/mini_voxceleb1_config.yaml
scale: 20  # Was 30
```

### Option 3: Disable Mixed Precision
```yaml
# In configs/mini_voxceleb1_config.yaml
mixedprec: false  # Was true
```

### Option 4: Increase Gradient Clipping
Currently set to `max_norm=5.0`. You could try:
- Edit `SpeakerNet_performance_updated.py`
- Change `max_norm=5.0` to `max_norm=1.0` (more aggressive)

## Files Modified

1. **SpeakerNet_performance_updated.py**
   - Added NaN detection in training loop (2 places: mixedprec and non-mixedprec)
   - Added NaN detection in validation scoring
   - Saves debug data when NaN detected

2. **tuneThreshold.py**
   - Added NaN filtering before ROC curve calculation
   - Prevents crashes from NaN scores
   - Reports filtered counts

3. **analyze_nan_debug.py** (NEW)
   - Comprehensive analysis tool for saved debug files
   - Identifies root causes
   - Provides specific recommendations

## Example Workflow

```bash
# 1. Run training (it will handle NaN gracefully now)
python3 trainSpeakerNet_performance_updated.py \
    --config configs/mini_voxceleb1_config.yaml \
    --save_path exps/debug_run

# 2. If NaN occurs, check the terminal output for details

# 3. Analyze the saved debug data
python3 analyze_nan_debug.py

# 4. Based on recommendations, adjust config and retry
# For example, lower learning rate:
#   lr: 0.0005
#   scale: 20

# 5. Resume or restart training with adjusted parameters
```

## Tips

- **Don't panic**: NaN is common in speaker recognition training
- **Check systematically**: Data → Model → Loss → Optimizer
- **Start conservative**: Lower LR, lower scale, disable mixed precision
- **Monitor early**: Watch first few epochs closely
- **Save checkpoints**: Use `test_interval: 1` to save frequently
- **Analyze patterns**: Does NaN always occur at same epoch/batch?

## Need More Help?

Check the debug output carefully - it will tell you:
1. **Where** NaN appeared (batch index, epoch)
2. **What** the data looked like (statistics)
3. **Which** model parameters were affected (gradient analysis)

Use this information to narrow down the root cause!
