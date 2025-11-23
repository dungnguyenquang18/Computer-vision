# coding: utf-8
"""
Test script for trained models
Evaluate trained CNN and U-Net models on test data
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[ERROR] nibabel not installed")
    sys.exit(1)


def load_patient_data(data_dir, patient_id):
    """Load patient MRI and ground truth"""
    data_dir = Path(data_dir)
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    # Load MRI
    volume_list = []
    for mod in modalities:
        filepath = data_dir / f"{patient_id}_{mod}.nii.gz"
        nifti = nib.load(str(filepath))
        volume_list.append(nifti.get_fdata())
    
    volume = np.stack(volume_list, axis=-1).astype(np.float32)
    
    # Load ground truth
    seg_file = data_dir / f"{patient_id}_seg.nii.gz"
    if seg_file.exists():
        nifti = nib.load(str(seg_file))
        gt = nifti.get_fdata().astype(np.uint8)
    else:
        gt = None
    
    return volume, gt


def calculate_metrics(pred, gt):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Dice scores for each label
    labels = {0: 'Background', 1: 'Necrotic', 2: 'Edema', 4: 'Enhancing'}
    
    dice_scores = {}
    for label, name in labels.items():
        if label not in np.unique(gt):
            continue
        
        pred_mask = (pred == label)
        gt_mask = (gt == label)
        
        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)
        
        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = 2.0 * intersection / union
        
        dice_scores[label] = dice
    
    metrics['dice_scores'] = dice_scores
    
    # Mean Dice (tumor regions only)
    tumor_dices = [dice_scores.get(l, 0) for l in [1, 2, 4] if l in dice_scores]
    metrics['mean_dice'] = np.mean(tumor_dices) if tumor_dices else 0.0
    
    # Hausdorff distance (simplified version)
    # Volume overlap
    for label in [1, 2, 4]:
        if label in dice_scores:
            pred_vol = np.sum(pred == label)
            gt_vol = np.sum(gt == label)
            if gt_vol > 0:
                vol_error = abs(pred_vol - gt_vol) / gt_vol
                metrics[f'volume_error_{label}'] = vol_error
    
    return metrics


def test_model(patient_ids, 
               data_dir='../BraTS_data',
               model_dir='./models',
               output_file='test_results.txt'):
    """
    Test trained models on multiple patients
    
    Args:
        patient_ids: List of patient IDs to test
        data_dir: Directory containing test data
        model_dir: Directory containing trained models
        output_file: File to save results
    """
    
    print("\n" + "="*80)
    print("TESTING TRAINED MODELS")
    print("="*80)
    print(f"\nTest patients: {len(patient_ids)}")
    print(f"Data dir: {data_dir}")
    print(f"Model dir: {model_dir}")
    
    # Import inference
    from inference import inference
    
    # Results storage
    all_results = []
    
    # Test each patient
    for i, patient_id in enumerate(patient_ids):
        print(f"\n{'='*80}")
        print(f"Testing patient {i+1}/{len(patient_ids)}: {patient_id}")
        print(f"{'='*80}")
        
        try:
            # Load ground truth
            _, gt = load_patient_data(data_dir, patient_id)
            
            if gt is None:
                print(f"[WARNING] No ground truth for {patient_id}, skipping...")
                continue
            
            # Run inference
            prediction = inference(
                patient_id=patient_id,
                data_dir=data_dir,
                model_dir=model_dir,
                output_dir='./test_predictions',
                use_trained=True,
                fast_mode=True  # Use fast mode for testing
            )
            
            # Calculate metrics
            metrics = calculate_metrics(prediction, gt)
            
            # Store results
            result = {
                'patient_id': patient_id,
                'metrics': metrics
            }
            all_results.append(result)
            
            # Print results
            print(f"\n[RESULTS] {patient_id}")
            print(f"{'='*60}")
            print(f"{'Label':<20} {'Dice Score':>15}")
            print(f"{'-'*60}")
            
            labels = {0: 'Background', 1: 'Necrotic', 2: 'Edema', 4: 'Enhancing'}
            for label, name in labels.items():
                if label in metrics['dice_scores']:
                    dice = metrics['dice_scores'][label]
                    print(f"{name:<20} {dice:>15.4f}")
            
            print(f"{'-'*60}")
            print(f"{'Mean Dice (Tumor)':<20} {metrics['mean_dice']:>15.4f}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"[ERROR] Failed to test {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    if all_results:
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS")
        print(f"{'='*80}")
        
        # Calculate mean metrics
        all_dice_1 = [r['metrics']['dice_scores'].get(1, 0) for r in all_results]
        all_dice_2 = [r['metrics']['dice_scores'].get(2, 0) for r in all_results]
        all_dice_4 = [r['metrics']['dice_scores'].get(4, 0) for r in all_results]
        all_mean_dice = [r['metrics']['mean_dice'] for r in all_results]
        
        print(f"\nAverage Dice Scores (n={len(all_results)}):")
        print(f"  Necrotic Core:     {np.mean(all_dice_1):.4f} ± {np.std(all_dice_1):.4f}")
        print(f"  Edema:             {np.mean(all_dice_2):.4f} ± {np.std(all_dice_2):.4f}")
        print(f"  Enhancing Tumor:   {np.mean(all_dice_4):.4f} ± {np.std(all_dice_4):.4f}")
        print(f"  Mean (Tumor):      {np.mean(all_mean_dice):.4f} ± {np.std(all_mean_dice):.4f}")
        
        # Save results to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BRAIN TUMOR SEGMENTATION - TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Patients: {len(all_results)}\n")
            f.write(f"Model Directory: {model_dir}\n\n")
            
            f.write("="*80 + "\n")
            f.write("Individual Results:\n")
            f.write("="*80 + "\n\n")
            
            for result in all_results:
                f.write(f"Patient: {result['patient_id']}\n")
                f.write(f"{'-'*60}\n")
                
                labels = {1: 'Necrotic', 2: 'Edema', 4: 'Enhancing'}
                for label, name in labels.items():
                    if label in result['metrics']['dice_scores']:
                        dice = result['metrics']['dice_scores'][label]
                        f.write(f"  {name:<15} Dice: {dice:.4f}\n")
                
                f.write(f"  {'Mean (Tumor)':<15} Dice: {result['metrics']['mean_dice']:.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("Aggregate Statistics:\n")
            f.write("="*80 + "\n\n")
            f.write(f"Necrotic Core:     {np.mean(all_dice_1):.4f} ± {np.std(all_dice_1):.4f}\n")
            f.write(f"Edema:             {np.mean(all_dice_2):.4f} ± {np.std(all_dice_2):.4f}\n")
            f.write(f"Enhancing Tumor:   {np.mean(all_dice_4):.4f} ± {np.std(all_dice_4):.4f}\n")
            f.write(f"Mean (Tumor):      {np.mean(all_mean_dice):.4f} ± {np.std(all_mean_dice):.4f}\n")
        
        print(f"\n[OK] Results saved to: {output_path}")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained segmentation models')
    parser.add_argument('--patients', type=str, nargs='+', required=True,
                        help='Patient IDs to test (space-separated)')
    parser.add_argument('--data', type=str, default='../BraTS_data',
                        help='Path to test data directory')
    parser.add_argument('--models', type=str, default='./models',
                        help='Path to trained models directory')
    parser.add_argument('--output', type=str, default='test_results.txt',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("#  BRAIN TUMOR SEGMENTATION - MODEL TESTING  #".center(80))
    print("#"*80)
    print(f"\nConfiguration:")
    print(f"  Test patients: {args.patients}")
    print(f"  Data dir: {args.data}")
    print(f"  Models dir: {args.models}")
    print(f"  Output file: {args.output}")
    
    # Run testing
    test_model(
        patient_ids=args.patients,
        data_dir=args.data,
        model_dir=args.models,
        output_file=args.output
    )
    
    print(f"\n[INFO] Testing completed!")
    print(f"[INFO] Results saved to {args.output}")
