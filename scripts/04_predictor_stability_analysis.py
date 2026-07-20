import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

results_list = []
all_results = []
all_noise_rs = []
all_reliability_rs = []

def generate_unified_predictor_name(row):
    metric = str(row['metric'])
    state = str(row['state'])
    ctype = str(row['type'])

    # 3. Handle CDS (exposure_percent) combinations
    if metric == 'cds':
        if ctype == 'KCDS_ONLY':
            return f"kcds_exposure_per_{state}"
        elif ctype == 'OHS_ONLY':
            return f"ohs_exposure_per_{state}"
        elif ctype == 'TOTAL':
            return f"total_speech_exposure_per_{state}"

    # 4. Handle KCS combinations
    elif metric == 'kcs':
        return f"kcs_per_{state}"

    # 1. Handle the "Simple" overall metrics
    elif state == 'overall' and ctype == 'overall':
        return metric

    return f"{metric}_{state}_{ctype}"

def split_segments_with_metadata(df, sec_per_id_df, output_folder):
    """
    Splits segments crossing fold boundaries into multiple rows,
    preserving original columns and adding a 'fold' column.
    """
    df["child_id"] = df['child_id'].astype(str)
    sec_per_id_df['child_id'] = sec_per_id_df['child_id'].astype(str)
    # 1. Ensure we have the master duration
    df = df.merge(sec_per_id_df[['child_id', 'seconds_per_id']], on='child_id', how='left')
    
    # 2. Calculate the "Offset" for each video
    # We need to know how long Video 1 was to know where Video 2 starts on the global timeline
    video_durations = df.groupby(['child_id', 'video_id'])['duration_sec'].sum().reset_index()
    video_durations = video_durations.sort_values(['child_id', 'video_id'])
    
    # Calculate the cumulative sum of previous videos
    video_durations['video_offset'] = video_durations.groupby('child_id')['duration_sec'].transform(
        lambda x: x.cumsum().shift(1, fill_value=0)
    )
    # Merge offsets back to the main dataframe
    df = df.merge(video_durations[['child_id', 'video_id', 'video_offset']], on=['child_id', 'video_id'], how='left')
    
    # 3. Create Global Times
    df['global_start'] = df['start_time_sec'] + df['video_offset']
    df['global_end'] = df['end_time_sec'] + df['video_offset']
    
    split_rows = []
    
    for _, row in df.iterrows():
        total_dur = row['seconds_per_id']
        fold_dur = total_dur * 0.2
        
        for fold_num in range(1, 6):
            fold_start = (fold_num - 1) * fold_dur
            fold_end = fold_num * fold_dur
            
            # Use global times to check overlap with the fold window
            overlap_start = max(row['global_start'], fold_start)
            overlap_end = min(row['global_end'], fold_end)
            
            if overlap_start < overlap_end:
                new_row = row.copy()
                new_row['fold'] = fold_num
                
                # 1. Define raw split boundaries (relative to video start)
                raw_start_sec = overlap_start - row['video_offset']
                raw_end_sec = overlap_end - row['video_offset']
                
                # 2. Convert to frames and round to the nearest 10th
                fps = 30
                # Using round() then snapping to 10 ensures we don't always drift downward
                f_start = int(round(raw_start_sec * fps) // 10) * 10
                f_end = int(round(raw_end_sec * fps) // 10) * 10
                
                # 3. SET THE FRAMES (Ground Truth)
                new_row['segment_start'] = f_start
                new_row['segment_end'] = f_end
                
                # 4. INFER SECONDS FROM FRAMES (To ensure alignment)
                # This ensures duration_sec * 30 will always = segment_end - segment_start
                new_row['start_time_sec'] = f_start / fps
                new_row['end_time_sec'] = f_end / fps
                new_row['duration_sec'] = (f_end - f_start) / fps

                split_rows.append(new_row)
                
    segments_fold_df = pd.DataFrame(split_rows)
    # save segments_fold_df to a new CSV for debugging
    segments_fold_df.to_csv(output_folder / "interaction_segments_fold.csv", index=False)

def extract_child_id(video_name: str) -> str:
    """
    Extracts the 6-digit child ID from a video name string.
    Example: 'id123456_video.mp4' -> '123456'
    
    Parameters:
    ----------
    video_name : str
        The name of the video file.
        
    Returns:
    -------
    str or None
        The extracted child ID, or None if not found.
    """
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def calculate_full_stability_metrics(file_path, metric_col, output_folder, label=""):
    """
    Calculates stability and reliability metrics per social state (if available)
    and generates separate plots for each.
    """
    # 1. Load Master Sheet
    master_path = "/home/nele_pauline_suffo/ProcessedData/quantex_data_sheet_MASTER.csv"
    sec_per_id_df = pd.read_csv(master_path, sep=";")
    sec_per_id_df['seconds_per_id'] = pd.to_numeric(
        sec_per_id_df['seconds_per_id'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )
    sec_per_id_df['child_id'] = sec_per_id_df['id'].astype(str)
    sec_per_id_df = sec_per_id_df[['child_id', 'seconds_per_id']].drop_duplicates()
    
    # 2. Load analysis results
    df = pd.read_csv(file_path)
          
    # --- SPECIAL HANDLING FOR SOCIAL STATE PROPORTIONS ---
    if label in ["pct_Interacting", "pct_Not_Interacting"]:
        target_state = label.replace("pct_", "")
        
        # 2. Sum the interaction durations
        fold_sums = df[df['interaction_type'] == target_state].groupby(['child_id', 'fold'])['duration_sec'].sum().reset_index()
                
        # 3. Create a complete grid of all 5 folds for every child
        all_cids = df['child_id'].unique()
        grid = pd.DataFrame([(c, f) for c in all_cids for f in range(1, 6)], columns=['child_id', 'fold'])
            
        # 4. Merge results into the grid and apply your 0.2 logic
        df_final = grid.merge(fold_sums, on=['child_id', 'fold'], how='left').fillna(0)

        duration_map = df.set_index('child_id')['seconds_per_id'].to_dict()
        df_final['seconds_per_id'] = df_final['child_id'].map(duration_map)
        
       # Use the master duration divided by 5 (0.2)
        df_final[label] = df_final['duration_sec'] / (df_final['seconds_per_id'] * 0.2 + 1e-6)
        
        df = df_final
        print(df.head())
        metric_col = label
        social_col = None 
        exposure_col = None

    # --- SPECIAL HANDLING FOR AGENCY ---
    elif label == "agency_initiation_prop":
        # Calculate total turns (blocks) per row
        df['total_successful_blocks'] = df['successful_initiations'] + df['successful_responses']
        
        # Filter out zero-activity segments
        df = df[df['total_successful_blocks'] > 0].copy()
        
        # To make this work with the rest of the script's loop, we need a single metric_col.
        # We aggregate by child and fold to get the weighted proportion for that fold.
        df_agged = df.groupby(['child_id', 'fold']).agg({
            'successful_initiations': 'sum',
            'total_successful_blocks': 'sum'
        }).reset_index()
        
        df_agged['agency_initiation_prop'] = df_agged['successful_initiations'] / df_agged['total_successful_blocks']
        
        # Replace df and set metric_col to our new calculated column
        df = df_agged
        metric_col = 'agency_initiation_prop'
          
    # Detect social state column
    social_col = 'interaction_type' if 'interaction_type' in df.columns else None
    exposure_col = 'exposure_type' if 'exposure_type' in df.columns else None
    if metric_col in ["has_face", "has_person", "person_or_face_present"]:
        df["person_or_face_present"] = df[["has_face", "has_person"]].any(axis=1).astype(int)
        df['child_id'] = df['video_name'].apply(extract_child_id)
        social_col = None

    social_states = df[social_col].unique() if social_col else [None]
    exposure_types = df[exposure_col].unique() if exposure_col else [None]
        
    # Initialize list to hold results for this file
    local_results = []
    child_level_accumulator = []
    
    for state in social_states:
        for exp_type in exposure_types:
            # Filter Data
            temp_df = df.copy()
            label_suffix = ""
            if social_col:
                temp_df = temp_df[temp_df[social_col] == state]
                label_suffix += f"_{state.lower()}"
            if exposure_col:
                temp_df = temp_df[temp_df[exposure_col] == exp_type]
                label_suffix += f"_{exp_type.lower()}"
        
            if temp_df.empty: continue
            
            print(f"Processing: Metric={label} | State={state} | Type={exp_type}")
            
            # --- PART A: CALCULATE PER-CHILD RELIABILITY ---
            child_ids = temp_df['child_id'].unique()
            child_reliability = []
            
            for cid in child_ids:
                c_data = temp_df[temp_df['child_id'] == cid].copy()

                # Aggregate segments into folds
                c_data = c_data.groupby('fold')[metric_col].mean().reset_index()
                
                loo_agreements = []
                for i in range(1, 6):
                    # Current Fold Value
                    fold_row = c_data[c_data['fold'] == i]
                    val_fold = fold_row[metric_col].values[0] if not fold_row.empty else 0.0
                    #if fold_row.empty:
                    #    print(f"⚠️ Missing Fold {i} for {cid} ({state}/{exp_type})")
                        
                    # Get mean of others (imputing 0 for missing)
                    others_vals = []
                    for j in range(1, 6):
                        if j == i: continue
                        f_val = c_data[c_data['fold'] == j][metric_col].values
                        others_vals.append(f_val[0] if len(f_val) > 0 else 0.0)
                    
                    mean_others = np.mean(others_vals)
                    
                    # Agreement Formula: 1 - (|obs - ref| / ref)
                    # If both are 0, they agree perfectly (1.0)
                    if mean_others == 0 and val_fold == 0:
                            agreement = 1.0
                    else:
                        denom = mean_others if mean_others != 0 else (val_fold if val_fold != 0 else 1)
                        agreement = 1 - (abs(val_fold - mean_others) / (denom + 1e-6))
                    loo_agreements.append(max(0, agreement))
                                
                child_reliability.append({'child_id': cid, 'avg_reliability': np.mean(loo_agreements)})
            
            rel_df = pd.DataFrame(child_reliability)
            
            # --- PART B: VARIANCE (CV) ---
            rel_df = pd.DataFrame(child_reliability)
            rel_df['child_id'] = rel_df['child_id'].astype(str)
            child_stats = temp_df.groupby('child_id').agg({metric_col: ['std', 'mean']}).reset_index()
            child_stats.columns = ['child_id', 'sd', 'mean']
            child_stats['cv'] = (child_stats['sd'] / child_stats['mean']).fillna(0)
            child_stats['child_id'] = child_stats['child_id'].astype(str)
            
            # Merge and Prep Plot Data
            summary = child_stats.merge(rel_df, on='child_id').merge(sec_per_id_df, on='child_id')
            summary['minutes'] = summary['seconds_per_id'] / 60

            # --- PART C: GLOBAL STABILITY ---
            loo_corrs = []
            for i in range(1, 6):
                # Aggregate fold means across all children for global r calculation
                v_fold = temp_df[temp_df['fold'] == i].groupby('child_id')[metric_col].mean().reset_index().rename(columns={metric_col: 'val'})
                d_folds = temp_df[temp_df['fold'] != i].groupby('child_id')[metric_col].mean().reset_index().rename(columns={metric_col: 'disc'})
                iter_df = pd.merge(v_fold, d_folds, on='child_id').dropna()
                if len(iter_df) > 1:
                    r_iter, _ = pearsonr(iter_df['val'], iter_df['disc'])
                    loo_corrs.append(r_iter)
            avg_global_r = np.mean(loo_corrs) if loo_corrs else 0

            # --- PART D: PLOTTING ---
            # 1. Noise vs Duration
            r_bias, p_bias = pearsonr(summary['minutes'], summary['cv'])
            plt.figure(figsize=(10, 6))
            sns.regplot(data=summary, x='minutes', y='cv', scatter_kws={'alpha':0.4}, line_kws={'color':'red'})
            plt.title(f"Noise: {label}{label_suffix}\nGlobal Stability: r={avg_global_r:.3f}")
            plt.xlabel("Total Recording Duration (Minutes)")
            plt.ylabel("Measurement Noise (Coefficient of Variation)")
            plt.text(0.05, 0.9, f"Duration-Noise r: {r_bias:.3f}\np: {p_bias:.3f}", transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            plt.savefig(output_folder / f"cv_{label}{label_suffix}.png", dpi=300)
            plt.close()

            # 2. Reliability vs Duration
            r_rel, p_rel = pearsonr(summary['minutes'], summary['avg_reliability'])
            plt.figure(figsize=(10, 6))
            sns.regplot(data=summary, x='minutes', y='avg_reliability', scatter_kws={'alpha':0.4}, line_kws={'color':'blue'})
            plt.title(f"Reliability: {label}{label_suffix}\nGlobal Stability: r={avg_global_r:.3f}")
            plt.xlabel("Total Recording Duration (Minutes)")
            plt.ylabel("Internal Consistency (Mean LOO Agreement)")
            plt.ylim(0, 1.1)
            plt.text(0.05, 0.05, f"Duration-Reliability r: {r_rel:.3f}\np: {p_rel:.3f}", transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
            plt.savefig(output_folder / f"stability_{label}{label_suffix}.png", dpi=300)
            plt.close()
            
            print(f"✅ Completed: {label}{label_suffix} | Global r: {avg_global_r:.3f} | Duration-Noise r: {r_bias:.3f} | Duration-Reliability r: {r_rel:.3f}")
        
            result_row = {
                'metric': label,
                'state': state if state else 'overall',
                'type': exp_type if exp_type else 'overall',
                'global_stability_r': avg_global_r,
                'duration_noise_r': r_bias,
                'duration_noise_p': p_bias,
                'duration_rel_r': r_rel,
                'duration_rel_p': p_rel,
                'avg_child_reliability': summary['avg_reliability'].mean(),
                'avg_child_noise_cv': summary['cv'].mean()
            }
            # Generate predictor name
            result_row['predictor'] = generate_unified_predictor_name(result_row)
            summary['predictor'] = result_row['predictor']
            child_level_accumulator.append(summary[['predictor', 'minutes', 'cv', 'avg_reliability']])
            local_results.append(result_row)
            
    return local_results, pd.concat(child_level_accumulator)

files_to_process = [
    ("02_kcs_summary.csv", "speech_activity_percent", "kcs"),
    ("02a_global_kcs_summary.csv", "speech_activity_percent", "kcs"),
    ("03_cds_summary.csv", "exposure_percent", "cds"),
    ("03a_global_cds_summary.csv", "exposure_percent", "cds"),
    ("04_turn_taking_summary.csv", "turns_per_minute", "turns_per_minute_interacting"),
    ("04_turn_taking_summary.csv", "block_duration_sec", "avg_block_duration"),
    ("04_turn_taking_summary.csv", "agency_initiation_prop", "agency_initiation_prop"),
    ("04a_global_turn_taking_summary.csv", "turns_per_minute", "turns_per_minute_overall"),
    ("04b_turn_durations_summary.csv", "vocalization_duration_sec", "avg_turn_duration"),
    ("05_interaction_composition.csv", "has_face", "pct_face_visible"),
    ("05_interaction_composition.csv", "has_person", "pct_person_visible"),
    ("05_interaction_composition.csv", "person_or_face_present", "pct_face_or_person_visible"),
    ("interaction_segments_fold.csv", "pct_Not_Interacting", "pct_Not_Interacting"),
    ("interaction_segments_fold.csv", "pct_Interacting", "pct_Interacting")
]
      
def main_process_and_save(output_folder, files_to_process):
    global_results = []
    all_child_data = []

    for filename, col, label_suffix in files_to_process:
        f_path = output_folder.parent / filename
        if f_path.exists():
            local_rs, child_df = calculate_full_stability_metrics(f_path, col, output_folder, label_suffix)
            global_results.extend(local_rs)
            all_child_data.append(child_df)
        else:
            print(f"⚠️ File not found: {f_path}")

    if not global_results:
        print("❌ No results generated. Check if files exist and contain data.")
        return

    # Save the Global Stability Summary
    summary_df = pd.DataFrame(global_results)
    output_file = output_folder / "stability_results_raw.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"✅ Data stored. Total results: {len(summary_df)}")
    print(f"   Saved to: {output_file}")
    
    # Save the Child-Level Data for the unified plots
    child_level_df = pd.concat(all_child_data)
    child_level_df.to_csv(output_folder / "child_level_stability_data.csv", index=False)
      
def generate_unified_smooth_plots(csv_path, output_folder, predictor_list):
    """Generate unified trend plots for selected predictors."""
    if not csv_path.exists():
        print(f"⚠️ Cannot create plots: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    sns.set_style("whitegrid")

    # 1. Unified Noise Plot (CV)
    plt.figure(figsize=(12, 8))
    for pred in predictor_list:
        sub = df[df['predictor'] == pred]
        if not sub.empty:
            # We use ci=None to keep it clean, or 95 for shaded error bands
            sns.regplot(data=sub, x='minutes', y='cv', scatter=False, label=pred, ci=95)
    
    plt.title("Growth of Measurement Precision vs. Session Duration")
    plt.xlabel("Total Recording Duration (Minutes)")
    plt.ylabel("Measurement Noise (CV)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / "unified_noise_trends.png", dpi=300)
    plt.close()

    # 2. Unified Reliability Plot (Agreement)
    plt.figure(figsize=(12, 8))
    for pred in predictor_list:
        sub = df[df['predictor'] == pred]
        if not sub.empty:
            sns.regplot(data=sub, x='minutes', y='avg_reliability', scatter=False, label=pred, ci=95)
    
    plt.title("Internal Consistency Trends Across Predictors")
    plt.xlabel("Total Recording Duration (Minutes)")
    plt.ylabel("Mean LOO Agreement")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_folder / "unified_reliability_trends.png", dpi=300)
    plt.close()
    
def create_global_summary(stability_file_path, output_folder, predictor_list):
    """Create final summary filtered to target predictors."""
    if not stability_file_path.exists():
        print(f"⚠️ Cannot create summary: {stability_file_path} not found")
        return
    
    df = pd.read_csv(stability_file_path)
    
    # Filter the dataframe to only keep the target predictors
    filtered_df = df[df['predictor'].isin(predictor_list)].copy()
    
    if filtered_df.empty:
        print(f"⚠️ No matching predictors found. Available: {df['predictor'].unique().tolist()}")
        return
    
    # Calculate the Global Average Row
    numeric_cols = [
        'global_stability_r', 'duration_noise_r', 'duration_rel_r', 
        'avg_child_reliability', 'avg_child_noise_cv'
    ]
    
    avg_values = filtered_df[numeric_cols].mean()
    
    # Create the summary row
    avg_row = {col: avg_values[col] for col in numeric_cols}
    avg_row['predictor'] = 'GLOBAL_AVERAGE'
    avg_row['metric'] = 'ALL'
    avg_row['state'] = 'COMBINED'
    avg_row['type'] = 'MEAN'
    
    # Append average row
    final_df = pd.concat([filtered_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Organize columns and save
    cols = ['predictor'] + [c for c in final_df.columns if c != 'predictor']
    final_df = final_df[cols]
    
    output_path = output_folder / "final_predictor_stability_report.csv"
    final_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("       FINAL STABILITY REPORT SUMMARY")
    print("="*50)
    print(f"Total Predictors: {len(filtered_df)}")
    print(f"Mean Stability (r):          {avg_row['global_stability_r']:.3f}")
    print(f"Mean Noise Bias (r):         {avg_row['duration_noise_r']:.3f}")
    print(f"Mean Reliability Bias (r):   {avg_row['duration_rel_r']:.3f}")
    print("="*50)
    print(f"\n✅ Report saved to: {output_path}")
        

output_folder = Path("/home/nele_pauline_suffo/outputs/quantex_analysis/analysis_20260417_143316_binary_final_folds/stability_analysis")
predictor_list = [
    "turns_per_minute_interacting", "agency_initiation_prop", "avg_turn_duration",
    "pct_face_or_person_visible",
    "pct_Not_Interacting",
    "ohs_exposure_per_Interacting",
    "kcs_per_overall", "kcs_per_Interacting",
    "kcds_exposure_per_Interacting", "kcds_exposure_per_Not_Interacting",
    "total_speech_exposure_per_Not_Interacting", "total_speech_exposure_per_Interacting"]
    
if __name__ == "__main__":
    #split_segments_with_metadata(df=segments_df, sec_per_id_df=sec_per_id_df, output_folder=output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing files from: {output_folder.parent}")
    print(f"Results will be saved to: {output_folder}\n")

    # --- PHASE 1: PROCESS RAW DATA ---
    main_process_and_save(output_folder, files_to_process)
    
    # --- PHASE 2: GENERATE THE TABLE ---
    # Filter results to 12 target predictors and add global average
    create_global_summary(
        output_folder / "stability_results_raw.csv", 
        output_folder, 
        predictor_list
    )
    
    # --- PHASE 3: UNIFIED PLOTS ---
    generate_unified_smooth_plots(
        output_folder / "child_level_stability_data.csv", 
        output_folder, 
        predictor_list
    )
    
    print("\n✅ Script execution complete!")