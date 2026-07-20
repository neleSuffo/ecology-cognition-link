# %%
import os
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# %%
# 1. Setup Data Paths and Mappings
dataset_path = '../data/final_feature_df.csv'
print(f"Loading data from {dataset_path}...")
df = pd.read_csv(dataset_path)

label_dict = {
    "age_at_test": "age",
    "pct_Interacting": "P[int]",
    "pct_Not_Interacting": "P[paste('¬', int)]",
    "kcs_per_overall": "KCS[all]",
    "kcs_per_Interacting": "KCS[int]",
    "kcs_per_Not_Interacting": "KCS[paste('¬',int)]",
    "total_speech_exposure_per_overall": "Exps[all]",
    "total_speech_exposure_per_Interacting": "Exps[int]",
    "total_speech_exposure_per_Not_Interacting": "Exps[paste('¬', int)]",
    "kcds_exposure_per_overall": "KCDS[all]",
    "kcds_exposure_per_Interacting": "KCDS[int]",
    "kcds_exposure_per_Not_Interacting": "KCDS[paste('¬', int)]",
    "ohs_exposure_per_overall": "OHS[all]",
    "ohs_exposure_per_Interacting": "OHS[int]",
    "ohs_exposure_per_Not_Interacting": "OHS[paste('¬', int)]",
    "turns_per_minute_overall": "TpM[all]",
    "turns_per_minute_interacting": "TpM[int]",
    "agency_initiation_prop": "P[agency]",
    "avg_turn_duration": "bar(D)[turn]",
    "avg_block_duration": "bar(D)[block]",
    "avg_turns_per_block": "bar(N)[paste(turn, '/', block)]",
    "pct_face_visible": "P[face]",
    "pct_person_visible": "P[person]",
    "pct_face_or_person_visible": "P[paste(face, '/', person)]",
    "avg_face_proximity": "bar(d)[face]"
}

predictors_to_include = ["age_at_test", 
                        "kcs_per_Interacting", 
                        "kcds_exposure_per_Not_Interacting", 
                        "agency_initiation_prop", 
                        "avg_turn_duration", 
                        "kcs_per_overall", 
                        "total_speech_exposure_per_Interacting", 
                        "kcds_exposure_per_Interacting", 
                        "turns_per_minute_interacting", 
                        "total_speech_exposure_per_Not_Interacting", 
                        "ohs_exposure_per_Interacting", 
                        "pct_face_or_person_visible",
                        "pct_Not_Interacting"]

# %%
# 2. Transform Targets and Features

# Convert to formal proportional boundaries
df['orev_prop'] = df['orev_successes'] / df['orev_trials']
df['prag_prop'] = df['prag_successes'] / df['prag_trials']

exclude_cols = [
    "child_id", "sex","age_at_recording", "min_age_rec", 
    "orev_successes", "prag_successes", 
    "orev_trials", "prag_trials", "tango_trials", 
    "tango_error", "orev_prop", "prag_prop"
]

# Enforce clean structural separation
iv_list = [col for col in df.columns if col not in exclude_cols and col in predictors_to_include]

X = df[iv_list].copy()
Y_orev = df['orev_prop'].copy()
Y_prag = df['prag_prop'].copy()

# Ensure there are no lingering NaNs in predictors
X = X.fillna(X.mean())

print(f"Features configured: {X.shape[1]} columns ready for training.")

# %%
# 3. Define Hyperparameter Tuning Grids (5 Values Per Parameter)

# High-precision regularized tuning ranges tailored specifically for small clinical cohorts
param_grid = {
    'depth': [2, 3, 4, 5, 6],
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
    'l2_leaf_reg': [3, 5, 10, 15, 20],
    'colsample_bylevel': [0.4, 0.5, 0.6, 0.8, 1.0],
    'loss_function': ['RMSE', 'MAE']
}

# Generate every unique combination variant systematically
keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(f"Total experiment combinations to evaluate per target: {len(experiments)}")

# %%
# 4. Create the Master Single-Target LOO Function
def run_loo_grid_search(X_data, Y_target, target_name, grid_setups, max_iterations=300):
    """
    Runs a full Leave-One-Out (LOO) loop for every parameter setup,
    calculates every key performance metric, and returns the complete log history.
    """
    n_samples = len(X_data)
    performance_records = []
    
    # Iterate through every single hyperparameter setup variant
    for setup_idx, setup in enumerate(grid_setups):
        oof_predictions = np.zeros(n_samples)
        current_loss = setup['loss_function']
        
        # --- THE INNER LEAVE-ONE-OUT LOOP ---
        for i in range(n_samples):
            # Isolate the current child completely
            X_train = X_data.drop(index=i)
            Y_train = Y_target.drop(index=i)
            X_test = X_data.iloc[[i]]
            
            # Construct standard pools
            train_pool = Pool(data=X_train, label=Y_train)
            test_pool = Pool(data=X_test)
            
            # Initialize model configuration
            model = CatBoostRegressor(
                iterations=max_iterations,
                depth=setup['depth'],
                learning_rate=setup['learning_rate'],
                l2_leaf_reg=setup['l2_leaf_reg'],
                colsample_bylevel=setup['colsample_bylevel'],
                loss_function=current_loss,
                random_seed=42,
                verbose=False
            )
            
            # Train on the 84 kids for fixed iteration steps (No early stopping leakage)
            model.fit(train_pool)
            
            # Extract out-of-fold testing prediction
            oof_predictions[i] = model.predict(test_pool)[0]
            
        ## 1. Standard Distance Metrics
        r2 = r2_score(Y_target, oof_predictions)
        mae = mean_absolute_error(Y_target, oof_predictions)
        rmse = root_mean_squared_error(Y_target, oof_predictions)
        
        # 2. Ranking Metric (Spearman Rank Correlation)
        # returns (correlation, p_value); we grab just the correlation coefficient [0]
        spearman_corr, _ = spearmanr(Y_target, oof_predictions)
        
        # 3. Scale-Relative Metric (Zero-Safe Adjusted MAPE)
        # Adds an epsilon to prevent division-by-zero if a child scored exactly 0.0
        eps = 1e-4
        mape = np.mean(np.abs((Y_target - oof_predictions) / (Y_target + eps))) * 100
        
        # 4. Boundary Probability Metric (Manual Cross-Entropy)
        # Clips predictions slightly away from strict 0 and 1 to prevent log(0) errors
        preds_clipped = np.clip(oof_predictions, 1e-5, 1 - 1e-5)
        cross_entropy = -np.mean(Y_target * np.log(preds_clipped) + (1 - Y_target) * np.log(1 - preds_clipped))
        
        # --- LOG EVERYTHING INTO THE PERFORMANCE DICTIONARY ---
        performance_records.append({
            'setup_index': setup_idx,
            'depth': setup['depth'],
            'learning_rate': setup['learning_rate'],
            'l2_leaf_reg': setup['l2_leaf_reg'],
            'colsample_bylevel': setup['colsample_bylevel'],
            'loss_function': current_loss,
            'OOF_R2': r2,
            'OOF_MAE': mae,
            'OOF_RMSE': rmse,
            'OOF_Spearman_Rho': spearman_corr,
            'OOF_Adjusted_MAPE': mape,
            'OOF_CrossEntropy': cross_entropy
        })
            
        if (setup_idx + 1) % 50 == 0 or (setup_idx + 1) == len(grid_setups):
            print(f"[{target_name.upper()}] Processed {setup_idx + 1}/{len(grid_setups)} grid combinations...")
            
    # Compile into a master DataFrame sorted by R2 as a baseline order
    tuning_log_df = pd.DataFrame(performance_records).sort_values(by='OOF_R2', ascending=False)
    return tuning_log_df

# %%
# 5. Execute Optimization Run for Both Targets Separately
print("\n=== PROCESSING TUNING LOOP FOR OREV (VOCABULARY) ===")
orev_history_df = run_loo_grid_search(X, Y_orev, "orev", experiments)

os.makedirs('../outputs', exist_ok=True)
orev_history_df.to_csv('../outputs/orev_hyperparameter_tuning_log.csv', index=False)

print("\n=== PROCESSING TUNING LOOP FOR PRAG (PRAGMATICS) ===")
prag_history_df = run_loo_grid_search(X, Y_prag, "prag", experiments)
prag_history_df.to_csv('../outputs/prag_hyperparameter_tuning_log.csv', index=False)

print("\nComplete logs successfully generated and saved to the ../outputs/ folder.")

# %%
# 6. Define winning hyperparameter setup
winning_params_orev = {
    'iterations': 300,
    'depth': 2,
    'learning_rate': 0.01,
    'l2_leaf_reg': 20,
    'colsample_bylevel': 1.0,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

winning_params_prag = {
    'iterations': 300,
    'depth': 3,                 # Upgraded to depth=3 for PRAG!
    'learning_rate': 0.05,      # Upgraded to 0.05 for PRAG!
    'l2_leaf_reg': 5,           # Upgraded to 5 for PRAG!
    'colsample_bylevel': 0.8,
    'loss_function': 'MAE',     # Switched to MAE for PRAG!
    'random_seed': 42,
    'verbose': False
}

target_of_interest = 'orev_prop'  # Run this entire script once for 'orev_prop' and once for 'prag_prop'
n_samples = len(X)
n_features = X.shape[1]

# Setup storage matrices
# CatBoost SHAP output includes an extra column at the very end for the "expected value" (baseline)
shap_matrix = np.zeros((n_samples, n_features + 1))
final_predictions = np.zeros(n_samples)

# For tracking training performance across the LOO loop (optional, can be removed if not needed)
train_r2_scores = []
train_rmse_scores = []
train_mae_scores = []

print(f"Extracting leak-free SHAP values for {target_of_interest} using winning architecture...")

# %%
# 7. Execute the Final Out-of-Fold SHAP Loop
for i in range(n_samples):
    # Partition data
    X_train = X.drop(index=i)
    Y_train = Y_orev.drop(index=i) if target_of_interest == 'orev_prop' else Y_prag.drop(index=i)
    
    X_test = X.iloc[[i]]
    
    # Pools
    train_pool = Pool(data=X_train, label=Y_train)
    test_pool = Pool(data=X_test)
    
    # Train the optimal model structure
    final_model = CatBoostRegressor(**winning_params_orev if target_of_interest == 'orev_prop' else winning_params_prag)
    final_model.fit(train_pool)
    
    # 1. Save the completely blind out-of-fold prediction
    final_predictions[i] = final_model.predict(test_pool)[0]
    
    # 2. Extract the true independent SHAP values for this specific child
    # type="ShapValues" returns an array of shape (1, n_features + 1)
    child_shap = final_model.get_feature_importance(test_pool, type="ShapValues")
    shap_matrix[i, :] = child_shap[0, :]
    
    # Track training performance on the training set for sanity check (not used in final analysis)
    preds_on_train_set = final_model.predict(train_pool)

    fold_train_r2 = r2_score(Y_train, preds_on_train_set)
    fold_train_rmse = root_mean_squared_error(Y_train, preds_on_train_set)
    fold_train_mae = mean_absolute_error(Y_train, preds_on_train_set)
    
    train_mae_scores.append(fold_train_mae)
    train_r2_scores.append(fold_train_r2)
    train_rmse_scores.append(fold_train_rmse)
    
# 3. Collapse the 85 folds into a single global average for your paper
mean_train_r2 = np.mean(train_r2_scores)
mean_train_rmse = np.mean(train_rmse_scores)
mean_train_mae = np.mean(train_mae_scores)
print("\n==========================================")
print(f"FINAL TRAINING METRICS FOR: {target_of_interest}")
print(f"Mean Training R2:   {mean_train_r2:.4f}")
print(f"Mean Training RMSE: {mean_train_rmse:.4f}")
print(f"Mean Training MAE:  {mean_train_mae:.4f}")
print("==========================================\n")

# %%
# 8. Restructure and Map into a Clean Long-Format DataFrame
shap_records = []

for i in range(n_samples):
    child_id = df.index[i]  # Re-link your row indexes or child_id strings here
    actual_val = Y_orev.iloc[i] if target_of_interest == 'orev_prop' else Y_prag.iloc[i]
    pred_val = final_predictions[i]
    expected_value = shap_matrix[i, -1] # The final column is the base intercept
    
    for f_idx, feature_name in enumerate(X.columns):
        raw_shap = shap_matrix[i, f_idx]
        feature_val = X.iloc[i, f_idx]
        
        # Pull the beautiful R plotmath label from your dictionary safely
        paper_label = label_dict.get(feature_name, feature_name)
        
        shap_records.append({
            'child_idx': i,
            'target': target_of_interest,
            'feature_raw': feature_name,
            'feature_paper_label': paper_label,
            'feature_value': feature_val,
            'shap_value': raw_shap,
            'baseline_value': expected_value,
            'actual_value': actual_val,
            'predicted_value': pred_val
        })

final_shap_df = pd.DataFrame(shap_records)

# %%
# 9. Save the Final Analytical Ground Truth
os.makedirs('../outputs', exist_ok=True)
output_name = f"../outputs/{target_of_interest}_final_oof_shap_values.csv"
final_shap_df.to_csv(output_name, index=False)
print(f"SHAP values saved to: {output_name}")

# %%
