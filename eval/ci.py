import json
import torch
import warnings
import numpy as np
from tqdm import tqdm
from eval import REG_Evaluator

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

GT_PATH = './json/ground-truth.json'
PRED_PATH = './json/submission.json'
COMPARISON_PATH = './json/comparison.json'

EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'
SPACY_MODEL = 'en_core_sci_lg'
N_BOOTSTRAP = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
with open(GT_PATH, 'r') as f:
    gt_data = json.load(f)
with open(PRED_PATH, 'r') as f:
    pred_data = json.load(f)

print(f"GT samples: {len(gt_data)}, Pred samples: {len(pred_data)}")

# Create dictionaries for fast lookup
gt_dict = {item['id']: item['report'] for item in gt_data}
pred_dict = {item['id']: item['report'] for item in pred_data}

# Find matching IDs
matched_ids = set(gt_dict.keys()) & set(pred_dict.keys())
print(f"Matched IDs: {len(matched_ids)}")

if not matched_ids:
    print("No matching IDs found!")
    exit()

# Prepare evaluation data
eval_pairs = [(gt_dict[id_], pred_dict[id_]) for id_ in matched_ids]
matched_ids_list = list(matched_ids)

print(f"Eval pairs created: {len(eval_pairs)}")

def bootstrap_scores(comparison_data, n_bootstrap=N_BOOTSTRAP):
    """Calculate bootstrap confidence intervals from precomputed scores"""
    np.random.seed(42)
    n_samples = len(comparison_data)
    
    # Extract individual scores
    rouge_scores = [item['scores']['rouge'] for item in comparison_data]
    bleu_scores = [item['scores']['bleu'] for item in comparison_data]
    key_scores = [item['scores']['key'] for item in comparison_data]
    emb_scores = [item['scores']['embedding'] for item in comparison_data]
    
    bootstrap_results = {
        'rouge': [], 'bleu': [], 'key': [], 'emb': [], 'ranking': []
    }
    
    # print(f"Bootstrap progress:")
    
    for i in range(n_bootstrap):
        # Bootstrap resample indices
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Calculate average scores for this bootstrap sample
        avg_rouge = np.mean([rouge_scores[idx] for idx in indices])
        avg_bleu = np.mean([bleu_scores[idx] for idx in indices])
        avg_key = np.mean([key_scores[idx] for idx in indices])
        avg_emb = np.mean([emb_scores[idx] for idx in indices])
        avg_ranking = avg_rouge * 0.15 + avg_bleu * 0.15 + avg_key * 0.40 + avg_emb * 0.30
        
        # Store results
        bootstrap_results['rouge'].append(avg_rouge)
        bootstrap_results['bleu'].append(avg_bleu)
        bootstrap_results['key'].append(avg_key)
        bootstrap_results['emb'].append(avg_emb)
        bootstrap_results['ranking'].append(avg_ranking)
        
        # Progress bar (100 steps)
        # if (i + 1) % (n_bootstrap // 100) == 0:
            # progress = int((i + 1) / n_bootstrap * 100)
            # print(f"\r[{'='*progress}{' '*(100-progress)}] {progress}%", end='', flush=True)
    
    print()  # New line after progress bar
    return bootstrap_results

def calculate_ci(bootstrap_results, confidence=0.95):
    """Calculate confidence intervals from bootstrap results"""
    alpha = 1 - confidence
    lower_p = (alpha / 2) * 100
    upper_p = (1 - alpha / 2) * 100
    
    ci_results = {}
    for metric, values in bootstrap_results.items():
        lower = np.percentile(values, lower_p)
        upper = np.percentile(values, upper_p)
        mean = np.mean(values)
        ci_results[metric] = {
            'mean': mean,
            'ci_lower': lower,
            'ci_upper': upper
        }
    
    return ci_results

def process_with_comparison(evaluator, eval_pairs, matched_ids, gt_dict, pred_dict):
    total_scores = {'rouge': 0, 'bleu': 0, 'key': 0, 'emb': 0}
    comparison_data = []
    
    for i in range(0, len(eval_pairs), len(eval_pairs)):
        batch = eval_pairs[i:i+len(eval_pairs)]
        batch_ids = matched_ids[i:i+len(eval_pairs)]
        
        # Get individual scores for each pair in batch
        batch_scores = []
        for j, (gt_report, pred_report) in enumerate(tqdm(batch)):
            pair_score = evaluator.evaluate_dummy([(gt_report, pred_report)])
            batch_scores.append(pair_score)
            
            # Calculate ranking score for this pair
            ranking_score = (pair_score['rouge'] * 0.15 + 
                           pair_score['bleu'] * 0.15 + 
                           pair_score['key'] * 0.40 + 
                           pair_score['emb'] * 0.30)
            
            # Add to comparison data
            comparison_data.append({
                'id': batch_ids[j],
                'ground_truth_report': gt_report,
                'predicted_report': pred_report,
                'scores': {
                    'rouge': pair_score['rouge'],
                    'bleu': pair_score['bleu'],
                    'key': pair_score['key'],
                    'embedding': pair_score['emb'],
                    'ranking_score': ranking_score
                }
            })
        
        # Accumulate total scores
        for score in batch_scores:
            for k in total_scores:
                total_scores[k] += score[k]
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Average the scores
    avg_scores = {k: v/len(eval_pairs) for k, v in total_scores.items()}
    return avg_scores, comparison_data

if eval_pairs:
    evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)
    
    # Original evaluation
    scores, comparison_data = process_with_comparison(evaluator, eval_pairs, matched_ids_list, gt_dict, pred_dict)
    avg_ranking_score = scores['rouge'] * 0.15 + scores['bleu'] * 0.15 + scores['key'] * 0.40 + scores['emb'] * 0.30
    
    # Bootstrap evaluation
    print(f"\n{'='*50}")
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'='*50}")
    
    bootstrap_results = bootstrap_scores(comparison_data, N_BOOTSTRAP)
    ci_results = calculate_ci(bootstrap_results)
    
    print(f"\n95% Confidence Intervals ({N_BOOTSTRAP} bootstraps):")
    print(f"{'Metric':<12} {'Mean':<8} {'95% CI':<20}")
    print("-" * 42)
    
    for metric in ['rouge', 'bleu', 'key', 'emb', 'ranking']:
        result = ci_results[metric]
        ci_str = f"[{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]"
        print(f"{metric.upper():<12} {result['mean']:<8.4f} {ci_str:<20}")
    
    # Save comparison data
    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison data saved to {COMPARISON_PATH}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print("No valid pairs to evaluate!")