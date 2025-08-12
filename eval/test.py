import json
import torch
import warnings
from eval import REG_Evaluator

warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

GT_PATH = './json/ground-truth.json'
PRED_PATH = './json/submission.json'
COMPARISON_PATH = './json/comparison.json'

EMBEDDING_MODEL = 'dmis-lab/biobert-v1.1'
SPACY_MODEL = 'en_core_sci_lg'
BATCH_SIZE = 32

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

def process_with_comparison(evaluator, eval_pairs, matched_ids, gt_dict, pred_dict, batch_size=BATCH_SIZE):
    total_scores = {'rouge': 0, 'bleu': 0, 'key': 0, 'emb': 0}
    comparison_data = []
    total_batches = (len(eval_pairs) + batch_size - 1) // batch_size
    
    for i in range(0, len(eval_pairs), batch_size):
        batch = eval_pairs[i:i+batch_size]
        batch_ids = matched_ids[i:i+batch_size]
        
        # Get individual scores for each pair in batch
        batch_scores = []
        for j, (gt_report, pred_report) in enumerate(batch):
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
        
        print(f"Processed batch {(i//batch_size)+1}/{total_batches}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Average the scores
    avg_scores = {k: v/len(eval_pairs) for k, v in total_scores.items()}
    return avg_scores, comparison_data

if eval_pairs:
    evaluator = REG_Evaluator(embedding_model=EMBEDDING_MODEL, spacy_model=SPACY_MODEL)
    
    scores, comparison_data = process_with_comparison(evaluator, eval_pairs, matched_ids_list, gt_dict, pred_dict, BATCH_SIZE)
    
    avg_ranking_score = scores['rouge'] * 0.15 + scores['bleu'] * 0.15 + scores['key'] * 0.40 + scores['emb'] * 0.30
    
    print(f"\nAverage Scores:")
    print(f"  ROUGE: {scores['rouge']:.4f}")
    print(f"  BLEU: {scores['bleu']:.4f}")
    print(f"  Key: {scores['key']:.4f}")
    print(f"  Embedding: {scores['emb']:.4f}")
    print(f"  Average Ranking Score: {avg_ranking_score:.4f}")
    
    # Save comparison data
    with open(COMPARISON_PATH, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison data saved to {COMPARISON_PATH}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    print("No valid pairs to evaluate!")