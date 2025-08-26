import torch
import numpy as np
import chromadb
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import json
from pathlib import Path
import threading
import gc
from tqdm import tqdm
import re
import random
warnings.filterwarnings("ignore")

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def clear_cpu_memory():
    gc.collect()

def full_cleanup():
    clear_gpu_memory()
    clear_cpu_memory()

def load_prompt(prompt_path="prompt.txt"):
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt from {prompt_path}: {e}")
        return ""

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

def load_embeddings_dict(npz_dir):
    embeddings_dict = {}
    npz_dir = Path(npz_dir)
    
    for npz_file in npz_dir.glob("*.npz"):
        filename = npz_file.stem
        try:
            data = np.load(npz_file)
            embedding_key = list(data.keys())[0]
            embedding = data[embedding_key].astype(np.float32)
            embeddings_dict[filename] = embedding
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
    
    print(f"Loaded {len(embeddings_dict)} embeddings from {npz_dir}")
    return embeddings_dict

def load_test_filenames(txt_path):
    with open(txt_path, 'r') as f:
        filenames = set(line.strip() for line in f if line.strip())
    return filenames

class MultiGPUMedicalInference:
    def __init__(self, db_path, model_path, embeddings_dict, num_gpus=None, need_projection=True, prompt_path="prompt.txt"):
        full_cleanup()
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.models = {}
        self.tokenizers = {}
        self.embeddings_dict = embeddings_dict
        self.lock = threading.Lock()
        self.prompt = load_prompt(prompt_path)
        
        print(f"Initializing with {self.num_gpus} GPUs")
        print(f"Loaded prompt from {prompt_path}")
        
        # Load database
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("medical_images")
        print(f"Loaded database with {self.collection.count()} medical cases")
        
        # Load projection model if needed (768->128)
        if need_projection:
            self.projection_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Get input dimension from first embedding
            first_embedding = next(iter(embeddings_dict.values()))
            input_dim = first_embedding.shape[0]
            
            self.projection_model = ProjectionHead(input_dim, 128).to(self.projection_device)
            checkpoint = torch.load(model_path, map_location=self.projection_device)
            self.projection_model.load_state_dict(checkpoint['model'])
            self.projection_model.eval()
            print(f"Loaded projection model ({input_dim}->128) on {self.projection_device}")
        else:
            self.projection_model = None
            print("Using pre-projected 128D embeddings")

        # Load models on each GPU
        for gpu_id in tqdm(range(self.num_gpus), desc="Loading models"):
            self._load_model_on_gpu(gpu_id)

    def _load_model_on_gpu(self, gpu_id):
        device = f'cuda:{gpu_id}'
        
        with torch.cuda.device(gpu_id):
            torch.cuda.empty_cache()
        
        tokenizer = AutoTokenizer.from_pretrained("../Qwen3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "../Qwen3-8B",
            device_map={"": device},
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        self.tokenizers[gpu_id] = tokenizer
        self.models[gpu_id] = model
        clear_cpu_memory()
    
    def cleanup_models(self):
        for gpu_id in list(self.models.keys()):
            if gpu_id in self.models:
                del self.models[gpu_id]
            if gpu_id in self.tokenizers:
                del self.tokenizers[gpu_id]
        
        self.models.clear()
        self.tokenizers.clear()
        
        if hasattr(self, 'projection_model') and self.projection_model:
            del self.projection_model
        
        if hasattr(self, 'client'):
            del self.client
            del self.collection
        
        full_cleanup()
    
    def project_embedding(self, raw_embedding):
        if not self.projection_model:
            return raw_embedding
        
        with torch.no_grad():
            embedding_tensor = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0).to(self.projection_device)
            projected = self.projection_model(embedding_tensor)
            return projected.cpu().numpy().flatten()
    
    def get_all_similarities(self, query_embedding):
        total_count = self.collection.count()
        
        with self.lock:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=total_count,
                include=["distances"]
            )
        
        distances = np.array(results['distances'][0])
        similarities = 1 - distances
        return similarities
    
    def get_dynamic_k(self, query_embedding, min_k=1, max_k=10, start_check=1):
        similarities = self.get_all_similarities(query_embedding)
        similarities = np.sort(similarities)[::-1]
        
        threshold = 0.0001
        
        for i in range(start_check-1, min(len(similarities)-1, max_k-1)):
            gap = similarities[i] - similarities[i+1]
            if gap > threshold:
                return i + 1
        
        return max_k
    
    def search_similar_cases(self, query_embedding, k=3):
        k = int(k)
        with self.lock:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "distances"]
            )
        return results['documents'][0], results['distances'][0]
    
    def generate_report(self, similar_reports, distances, gpu_id):
        context = ""
        for i, (report, dist) in enumerate(zip(similar_reports, distances)):
            similarity_score = 1 - dist
            context += "-----------------------\n"
            context += f"[Similar Report {i}] Score: {similarity_score:.4f}\n"
            context += f"{report}\n"
        
        prompt = f"{self.prompt}\n\n{context}"
        
        device = f'cuda:{gpu_id}'
        model = self.models[gpu_id]
        tokenizer = self.tokenizers[gpu_id]

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=8192)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try: 
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        del model_inputs, output_ids
        torch.cuda.empty_cache()
        
        return content
    
    def process_single_file(self, filename, gpu_id, skip_projection=False):
        try:
            raw_embedding = self.embeddings_dict[filename]
            
            if skip_projection:
                projected_embedding = raw_embedding
            else:
                projected_embedding = self.project_embedding(raw_embedding)
            
            k = self.get_dynamic_k(projected_embedding)
            similar_reports, distances = self.search_similar_cases(projected_embedding, k=k)
            for i, similar_report in enumerate(similar_reports):
                print('-' * 115)
                print(f"{similar_report}")
            
            if k == 1 or k == 2: 
                report = similar_reports[0]
            else:
                report = self.generate_report(similar_reports, distances, gpu_id)

            print('-' * 115)
            print(' ' * 50 + 'FINAL REPORT')
            print('-' * 115)
            print(report)
            print('-' * 115)
            
            return {"id": filename + '.tiff', "report": report, "k_used": k}
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return {"id": filename, "report": "Error: Unable to generate report", "k_used": 0}
    
    def process_batch_parallel(self, filenames, max_workers=None, skip_projection=False):
        if max_workers is None:
            max_workers = self.num_gpus
        
        results = []
        
        print('-' * 115)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filename = {}
            for i, filename in enumerate(filenames):
                gpu_id = i % self.num_gpus
                future = executor.submit(self.process_single_file, filename, gpu_id, skip_projection)
                future_to_filename[future] = filename
            
            with tqdm(total=len(filenames), desc="Processing files") as pbar:
                for future in as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.set_postfix({"Current": f"{filename} (k={result.get('k_used', 'N/A')})"})
                    except Exception as e:
                        print(f"Error with {filename}: {e}")
                        results.append({"id": filename, "report": "Error: Processing failed", "k_used": 0})
                    pbar.update(1)
        
        return results

def main():
    full_cleanup()

    # Train/Test Configuration
    MODE = "train"  # Change to "train" or "test"
    RANDOM = False  # Add random parameter
    test_npz_dir = "../../vectors/gigapath_phase2"
    RANDOM_SEED = 4211  # Change this seed as needed

    test_txt = "eval/test_list.txt"

    db_path = "./medical_db"
    model_path = "model/semantic_model.pth"
    prompt_path = "prompt.txt"
    output_filename = f"eval/json/submission.json"
    
    inference = None
    try:
        print(f"Running in {MODE} mode with {torch.cuda.device_count()} GPUs")
        
        if MODE == "train":
            if RANDOM:
                # Set random seed
                random.seed(RANDOM_SEED)
                print(f"Using random seed: {RANDOM_SEED}")
                
                # Get all IDs from ChromaDB
                client = chromadb.PersistentClient(path=db_path)
                collection = client.get_collection("medical_images")
                all_data = collection.get(include=["embeddings"])
                all_ids = all_data['ids']
                all_embeddings = all_data['embeddings']
                print(f"Found {len(all_ids)} entries in ChromaDB")
                
                # Filter IDs that contain 'original'
                original_ids = [id_str for id_str in all_ids if 'original' in id_str]
                print(f"Found {len(original_ids)} entries with 'original'")
                
                # Randomly sample 100 from these IDs
                sampled_ids_raw = random.sample(original_ids, min(100, len(original_ids)))
                sampled_ids = [id_str.replace('_original', '') for id_str in sampled_ids_raw]
                print(f"Randomly sampled {len(sampled_ids)} IDs with 'original'")
                
                # Create a temporary ChromaDB copy without the sampled entries
                import tempfile
                import shutil
                temp_db_path = tempfile.mkdtemp(prefix="temp_medical_db_")
                shutil.copytree(db_path, temp_db_path, dirs_exist_ok=True)
                print(f"Created temporary ChromaDB copy at: {temp_db_path}")
                
                # Remove entries from the temporary ChromaDB copy
                temp_client = chromadb.PersistentClient(path=temp_db_path)
                temp_collection = temp_client.get_collection("medical_images")
                
                # Find all IDs that contain any of the sampled base filenames
                ids_to_remove = []
                for id_str in all_ids:
                    for sampled_base in sampled_ids:
                        if sampled_base in id_str:
                            ids_to_remove.append(id_str)
                            break
                
                print(f"Removing {len(ids_to_remove)} entries from temporary ChromaDB to avoid data leakage")
                print(f"Count before deletion: {temp_collection.count()}")
                print(f"Sample of IDs to delete: {ids_to_remove[:5]}")
                
                if ids_to_remove:
                    temp_collection.delete(ids=ids_to_remove)
                    print(f"Successfully removed {len(ids_to_remove)} entries from temporary copy")
                
                print(f"Temporary ChromaDB now has {temp_collection.count()} entries")
                
                del temp_client, temp_collection
                del client, collection
                
                # Update db_path to use the temporary copy
                db_path = temp_db_path
                
                # Get embeddings for sampled IDs from ChromaDB data
                embeddings_dict = {}
                
                for i, id_str in enumerate(all_ids):
                    if id_str in sampled_ids_raw:  # Use original sampled IDs with _original
                        filename = id_str.replace('_original', '').replace('.tiff', '')
                        embeddings_dict[filename] = np.array(all_embeddings[i], dtype=np.float32)
                
                filenames = list(embeddings_dict.keys())
                print(f"Loaded {len(embeddings_dict)} embeddings from ChromaDB")
                need_projection = False  # Already projected in ChromaDB
                skip_projection = True
                
            else:
                # Use test_list.txt for filenames
                with open(test_txt, 'r') as f:
                    filenames = [line.strip() for line in f if line.strip()]
                
                # Get all IDs from ChromaDB
                client = chromadb.PersistentClient(path=db_path)
                collection = client.get_collection("medical_images")
                all_data = collection.get(include=["embeddings"])
                all_ids = all_data['ids']
                all_embeddings = all_data['embeddings']
                print(f"Found {len(all_ids)} entries in ChromaDB")
                
                # Create sampled_ids_raw from filenames in test_list.txt
                sampled_ids_raw = [f"{filename}_original" for filename in filenames if f"{filename}_original" in all_ids]
                sampled_ids = filenames
                print(f"Using {len(sampled_ids)} filenames from {test_txt}")
                
                # Create a temporary ChromaDB copy without the sampled entries
                import tempfile
                import shutil
                temp_db_path = tempfile.mkdtemp(prefix="temp_medical_db_")
                shutil.copytree(db_path, temp_db_path, dirs_exist_ok=True)
                print(f"Created temporary ChromaDB copy at: {temp_db_path}")
                
                # Remove entries from the temporary ChromaDB copy
                temp_client = chromadb.PersistentClient(path=temp_db_path)
                temp_collection = temp_client.get_collection("medical_images")
                
                # Find all IDs that contain any of the sampled base filenames
                ids_to_remove = []
                for id_str in all_ids:
                    for sampled_base in sampled_ids:
                        if sampled_base in id_str:
                            ids_to_remove.append(id_str)
                            break
                
                print(f"Removing {len(ids_to_remove)} entries from temporary ChromaDB to avoid data leakage")
                print(f"Count before deletion: {temp_collection.count()}")
                print(f"Sample of IDs to delete: {ids_to_remove[:5]}")
                
                if ids_to_remove:
                    temp_collection.delete(ids=ids_to_remove)
                    print(f"Successfully removed {len(ids_to_remove)} entries from temporary copy")
                
                print(f"Temporary ChromaDB now has {temp_collection.count()} entries")
                
                del temp_client, temp_collection
                del client, collection
                
                # Update db_path to use the temporary copy
                db_path = temp_db_path
                
                # Get embeddings for sampled IDs from ChromaDB data
                embeddings_dict = {}
                
                for i, id_str in enumerate(all_ids):
                    if id_str in sampled_ids_raw:  # Use original sampled IDs with _original
                        filename = id_str.replace('_original', '').replace('.tiff', '')
                        embeddings_dict[filename] = np.array(all_embeddings[i], dtype=np.float32)
                
                filenames = list(embeddings_dict.keys())
                print(f"Loaded {len(embeddings_dict)} embeddings from ChromaDB")
                need_projection = False  # Already projected in ChromaDB
                skip_projection = True
            
        elif MODE == "test":
            # Load all embeddings from test directory
            embeddings_dict = load_embeddings_dict(test_npz_dir)
            filenames = list(embeddings_dict.keys())
            need_projection = True  # Raw embeddings need projection  
            skip_projection = False
            
        else:
            raise ValueError(f"Invalid MODE: {MODE}. Must be 'train' or 'test'")
        
        # Initialize inference system
        inference = MultiGPUMedicalInference(db_path, model_path, embeddings_dict, need_projection=need_projection, prompt_path=prompt_path)
        
        # Process files
        results = inference.process_batch_parallel(filenames, skip_projection=skip_projection)
        
        # Statistics
        k_values = [r.get('k_used', 0) for r in results if r.get('k_used', 0) > 0]
        if k_values:
            print(f"\nDynamic k distribution:")
            print(f"  Min k: {min(k_values)}")
            print(f"  Max k: {max(k_values)}")
            print(f"  Mean k: {np.mean(k_values):.2f}")
            print(f"  Std k: {np.std(k_values):.2f}")
        
        # Save results
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_filename} with {len(results)} entries")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    
    finally:
        if inference:
            inference.cleanup_models()
        full_cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted, cleaning up...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        full_cleanup()
        print("Memory cleanup completed")
