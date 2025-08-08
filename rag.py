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
warnings.filterwarnings("ignore")

def clear_gpu_memory():
    """Clear GPU memory completely"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def clear_cpu_memory():
    """Clear CPU memory"""
    gc.collect()

def full_cleanup():
    """Complete memory cleanup"""
    clear_gpu_memory()
    clear_cpu_memory()

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
    """Load embeddings from NPZ files in directory into dictionary"""
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

class MultiGPUMedicalInference:
    def __init__(self, db_path, model_path, embeddings_dict, num_gpus=None):
        full_cleanup()
        
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.models = {}
        self.tokenizers = {}
        self.embeddings_dict = embeddings_dict
        self.lock = threading.Lock()
        
        print(f"Initializing with {self.num_gpus} GPUs")
        
        # Load database
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("medical_images")
        print(f"Loaded database with {self.collection.count()} medical cases")
        
        # Load projection model
        self.projection_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.projection_model = ProjectionHead(768).to(self.projection_device)
        checkpoint = torch.load(model_path, map_location=self.projection_device)
        self.projection_model.load_state_dict(checkpoint['model'])
        self.projection_model.eval()
        print(f"Loaded projection model on {self.projection_device}")

        # Load models on each GPU
        for gpu_id in tqdm(range(self.num_gpus), desc="Loading models"):
            self._load_model_on_gpu(gpu_id)

    def _load_model_on_gpu(self, gpu_id):
        """Load model on specific GPU with memory management"""
        device = f'cuda:{gpu_id}'
        print(f"Loading LLM on {device}...")
        
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
        """Explicitly delete models and free memory"""
        for gpu_id in list(self.models.keys()):
            if gpu_id in self.models:
                del self.models[gpu_id]
            if gpu_id in self.tokenizers:
                del self.tokenizers[gpu_id]
        
        self.models.clear()
        self.tokenizers.clear()
        
        if hasattr(self, 'projection_model'):
            del self.projection_model
        
        if hasattr(self, 'client'):
            del self.client
            del self.collection
        
        full_cleanup()
    
    def project_embedding(self, raw_embedding):
        """Project 768-dim embedding to 256-dim using trained model"""
        with torch.no_grad():
            embedding_tensor = torch.tensor(raw_embedding, dtype=torch.float32).unsqueeze(0).to(self.projection_device)
            projected = self.projection_model(embedding_tensor)
            return projected.cpu().numpy().flatten()
    
    def get_all_similarities(self, query_embedding):
        """Get similarities to all cases in database"""
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
        """Dynamically determine k based on similarity gaps"""
        similarities = self.get_all_similarities(query_embedding)
        similarities = np.sort(similarities)[::-1]
        
        threshold = 0.0010
        
        for i in range(start_check-1, min(len(similarities)-1, max_k-1)):
            gap = similarities[i] - similarities[i+1]
            if gap > threshold:
                return i + 1
        
        return max_k
    
    def search_similar_cases(self, query_embedding, k=3):
        """Find similar cases in database"""
        k = int(k)
        with self.lock:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "distances"]
            )
        return results['documents'][0], results['distances'][0]
    
    def generate_report(self, similar_reports, distances, gpu_id):
        """Generate medical report using LLM on specific GPU"""
        context = ""
        for i, (report, dist) in enumerate(zip(similar_reports, distances)):
            similarity_score = 1 - dist
            context += "-----------------------\n"
            context += f"[Similar Report {i}]"
            context += f"Similarity Score: {similarity_score}\n"
            context += f"{report}\n"
        
        prompt = f"""You are a medical report compiler. 

        You will be given pathology reports corresponding to Whole Slide Images (WSIs) that are most similar to a new WSI.
        Your task is to synthesize a new report using a sequential majority-voting strategy for histologic types.

        ### Histologic Type Selection Process

        **Step 1 - First Histologic Type (Required):**
        - Count occurrence of each histologic type across ALL reports
        - Select the histologic type that appears in the majority (>50%) of reports
        - If no single type has majority, select the most frequent one
        - This becomes histologic type #1

        **Step 2 - Additional Histologic Types (Optional):**
        - Filter to ONLY reports that contain the first histologic type
        - Among these filtered reports, count occurrence of OTHER histologic types
        - If other histologic type appears in majority (more than 50%) of these filtered reports, include it as histologic type. 

        **Step 3 - Note Types (Optional):**
        - Among ALL reports, count occurrence of either Note types: "- \n\n   Note) The specimen does not include muscle proper." and "\n\n   Note) The specimen includes muscle proper."
        - If one Note type appears in majority (more than 50%) of ALL reports, include it after all histologic types. 

        ---

        ### Output Format

        The output must follow this format exactly:

        Organ, Procedure;
           Histologic type

        **IMPORTANT**: 
        - There needs to be exactly 3 spaces after the new line operator and before the histologic type like the reports given to you.
        - Do NOT number histologic types if there is only one histologic type. Only number them if there are more than one histology type. 
        - If you number histologic types when there is only one histologic type, you lose your job. 

        ---

        ### Mandatory Constraints

        - Use only terms from the provided Organ, Procedure, and Histologic type lists.
        - Copy terms **exactly** — do not rephrase, change punctuation, capitalization, or use synonyms.
        - For histologic types with placeholders (e.g., [ductal_carcinoma_type]), you MUST:
        1. Use the complete template format exactly as written
        2. Fill placeholders with options provided below exactly as written (preserve case)
        3. Do NOT abbreviate, omit, or modify any part of the template structure
        - Do NOT include histologic types that begin with the same two words (e.g., do NOT use both "Non-small cell carcinoma, favor adenocarcinoma" and "Non-small cell carcinoma, favor squamous cell carcinoma").

        **CRITICAL: Preserve exact capitalization and formatting from the lists below. Do not change "moderately" to "Moderately" or "low" to "Low", etc.**

        Break down each report into Organ, Procedure, and Histologic type. For each component:
        - If ≥50% of reports agree on a component, use that component
        - Apply this separately for Organ, Procedure, and Histologic type

        **For numbering histologic types:**
        - Number histologic types (1., 2., 3., 4.) ONLY if there are multiple histologic types in the report
        - Exception: Do NOT number histologic types that start with "Note)" - these appear without numbers

        **Order histologic types by priority:**
        - **Priority 1** (highest): Invasive carcinoma*, invasive lobular carcinoma, papillary neoplasm, non-invasive papillary urothelial carcinoma*, invasive urothelial carcinoma*, no tumor present, no evidence of malignancy or granuloma
        - **Priority 2**: Atypical ductal hyperplasia, ductal carcinoma in situ, duct ectasia, lobular carcinoma in situ
        - **Priority 3**: Chronic granulomatous inflammation*, chronic inflammation*
        - **Priority 4**: All other histologic types
        - **Priority 5**: Microcalcification
        - **Priority 6** (lowest): Note) statements
        - (*) indicates templates starting with these terms

        **Important:** Do not add explanations or additional descriptions. Only return the final report.

        ---

        ### Organ (choose one)

        - Breast
        - Colon
        - Lung
        - Prostate
        - Stomach
        - Rectum
        - Urinary bladder
        - Uterine cervix

        ---

        ### Procedure (choose one)

        - biopsy
        - core biopsy
        - colonoscopic biopsy
        - colonoscopic mucosal resection
        - colonoscopic polypectomy
        - colonoscopic submucosal dissection
        - colposcopic biopsy
        - endoscopic biopsy
        - endoscopic mucosal resection
        - endoscopic submucosal dissection
        - loop electrosurgical excision procedure
        - mammotome biopsy
        - sono-guided core biopsy
        - transurethral resection
        - punch biopsy

        ---

        ### Histologic type (choose one or more)

        Select one or more histologic types **from the full list below**.

        Below are the available options:

        - Acinar adenocarcinoma, Gleason's score [acinar_adenocarcinoma_score_total] ([acinar_adenocarcinoma_score_1]+[acinar_adenocarcinoma_score_2]), grade group [acinar_adenocarcinoma_grade_group], tumor volume: [acinar_adenocarcinoma_tumor_volume]%
        - Acinar adenocarcinoma, Gleason's score [acinar_adenocarcinoma_score_total] ([acinar_adenocarcinoma_score_1]+[acinar_adenocarcinoma_score_2]), grade group [acinar_adenocarcinoma_grade_group] (Gleason pattern 4: [acinar_adenocarcinoma_gleason_pattern]%), tumor volume: [acinar_adenocarcinoma_tumor_volume]%
        - Acute prostatitis
        - Adenocarcinoma
        - Adenocarcinoma, favor colorectal primary
        - Adenocarcinoma, [adenocarcinoma_differentiation] differentiated [adenocarcinoma_component]
        - Adenosquamous carcinoma
        - Amyloidosis
        - Apocrine adenosis
        - Apocrine metaplasia
        - Atypical ductal hyperplasia
        - Atypical lobular hyperplasia
        - Carcinoid/neuroendocrine tumor, NOS
        - Carcinoma with apocrine differentiation
        - Chronic active cervicitis
        - Chronic active colitis
        - Chronic active gastritis
        - Chronic active gastritis\n  with [chronic_active_gastritis]
        - Chronic gastritis
        - Chronic gastritis\n  with [chronic_gastritis]
        - Chronic granulomatous inflammation with foreign body reaction
        - Chronic granulomatous inflammation with necrosis
        - Chronic granulomatous inflammation without necrosis
        - Chronic inflammation with organizing fibrosis
        - Chronic inflammation with type 2 pneumocyte hyperplasia
        - Chronic nonspecific cervicitis
        - Chronic nonspecific cervicitis with squamous metaplasia
        - Chronic nonspecific inflammation
        - Colmnar cell lesion
        - Columnar cell change
        - Columnar cell hyperplasia
        - Columnar cell lesion
        - Desmoid fibromatosis
        - Duct ectasia
        - Ductal carcinoma in situ
        - Ductal carcinoma in situ\n  - Type: [ductal_carcinoma_type]\n  - Nuclear grade: [ductal_carcinoma_nuclear_grade]\n  - Necrosis: [ductal_carcinoma_necrosis]
        - Ductal carcinoma in situ in intraductal papilloma
        - Endocervical adenocarcinoma in situ (AIS), HPV-associated
        - Endocervical adenocarcinoma, HPV-associated, usual type
        - Endocervical adenocarcinoma, HPV-independent, clear cell type
        - Endocervical adenocarcinoma, HPV-independent, gastric type
        - Endocervical polyp
        - Endometrioid adenocarcinoma
        - Endometrioid carcinoma
        - Extranodal marginal zone B cell lymphoma of MALT type
        - Fat necrosis
        - Fibroadenoma
        - Fibroadenomatoid change
        - Fibrocystic change
        - Fibroepithelial lesion, favor fibroadenoma
        - Fibroepithelial tumor
        - Fibroepithelial tumor, favor fibroadenoma
        - Fibroepithelial tumor, favor phyllodes tumor
        - Flat epithelial atypia
        - Foreign body reaction
        - Fundic gland polyp
        - Fungal ball, morphologically consistent with Aspergillus spp.
        - Fungal infection, morphologically consistent with cryptococcus spp.
        - Gastrointestinal stromal tumor
        - Granulomatous lobular mastitis
        - High-grade squamous intraepithelial lesion (HSIL; CIN 2)
        - High-grade squamous intraepithelial lesion (HSIL; CIN 2)\n  with glandular involvement
        - High-grade squamous intraepithelial lesion (HSIL; CIN 3)
        - High-grade squamous intraepithelial lesion (HSIL; CIN 3)\n  with glandular involvement
        - Hyperplastic polyp
        - Inflammatory polyp
        - Intraductal papilloma
        - Intraductal papilloma with apocrine metaplasia
        - Intraductal papilloma with usual ductal hyperplasia
        - Invasive carcinoma of no special type, grade [invasive_carcinoma_nos_grade] (Tubule formation: [invasive_carcinoma_nos_tubule], Nuclear grade: [invasive_carcinoma_nos_nuclear], Mitoses: [invasive_carcinoma_nos_mitoses])
        - Invasive carcinoma with features of mucinous carcinoma
        - Invasive cribriform carcinoma
        - Invasive lobular carcinoma
        - Invasive lobular carcinoma, pleomorphic type
        - Invasive micropapillary carcinoma
        - Invasive mucinous adenocarcinoma
        - Invasive squamous cell carcinoma
        - Invasive urothelial carcinoma, with involvement of subepithelial connective tissue
        - Invasive urothelial carcinoma, with 1) involvement of subepithelial connective tissue\n  2) squamous differentiation
        - Invasive urothelial carcinoma, \n  with involvement of muscle proper
        - Large cell neuroendocrine carcinoma
        - Lobular carcinoma in situ
        - Low-grade squamous intraepithelial lesion (LSIL; CIN 1)
        - Malignant lymphoma
        - Malignant melanoma
        - Metaplastic carcinoma
        - Metastatic adenocarcinoma, from colon primary
        - Metastatic carcinoma, from breast primary
        - Metastatic high grade serous carcinoma
        - Metastatic leiomyoma
        - Micro-invasive carcinoma
        - Microcalcification
        - Microglandular hyperplasia
        - Mucinous adenocarcinoma
        - Mucinous carcinoma
        - Mucocele-like lesion
        - Mucoepidermoid carcinoma
        - Neuroendocrine tumor, grade 1
        - No evidence of malignancy or granuloma
        - No evidence of tumor
        - No tumor present
        - Non-invasive papillary urothelial carcinoma, high grade
        - Non-invasive papillary urothelial carcinoma, high grade, with squamous differentiation
        - Non-invasive papillary urothelial carcinoma, low grade
        - Non-small cell carcinoma, favor adenocarcinoma
        - Non-small cell carcinoma, favor squamous cell carcinoma
        - Non-small cell carcinoma, not otherwise specified
        - Papillary neoplasm
        - Papillary neoplasm with apocrine metaplasia
        - Papillary neoplasm with atypical ductal hyperplasia
        - Papillary neoplasm with usual ductal hyperplasia
        - Phyllodes tumor
        - Pleomorphic carcinoma
        - Poorly cohesive carcinoma, not otherwise specified
        - Poorly cohesive carcinoma, signet ring cell type
        - Pseudoangiomatous stromal hyperplasia
        - Pulmonary hamartoma
        - Radiation-related atypia
        - Sclerosing adenosis
        - Sclerosing pneumocytoma
        - Sessile serrated adenoma with low grade dysplasia
        - Sessile serrated lesion
        - Sessile serrated lesion with low grade dysplasia
        - Signet-ring cell carcinoma
        - Small cell carcinoma
        - Solid papillary carcinoma in situ
        - Squamous cell carcinoma
        - Traditional serrated adenoma
        - Tubular adenoma with [tubular_adenoma_dysplasia] grade dysplasia
        - Tubular carcinoma
        - Tubulovillous adenoma with [tubuvillous_adenoma_dysplasia] grade dysplasia
        - Urothelial carcinoma in situ
        - Usual ductal hyperplasia

        Some histologic types contain placeholders like [acinar_adenocarcinoma_score_total] and [tubuvillous_adenoma_dysplasia].
        **Fill these placeholders with the available options given to you below EXACTLY AS WRITTEN (preserve case).** 

        [ductal_carcinoma_type] can be one or two of: Cribriform, Micropapillary, Solid, Papillary, or Flat. 
        If two, connect with "and" and make the second one lower-case. 
        If two and one is Cribriform or Solid, they come first. 

        [ductal_carcinoma_nuclear_grade] is one of: High, Intermediate, or Low. 

        [ductal_carcinoma_necrosis] is one of: Present (Comedo-type), Present (Focal), Absent, Present

        [invasive_carcinoma_nos_grade] is one of: I, II, III

        [invasive_carcinoma_nos_tubule], [invasive_carcinoma_nos_nuclear], [invasive_carcinoma_nos_mitoses] are one of: 1, 2, 3

        [adenocarcinoma_differentiation] is one of: moderately, poorly, well

        [adenocarcinoma_component] is one (could be two) of: mucinous component, poorly cohesive carcinoma component, signet ring cell component, arising from tubulovillous adenoma with high grade dysplasia

        [tubular_adenoma_dysplasia] is one of: low, high

        [tubuvillous_adenoma_dysplasia] is one of: low, high

        [chronic_active_gastritis] is 1 to 4 of: erosion, interstinal metaplasia, lymphoid aggregates, CMV-infected cells, reactive atypia, foveolar epithelial hyperplasia, ulcer
        Number them if more than one apply (e.g., 1), 2), 3), 4))

        [chronic_gastritis] is one or two of: erosion, interstinal metaplasia, lymphoid aggregates, foveolar epithelial hyperplasia, ulcer
        Number them if more than one apply (e.g., 1), 2)) 

        ---
        
        ### Note type (choose one if applicable)

        - \n\n   Note) The specimen does not include muscle proper.
        - \n\n   Note) The specimen includes muscle proper.

        ---

        ### Input

        Below are the reports from WSIs most similar to the new case, along with their Similarity Scores.

        {context}
        -----------------------

        Generate the new medical report below:

        """

        print(context + '\n')
        
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
    
    def process_single_file(self, filename, gpu_id):
        """Process a single file with dynamic k selection"""
        try:
            raw_embedding = self.embeddings_dict[filename]
            projected_embedding = self.project_embedding(raw_embedding)
            
            k = self.get_dynamic_k(projected_embedding)
            similar_reports, distances = self.search_similar_cases(projected_embedding, k=k)
            
            if k == 1 or k == 2: 
                report = similar_reports[0]
            else:
                report = self.generate_report(similar_reports, distances, gpu_id)
            
            print('-----------------------')
            print(f"File: {filename}, Dynamic k: {k}")
            print(f"Report: {report}")
            print('-----------------------')
            
            return {"id": filename, "report": report, "k_used": k}
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return {"id": filename, "report": "Error: Unable to generate report", "k_used": 0}
    
    def process_batch_parallel(self, filenames, max_workers=None):
        """Process all test files in parallel across GPUs with dynamic k"""
        if max_workers is None:
            max_workers = self.num_gpus
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filename = {}
            for i, filename in enumerate(filenames):
                gpu_id = i % self.num_gpus
                future = executor.submit(self.process_single_file, filename, gpu_id)
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

    # Configuration - set once
    db_path = "./medical_db"
    model_path = "model/semantic_model.pth"
    npz_dir = "../../reg2025/phase1_vectors"
    
    inference = None
    try:
        print(f"\nUsing multi-GPU processing with {torch.cuda.device_count()} GPUs")
        
        # Load embeddings once
        embeddings_dict = load_embeddings_dict(npz_dir)
        filenames = list(embeddings_dict.keys())
        
        # Initialize inference system
        inference = MultiGPUMedicalInference(db_path, model_path, embeddings_dict)
        
        # Process files
        results = inference.process_batch_parallel(filenames)
        
        # Analyze k distribution
        k_values = [r.get('k_used', 0) for r in results if r.get('k_used', 0) > 0]
        if k_values:
            print(f"\nDynamic k distribution:")
            print(f"  Min k: {min(k_values)}")
            print(f"  Max k: {max(k_values)}")
            print(f"  Mean k: {np.mean(k_values):.2f}")
            print(f"  Std k: {np.std(k_values):.2f}")
        
        # Save submission
        with open("submission.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Submission saved with {len(results)} results using dynamic k values")
        
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