import pandas as pd
from datasets import Dataset
import os

# --- Configuration ---
OUTPUT_PATH = "./processed_data/val_biotools"
BIOMNI_HF_URL = "hf://datasets/biomni/Eval1/biomni_eval1_dataset.parquet"

# --- 1. Define "Unseen" Tool Schemas ---
# These definitions simulate the "Documentation" the agent must read to solve the task.
# The agent has never seen these tools during training (it trained on generic ToolBench).
TOOL_DEFINITIONS = {
    "gwas_variant_prioritization": """
Tool: GWAS_Variant_Prioritizer
Description: Analyzes a list of genetic variants to identify the most likely causal variant for a specific disease trait based on functional genomics and LD scores.
Parameters:
- phenotype (str): The disease or trait name (e.g., 'Type 2 Diabetes').
- variant_list (list[str]): A list of rsIDs to evaluate (e.g., ['rs123', 'rs456']).
- population (str): The target population ancestry (e.g., 'EUR', 'EAS'). Default: 'EUR'.
""",
    "rare_disease_diagnosis": """
Tool: RareDisease_Diagnoser
Description: Matches a set of patient phenotypes and candidate genes to known rare diseases using OMIM and Orphanet data.
Parameters:
- phenotypes (list[str]): List of HPO terms or symptom descriptions (e.g., ['Ataxia', 'Seizures']).
- candidate_genes (list[str]): List of gene symbols found in sequencing (e.g., ['ATM', 'BRCA1']).
- inheritance_mode (str): Expected inheritance pattern (e.g., 'autosomal_recessive'). Optional.
""",
    "crispr_delivery": """
Tool: CRISPR_Delivery_Opt
Description: Recommends the optimal viral or non-viral delivery vector for CRISPR-Cas9 components based on tissue type and payload size.
Parameters:
- target_tissue (str): The specific organ or cell type (e.g., 'Liver', 'HSC').
- payload_size_kb (float): The total size of the genetic payload in kilobases.
- editing_type (str): 'nuclease', 'base_editor', or 'prime_editor'.
""",
    "drug_repurposing": """
Tool: DrugRepurpose_AI
Description: Predicts the efficacy of existing approved drugs for a new disease indication based on transcriptomic signature matching.
Parameters:
- disease_name (str): The target disease.
- candidate_drugs (list[str]): List of DrugBank IDs or names to screen.
- gene_signature (list[str]): List of up/down-regulated genes in the disease state.
"""
}

def format_biomni_sample(row):
    """
    Wraps the raw Biomni Q&A pair into our 'System 2' Agent Prompt format.
    """
    task = row['task_name']
    
    # Filter: Only process tasks for which we have defined a tool
    if task not in TOOL_DEFINITIONS:
        return None
        
    tool_doc = TOOL_DEFINITIONS[task]
    query = row['prompt']
    ground_truth = row['answer']
    
    # Construct Prompt (Gemma Tokenizer Format)
    # We explicitly instruct the model to use the tool definition provided.
    prompt = (
        f"<start_of_turn>user\n"
        f"You are an expert biomedical agent. Use the provided tool definition to solve the research problem.\n\n"
        f"{tool_doc}\n\n"
        f"Query: {query}<end_of_turn>\n"
        f"<start_of_turn>model\n<think>\n" 
    )
    
    # Return dict compatible with our eval_student.py script
    return {
        "text": prompt,
        "tool_id": task,
        "ground_truth": ground_truth,
        "task_type": task
    }

def main():
    print(f"Downloading Biomni Benchmark from {BIOMNI_HF_URL}...")
    try:
        # Pandas can read parquet URLs directly with pyarrow installed
        df = pd.read_parquet(BIOMNI_HF_URL)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Try installing dependencies: pip install pandas pyarrow")
        return

    print(f"Loaded {len(df)} total instances. Filtering for target tasks...")
    
    formatted_data = []
    stats = {}
    
    # Process every row (Option B: Use full dataset as test set)
    for _, row in df.iterrows():
        sample = format_biomni_sample(row)
        if sample:
            formatted_data.append(sample)
            # Track stats
            t = sample['task_type']
            stats[t] = stats.get(t, 0) + 1
            
    print(f"\nCompiled {len(formatted_data)} Zero-Shot Evaluation Samples:")
    for task, count in stats.items():
        print(f" - {task}: {count}")
        
    # Create HF Dataset and Save
    dataset = Dataset.from_list(formatted_data)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    dataset.save_to_disk(OUTPUT_PATH)
    
    print(f"\nSuccess! Saved benchmark to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()