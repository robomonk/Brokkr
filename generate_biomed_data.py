import json
import os
import random
from datasets import Dataset

# Configuration
OUTPUT_DIR = "./processed_data"
NUM_SAMPLES = 500  # How many synthetic samples to generate

# --- 1. Domain Entities (for randomization) ---
PROTEINS = ["P53_HUMAN", "INS_HUMAN", "ALBU_HUMAN", "BRCA1_HUMAN", "EGFR_HUMAN", "P00533", "P01308"]
DBS = ["uniprotkb", "embl", "pdb", "refseq"]
VARIANTS = ["rs123", "rs4988235", "rs1800497", "rs234", "rs99"]
TRAITS = ["height", "diabetes", "schizophrenia", "body mass index", "lung cancer"]
SEQUENCES = [
    ">seq1\\nATGCGT...", 
    ">seqA\\nMKV...", 
    ">sample1\\nCGTACG..."
]

# --- 2. Tool Definitions & Trace Templates ---

def generate_dbfetch_trace():
    """Generates a trace for the EBI DBFetch API (Simple Retrieval)"""
    prot = random.choice(PROTEINS)
    db = random.choice(DBS)
    
    query = f"Retrieve the entry for {prot} from {db}."
    
    # Trace Logic
    trace = f"<start_of_turn>user\n{query}<end_of_turn>\n"
    trace += f"<start_of_turn>model\n<think>\n"
    trace += f"I need to fetch data from {db} for the identifier {prot}.\n"
    trace += f"Action: dbfetch(db='{db}', id='{prot}', format='default')\n"
    trace += f"Observation: {{'entry': '{prot}', 'database': '{db}', 'content': 'ID {prot}...'}}\n"
    trace += f"Final Answer: I have retrieved the entry for {prot} from {db}.</think><end_of_turn>"
    
    return {"text": trace, "tool_id": "dbfetch"}

def generate_gwas_trace():
    """Generates a trace for the EBI GWAS Catalog API (Search)"""
    if random.random() > 0.5:
        # Variant Search
        var = random.choice(VARIANTS)
        query = f"Find GWAS associations for the variant {var}."
        trace = f"<start_of_turn>user\n{query}<end_of_turn>\n"
        trace += f"<start_of_turn>model\n<think>\n"
        trace += f"Action: gwas_associations(variantId='{var}')\n"
        trace += f"Observation: [{{'associationId': '123', 'pvalue': 1e-8, 'trait': 'Height'}}]\n"
        trace += f"Final Answer: The variant {var} is associated with traits like Height.</think><end_of_turn>"
        return {"text": trace, "tool_id": "gwas_associations"}
    else:
        # Trait Search
        trait = random.choice(TRAITS)
        query = f"What studies are available for {trait}?"
        trace = f"<start_of_turn>user\n{query}<end_of_turn>\n"
        trace += f"<start_of_turn>model\n<think>\n"
        trace += f"Action: gwas_studies(trait='{trait}')\n"
        trace += f"Observation: [{{'studyId': 'GCST001', 'title': 'Meta-analysis of {trait}'}}]\n"
        trace += f"Final Answer: I found several studies related to {trait}, including GCST001.</think><end_of_turn>"
        return {"text": trace, "tool_id": "gwas_studies"}

def generate_jdispatcher_trace():
    """Generates a trace for JDispatcher (Async Job: Submit -> Status -> Result)"""
    tool = random.choice(["ncbiblast", "clustalo"])
    seq = random.choice(SEQUENCES)
    job_id = f"{tool}-2024-{random.randint(1000,9999)}"
    
    if tool == "ncbiblast":
        query = "Run a BLAST search for this sequence."
    else:
        query = "Align these protein sequences using Clustal Omega."
        
    trace = f"<start_of_turn>user\n{query}\n{seq}<end_of_turn>\n"
    trace += f"<start_of_turn>model\n<think>\n"
    
    # Step 1: Submit
    trace += f"I will submit the job to {tool}.\n"
    trace += f"Action: {tool}_run(email='user@example.com', sequence='{seq[:10]}...')\n"
    trace += f"Observation: '{job_id}'\n"
    
    # Step 2: Check Status (Loop simulation)
    trace += f"Job submitted. Checking status for {job_id}.\n"
    trace += f"Action: {tool}_status(jobId='{job_id}')\n"
    trace += f"Observation: 'RUNNING'\n"
    trace += f"The job is still running. I will wait and check again.\n"
    trace += f"Action: {tool}_status(jobId='{job_id}')\n"
    trace += f"Observation: 'FINISHED'\n"
    
    # Step 3: Get Result
    trace += f"Job finished. Retrieving results.\n"
    trace += f"Action: {tool}_result(jobId='{job_id}', type='out')\n"
    trace += f"Observation: '...Alignment Score: 98.5...'\n"
    
    trace += f"Final Answer: The {tool} analysis is complete. The results show a high alignment score.</think><end_of_turn>"
    
    return {"text": trace, "tool_id": f"{tool}_run"}

def main():
    print(f"Generating {NUM_SAMPLES} synthetic biomedical traces...")
    
    samples = []
    
    for _ in range(NUM_SAMPLES):
        r = random.random()
        if r < 0.33:
            samples.append(generate_dbfetch_trace())
        elif r < 0.66:
            samples.append(generate_gwas_trace())
        else:
            samples.append(generate_jdispatcher_trace())
            
    # Split into Train/Test (No Val needed for synthetic, we trust the generator)
    random.shuffle(samples)
    split = int(NUM_SAMPLES * 0.9)
    train_data = samples[:split]
    test_data = samples[split:]
    
    os.makedirs(f"{OUTPUT_DIR}/biomed_train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/biomed_test", exist_ok=True)
    
    Dataset.from_list(train_data).save_to_disk(f"{OUTPUT_DIR}/biomed_train")
    Dataset.from_list(test_data).save_to_disk(f"{OUTPUT_DIR}/biomed_test")
    
    print(f"Success! Saved {len(train_data)} Train and {len(test_data)} Test traces.")
    print(f"Location: {OUTPUT_DIR}/biomed_train")

if __name__ == "__main__":
    main()
