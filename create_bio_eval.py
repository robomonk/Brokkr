import os
from datasets import Dataset

# Output Directory
OUTPUT_PATH = "./processed_data/val_biotools"

# --- Define Unseen Tools (The "Test") ---
# These are tools TxGemma knows scientifically, but has never seen as APIs.
biotools_data = [
    {
        "tool_id": "ClustalOmega_Align",
        "query": "I have three protein sequences that I suspect are related. Align them to find conserved regions.\nSequences:\n>SeqA\nMVLSPADKTNVKAA\n>SeqB\nMVLSPADKTNVKAA\n>SeqC\nMVLSPADKTNVKA",
        "api_doc": """
Tool: ClustalOmega
Description: Multiple Sequence Alignment tool for proteins or DNA.
Parameters:
- sequences (list[str]): List of sequences in FASTA format.
- outfmt (str): Output format. Options: 'alu', 'clustal', 'fasta', 'msf', 'phylip', 'selex', 'stockholm', 'vienna'. Default: 'clustal'.
- dealign (bool): Dealign input sequences. Default: false.
- mbed (bool): Use mBed-like clustering guide tree. Default: true.
""",
        "ground_truth": '{"name": "ClustalOmega", "arguments": {"sequences": [">SeqA\\nMVLSPADKTNVKAA", ">SeqB\\nMVLSPADKTNVKAA", ">SeqC\\nMVLSPADKTNVKA"], "outfmt": "clustal"}}'
    },
    {
        "tool_id": "NCBI_BLAST_p",
        "query": "Identify this protein sequence by searching the SwissProt database: MKWVTFISLLLLFSSAYSRGVFRR",
        "api_doc": """
Tool: NCBI_BLAST
Description: Basic Local Alignment Search Tool. Finds regions of local similarity between sequences.
Parameters:
- program (str): Program name. Valid: 'blastp' (protein-protein), 'blastn' (nucleotide-nucleotide), 'blastx', 'tblastn'.
- database (str): Database to search. Valid: 'nr', 'refseq', 'swissprot', 'pdb'.
- sequence (str): The query sequence.
- expect (float): Expect value (E-value) cutoff. Default: 10.0.
""",
        "ground_truth": '{"name": "NCBI_BLAST", "arguments": {"program": "blastp", "database": "swissprot", "sequence": "MKWVTFISLLLLFSSAYSRGVFRR"}}'
    },
    {
        "tool_id": "DiffDock_Docking",
        "query": "Run a molecular docking simulation for the ligand 'Imatinib' against the protein structure 1IEP.",
        "api_doc": """
Tool: DiffDock
Description: Diffusion-based generative model for molecular docking. Predicts the 3D structure of a drug-target complex.
Parameters:
- protein_pdb_id (str): The PDB ID of the target protein (e.g., '6w01').
- ligand_description (str): SMILES string or name of the ligand.
- num_poses (int): Number of poses to generate. Default: 10.
- precision (str): 'low', 'medium', 'high'. Higher precision takes longer. Default: 'low'.
""",
        "ground_truth": '{"name": "DiffDock", "arguments": {"protein_pdb_id": "1IEP", "ligand_description": "Imatinib", "num_poses": 10, "precision": "medium"}}'
    }
]

def format_for_gemma(item):
    # Construct the Zero-Shot Prompt
    # 1. Inject API Documentation (The "Manual")
    # 2. Inject User Query
    # 3. Expect "Thought" and "Action"
    
    prompt = (
        f"<start_of_turn>user\n"
        f"You are an expert bioinformatics agent. Use the provided tool definition to solve the query.\n\n"
        f"{item['api_doc']}\n\n"
        f"Query: {item['query']}<end_of_turn>\n"
        f"<start_of_turn>model\n<think>\n" 
    )
    
    # We leave the text "open" for the model to complete the thought and action
    return {
        "text": prompt, # Used for input
        "tool_id": item['tool_id'],
        "ground_truth": item['ground_truth'] # Used for manual eval check
    }

def main():
    print("Generatng Bio-Tools Benchmark...")
    
    formatted_data = [format_for_gemma(x) for x in biotools_data]
    
    # Create Dataset Object
    dataset = Dataset.from_list(formatted_data)
    
    # Save to Disk (so eval_student.py can load it)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    dataset.save_to_disk(OUTPUT_PATH)
    
    print(f"Saved {len(dataset)} challenging bio-tasks to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()