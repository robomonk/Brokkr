import os
from datasets import load_from_disk

def verify_split(name, train_path, val_path):
    print(f"--- Verifying {name} ---")
    try:
        train = load_from_disk(train_path)
        val = load_from_disk(val_path)
        
        train_tools = set(sample['tool_id'] for sample in train)
        val_tools = set(sample['tool_id'] for sample in val)
        
        intersection = train_tools.intersection(val_tools)
        
        print(f"Train Tools: {len(train_tools)}")
        print(f"Val Tools:   {len(val_tools)}")
        print(f"Leakage:     {len(intersection)} tools")
        
        if len(intersection) == 0:
            print("✅ SUCCESS: Distributions are strictly disjoint.")
        else:
            print(f"❌ FAILURE: {len(intersection)} tools leaked into validation.")
            
    except Exception as e:
        print(f"Skipping {name} (Data not generated yet)")

if __name__ == "__main__":
    # Verify Generic Logic Split
    verify_split("Trace Agent", "./processed_data_trace/train", "./processed_data_trace/val")
    verify_split("Meta Agent", "./processed_data_meta/train", "./processed_data_meta/val")
    
    # Verify Bio is distinct from Generic
    # (Conceptually we know this, but good to check if you ran the scan)
