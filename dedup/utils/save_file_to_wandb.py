import wandb
from typing import List, Dict
import pandas as pd

# Module-level state
_duplicates_buffer: List[Dict] = []
_max_duplicates = 10  # Hardcoded or make configurable via function params

def log_duplicate_pair(
    original_text: str,
    duplicate_text: str,
    threshold: float  # Takes threshold directly
) -> None:
    """Log a duplicate pair to in-memory buffer"""
    global _duplicates_buffer
    
    if len(_duplicates_buffer) >= _max_duplicates:
        return
        
    _duplicates_buffer.append({
        
        'Original Text': original_text,
        'Duplicate Text': duplicate_text,
        'Similarity Threshold': threshold
    })

def save_duplicates_to_wandb(threshold: float) -> None:
    """Save collected duplicates directly to wandb"""
    global _duplicates_buffer
    
    if not _duplicates_buffer or not wandb.run:
        return
    
    df = pd.DataFrame(_duplicates_buffer).head(_max_duplicates)
    
    # 1. Log as interactive table
    wandb.log({
        "duplicates_table": wandb.Table(dataframe=df),
        "total_duplicates": len(_duplicates_buffer)
    })
    
    # 2. Save as versioned artifact
    artifact = wandb.Artifact(
        name=f"duplicates_thresh_{threshold}",
        type="dataset",
        metadata={"threshold": threshold}
    )
    with artifact.new_file("duplicates.csv") as f:
        df.to_csv(f, index=False)
    wandb.log_artifact(artifact)
    
    _duplicates_buffer = []  # Reset buffer