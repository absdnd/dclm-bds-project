import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import wandb

# Module-level state
_duplicates_buffer: List[Dict] = []
_excel_file_path: Optional[str] = None
_threshold: Optional[float] = None
_max_duplicates = 10
_initialized = False

def init_duplicate_logging(enabled: bool, threshold: float) -> None:
    global _excel_file_path, _threshold, _initialized
    if _initialized:
        return
        
    _threshold = threshold
    _initialized = True
    
    if not enabled:
        return
        
    os.makedirs("duplicates_logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _excel_file_path = f"duplicates_logs/duplicates_thresh_{_threshold}_{timestamp}.xlsx"
    # Create empty Excel file with header
    pd.DataFrame(columns=[
        'Original Text',
        'Duplicate Text',
        'Similarity Threshold'
    ]).to_excel(_excel_file_path, index=False, engine='openpyxl')

def log_duplicate_pair(
    original_text: str,
    duplicate_text: str,
    
) -> None:
    global _duplicates_buffer
    
    if _excel_file_path is None or len(_duplicates_buffer) >= _max_duplicates:
        return
        
    _duplicates_buffer.append({
        
        'Original Text': original_text,
        'Duplicate Text': duplicate_text,
        'Similarity Threshold': _threshold
    })

def save_duplicates_to_excel() -> None:
    global _duplicates_buffer
    
    if _excel_file_path is None or not _duplicates_buffer:
        return
    
    # Read existing data if file exists and is not empty
    try:
        existing_df = pd.read_excel(_excel_file_path, engine='openpyxl')
        if existing_df.empty:
            existing_df = pd.DataFrame(columns=[
                'Group ID', 'Group Size', 'Original Text', 
                'Duplicate Text', 'Similarity Threshold'
            ])
    except:
        existing_df = pd.DataFrame(columns=[
            'Group ID', 'Group Size', 'Original Text', 
            'Duplicate Text', 'Similarity Threshold'
        ])
    
    # Create new DataFrame from buffer
    new_df = pd.DataFrame(_duplicates_buffer)
    
    # Explicitly handle empty DataFrames before concat
    if existing_df.empty:
        combined_df = new_df
    elif new_df.empty:
        combined_df = existing_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Ensure max 10 records and save
    combined_df = combined_df.head(_max_duplicates)
    
    with pd.ExcelWriter(_excel_file_path, engine='openpyxl', mode='w') as writer:
        combined_df.to_excel(writer, index=False)
        
        # Auto-adjust column widths
        if not combined_df.empty:
            for column in combined_df:
                col_idx = combined_df.columns.get_loc(column)
                writer.sheets['Sheet1'].column_dimensions[
                    chr(65 + col_idx)
                ].width = max(20, len(str(combined_df[column].iloc[0])) + 2)
    
    wandb.save(_excel_file_path, policy="now")
    _duplicates_buffer = []
    return _excel_file_path