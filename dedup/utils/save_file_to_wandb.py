import wandb
from typing import List, Dict
import pandas as pd
from difflib import Differ
import html

# Module-level state
_duplicates_buffer: List[Dict] = []
_max_duplicates = 10

def _highlight_differences(original: str, duplicate: str) -> str:
    """Generate HTML with highlighted differences between texts"""
    d = Differ()
    diff = list(d.compare(original.splitlines(), duplicate.splitlines()))
    
    html_output = []
    for line in diff:
        if line.startswith('  '):
            # Unchanged
            html_output.append(f"<div>{html.escape(line[2:])}</div>")
        elif line.startswith('- '):
            # Removed from original
            html_output.append(f"<div style='background-color:#ffdddd;text-decoration:line-through'>{html.escape(line[2:])}</div>")
        elif line.startswith('+ '):
            # Added in duplicate
            html_output.append(f"<div style='background-color:#ddffdd'>{html.escape(line[2:])}</div>")
    
    return "<br>".join(html_output)

def log_duplicate_pair(
    original_text: str,
    duplicate_text: str,
    threshold: float
) -> None:
    """Log a duplicate pair with visual diff highlighting"""
    global _duplicates_buffer
    
    if len(_duplicates_buffer) >= _max_duplicates:
        return
    
    diff_html = _highlight_differences(original_text, duplicate_text)
    
    _duplicates_buffer.append({
        'Original': original_text,
        'Duplicate': duplicate_text,
        'Visual Diff': wandb.Html(diff_html),
        'Parameter': f"{threshold}"
    })

def save_duplicates_to_wandb() -> None:
    """Save duplicates directly to wandb with visual diffs"""
    global _duplicates_buffer
    
    if not _duplicates_buffer or not wandb.run:
        return
    
    # Create and log table
    table = wandb.Table(columns=["Original", "Duplicate", "Visual Diff", "Similarity"])
    for dup in _duplicates_buffer[:_max_duplicates]:
        table.add_data(
            dup['Original'],
            dup['Duplicate'],
            dup['Visual Diff'],
            dup['Similarity']
        )
    
    wandb.log({
        "duplicates": table,
        "total_duplicates": len(_duplicates_buffer)
    })
    
    # Clear buffer
    _duplicates_buffer = []