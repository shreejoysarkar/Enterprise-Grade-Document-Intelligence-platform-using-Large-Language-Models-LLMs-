'''

Helper functions for the RAG implementation.

'''

# Import required libraries
from dotenv import load_dotenv
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

#Helper Functions

def display_chunks(chunks, max_display=3, show_metadata=True):
    """Display chunks with metadata."""
    print(f"\nTotal chunks: {len(chunks)}")
    print("="*80)
    
    for i, chunk in enumerate(chunks[:max_display]):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk.text[:200]}...")
        if show_metadata and hasattr(chunk, 'token_count'):
            print(f"Token count: {chunk.token_count}")
        if show_metadata and hasattr(chunk, 'start_char'):
            print(f"Position: {chunk.start_char} - {chunk.end_char}")
        print("-" * 80)
    
    if len(chunks) > max_display:
        print(f"\n... and {len(chunks) - max_display} more chunks")


def visualize_chunk_sizes(chunks, title="Chunk Size Distribution"):
    """Create bar chart of chunk token counts."""
    token_counts = [chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text) 
                   for chunk in chunks]
    
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(token_counts, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.title(f'{title}\nDistribution')
    plt.axvline(np.mean(token_counts), color='red', linestyle='--', label=f'Mean: {np.mean(token_counts):.0f}')
    plt.legend()
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(token_counts, vert=True)
    plt.ylabel('Token Count')
    plt.title(f'{title}\nBox Plot')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(token_counts):.1f}")
    print(f"  Median: {np.median(token_counts):.1f}")
    print(f"  Std Dev: {np.std(token_counts):.1f}")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")


def compare_chunkers(text, chunkers_dict, sample_size=1000):
    """Compare multiple chunkers on same text."""
    results = {}
    sample_text = text[:sample_size] if len(text) > sample_size else text
    
    for name, chunker in chunkers_dict.items():
        start_time = time.time()
        chunks = chunker.chunk(sample_text)
        elapsed = time.time() - start_time
        
        token_counts = [chunk.token_count if hasattr(chunk, 'token_count') else len(chunk.text) 
                       for chunk in chunks]
        
        results[name] = {
            'num_chunks': len(chunks),
            'avg_size': np.mean(token_counts),
            'std_dev': np.std(token_counts),
            'time': elapsed
        }
    
    df = pd.DataFrame(results).T
    return df

print("Helper functions loaded!")