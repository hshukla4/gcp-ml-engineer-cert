#!/usr/bin/env python3
"""
Quick script to add learning notes after each lab
Usage: python scripts/add_learning_note.py --lab lab1 --type insight
"""

import argparse
from datetime import datetime
import os

def main():
    parser = argparse.ArgumentParser(description='Add learning note')
    parser.add_argument('--lab', type=str, required=True, help='Lab name')
    parser.add_argument('--type', type=str, choices=['insight', 'question', 'experiment'], 
                       default='insight', help='Type of note')
    parser.add_argument('--note', type=str, help='Your note (or will open editor)')
    
    args = parser.parse_args()
    
    # Create notes directory
    notes_dir = f"learning/notes"
    os.makedirs(notes_dir, exist_ok=True)
    
    # Create or append to lab notes
    notes_file = f"{notes_dir}/{args.lab}_notes.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare note content
    if args.type == 'insight':
        prefix = "💡 INSIGHT"
    elif args.type == 'question':
        prefix = "❓ QUESTION"
    else:
        prefix = "🧪 EXPERIMENT"
    
    # Get note content
    if args.note:
        content = args.note
    else:
        # Open editor for longer notes
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as tf:
            tf.write(b"# Type your note here\n\n")
            tf.flush()
            subprocess.call(['nano', tf.name])  # or 'vim' or 'code'
            
            with open(tf.name, 'r') as f:
                content = f.read().strip()
            
            os.unlink(tf.name)
    
    # Append to notes file
    with open(notes_file, 'a') as f:
        if os.path.getsize(notes_file) == 0:
            f.write(f"# {args.lab.upper()} Learning Notes\n\n")
        
        f.write(f"\n## {prefix} - {timestamp}\n\n")
        f.write(f"{content}\n")
        f.write("-" * 50 + "\n")
    
    print(f"✅ Note added to: {notes_file}")

if __name__ == "__main__":
    main()
