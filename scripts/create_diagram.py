#!/usr/bin/env python3
"""
Quick script to create diagrams after completing a lab
Usage: python scripts/create_diagram.py --lab lab1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import create_ml_pipeline_diagram, create_code_flow_diagram
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Create diagram for a lab')
    parser.add_argument('--lab', type=str, required=True, help='Lab name (e.g., lab1)')
    parser.add_argument('--type', type=str, default='pipeline', 
                       choices=['pipeline', 'flow', 'mental'], 
                       help='Type of diagram to create')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = f"learning/diagrams/{args.lab}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate diagram based on type
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.type == 'pipeline':
        save_path = f"{output_dir}/{args.lab}_pipeline_{timestamp}.png"
        create_ml_pipeline_diagram(
            title=f"{args.lab.upper()} - ML Pipeline",
            save_path=save_path
        )
    elif args.type == 'flow':
        # You'll need to define the flow for each lab
        code_blocks = [
            ("Setup", "Import and configure"),
            ("Data", "Load and prepare data"),
            ("Train", "Train models"),
            ("Evaluate", "Check metrics"),
            ("Decision", "Meet requirements?")
        ]
        create_code_flow_diagram(code_blocks, f"{args.lab.upper()} Code Flow")
    
    print(f"✅ Diagram created in: {output_dir}")
    print("📝 Don't forget to add your notes and insights!")

if __name__ == "__main__":
    main()
