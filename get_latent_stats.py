import os
import json
import argparse
from pathlib import Path

def find_latent_stats(search_dir):
    """
    Recursively searches for 'latent_stats.json' in the specified directory.
    """
    search_path = Path(search_dir)
    if not search_path.exists():
        print(f"Error: Directory '{search_dir}' does not exist.")
        return None

    print(f"Searching for 'latent_stats.json' in: {search_path.absolute()}")
    
    # Use rglob for recursive search
    matches = list(search_path.rglob("latent_stats.json"))
    
    if not matches:
        print("No 'latent_stats.json' found.")
        return None
    
    if len(matches) > 1:
        print(f"Found multiple matches. Returning the most recent one (by modification time).")
        # Sort by modification time, newest first
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for m in matches:
            print(f"  - {m}")
            
    return str(matches[0].absolute())

def main():
    parser = argparse.ArgumentParser(description="Find and print the absolute path to 'latent_stats.json'.")
    parser.add_argument("dir", type=str, nargs='?', default=".", 
                        help="Directory to search in (default: current directory).")
    args = parser.parse_args()

    path = find_latent_stats(args.dir)
    if path:
        print(f"\nLATENT_STATS_PATH={path}")
    else:
        exit(1)

if __name__ == "__main__":
    main()
