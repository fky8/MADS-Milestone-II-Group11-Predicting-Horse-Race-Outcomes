import subprocess
import glob
import os
import re


def get_script_order():
    """
    Find all scripts in the current directory that start with a number (e.g., 1_*.py),
    sort them by that number, and return the ordered list of script filenames.
    """
    scripts = []
    for path in glob.glob("*.py"):
        m = re.match(r"(\d+)_.*\.py$", path)
        if m:
            scripts.append((int(m.group(1)), path))
    scripts.sort()
    return [p for _, p in scripts]


def run_scripts():
    """
    Run all numbered scripts in order, printing progress.
    """
    scripts = get_script_order()
    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)


if __name__ == "__main__":
    run_scripts()
