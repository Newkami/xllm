#!/usr/bin/env python3

import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path
from typing import Set, Dict, List, Tuple


def extract_whitelist_kernels(header_file: str) -> Set[str]:
    """Extract kernel names from REG_KERNEL_LAUNCHER macros in header file."""
    whitelist = set()
    try:
        with open(header_file, 'r') as f:
            content = f.read()
        
        pattern = r'REG_KERNEL_LAUNCHER\s*\(\s*([^,\s\)]+)'
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            kernel_name = match.strip()
            if kernel_name:
                whitelist.add(kernel_name)
        
        return whitelist
    except Exception as e:
        print(f"ERROR: Failed to parse header file {header_file}: {e}")
        sys.exit(1)


def clear_triton_cache(cache_dir: str):
    """Clear triton cache directory."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
            print(f"INFO: Cleared triton cache directory: {cache_dir}")
        except Exception as e:
            print(f"WARNING: Failed to clear cache directory {cache_dir}: {e}")
    else:
        print(f"INFO: Cache directory does not exist: {cache_dir}")


def run_pytest(test_dir: str):
    """Run pytest on all Python files in the test directory."""
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"ERROR: Test directory does not exist: {test_dir}")
        sys.exit(1)
    
    py_files = list(test_path.glob("*.py"))
    if not py_files:
        print(f"ERROR: No Python files found in {test_dir}")
        sys.exit(1)
    
    print(f"INFO: Running pytest on {len(py_files)} Python file(s)...")
    for py_file in py_files:
        print(f"INFO: Running pytest on {py_file.name}...")
        result = subprocess.run(
            ["pytest", str(py_file), "-v"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"WARNING: pytest failed for {py_file.name}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        else:
            print(f"INFO: pytest passed for {py_file.name}")


def find_kernel_binaries(cache_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Find all .npubin files and their corresponding .json files in subdirectories."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return {}
    
    kernel_map = {}
    found_count = 0
    
    for npubin_file in cache_path.rglob("*.npubin"):
        json_file = npubin_file.with_suffix(".json")
        found_count += 1
        
        kernel_name = None
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    if 'kernel_name' in json_data:
                        kernel_name = json_data['kernel_name']
            except Exception as e:
                print(f"WARNING: Failed to parse JSON file {json_file}: {e}")
        
        if kernel_name is None:
            kernel_name = npubin_file.stem
        
        print(f"INFO: Found binary: {npubin_file.relative_to(cache_path)} (kernel: {kernel_name})")
        
        if kernel_name not in kernel_map:
            kernel_map[kernel_name] = []
        kernel_map[kernel_name].append((str(npubin_file), str(json_file)))
    
    print(f"INFO: Scanned subdirectories, found {found_count} .npubin file(s)")
    return kernel_map


def validate_and_copy_kernels(
    kernel_map: Dict[str, List[Tuple[str, str]]],
    whitelist: Set[str],
    dest_dir: str
):
    """Validate kernel binaries against whitelist and copy them to destination."""
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for kernel_name, files in kernel_map.items():
        if kernel_name not in whitelist:
            print(f"INFO: Skipping kernel '{kernel_name}' (not in whitelist)")
            skipped_count += 1
            continue
        
        if len(files) > 1:
            print(f"ERROR: Multiple binaries found for kernel '{kernel_name}':")
            for npubin_file, json_file in files:
                print(f"  - {npubin_file}")
            sys.exit(1)
        
        npubin_file, json_file = files[0]
        npubin_path = Path(npubin_file)
        json_path = Path(json_file)
        
        if not npubin_path.exists():
            print(f"ERROR: Binary file does not exist: {npubin_file}")
            sys.exit(1)
        
        dest_npubin = dest_path / npubin_path.name
        dest_json = dest_path / json_path.name
        
        try:
            shutil.copy2(npubin_path, dest_npubin)
            print(f"INFO: Copied {npubin_path.name} -> {dest_npubin}")
            
            if json_path.exists():
                shutil.copy2(json_path, dest_json)
                print(f"INFO: Copied {json_path.name} -> {dest_json}")
            else:
                print(f"WARNING: JSON file not found: {json_file}")
            
            copied_count += 1
        except Exception as e:
            print(f"ERROR: Failed to copy files for kernel '{kernel_name}': {e}")
            sys.exit(1)
    
    print(f"INFO: Summary - Copied: {copied_count}, Skipped: {skipped_count}")
    
    missing_kernels = whitelist - set(kernel_map.keys())
    if missing_kernels:
        print(f"ERROR: Missing kernels in whitelist: {', '.join(sorted(missing_kernels))}")
        sys.exit(1)


def main():
    script_dir = Path(__file__).parent
    
    header_file = script_dir / "kernel_launchers.h"
    test_dir = script_dir / "triton_src"
    
    triton_cache_dir = os.getenv("TRITON_CACHE_DIR", "/root/.triton/cache")
    binary_path = os.getenv("TRITON_BINARY_PATH")
    
    if not binary_path:
        print("ERROR: TRITON_BINARY_PATH environment variable is not set")
        sys.exit(1)
    
    if not header_file.exists():
        print(f"ERROR: Header file not found: {header_file}")
        sys.exit(1)
    
    print("INFO: Extracting whitelist kernels from header file...")
    whitelist = extract_whitelist_kernels(str(header_file))
    print(f"INFO: Found {len(whitelist)} kernel(s) in whitelist: {', '.join(sorted(whitelist))}")
    
    print(f"INFO: Clearing triton cache directory: {triton_cache_dir}")
    clear_triton_cache(triton_cache_dir)
    
    print(f"INFO: Running pytest on test files...")
    run_pytest(str(test_dir))
    
    print(f"INFO: Scanning for kernel binaries in {triton_cache_dir} and its subdirectories...")
    kernel_map = find_kernel_binaries(triton_cache_dir)
    print(f"INFO: Found {len(kernel_map)} unique kernel(s)")
    
    print(f"INFO: Validating and copying kernels to {binary_path}...")
    validate_and_copy_kernels(kernel_map, whitelist, binary_path)
    
    print("INFO: Script completed successfully")


if __name__ == "__main__":
    main()

