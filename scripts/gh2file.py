# This script is used to download zip archive from github and convert it into a single file
# inspired by: https://github.com/cognitivecomputations/github2file

import ast
import io
from typing import Callable, List
import requests
import zipfile
import os
import sys
from tqdm import tqdm


def has_sufficient_content(file_content, min_line_count=10):
    """Check if the file has a minimum number of substantive lines."""
    lines = [line for line in file_content.split('\n') if line.strip() and not line.strip().startswith('#')]
    return len(lines) >= min_line_count


def identity_transform(content):
    return content

def file_path_check(file_path, checkers=List[Callable[[str], bool]]):
    """Check if the file path passes all the checkers."""
    return all(checker(file_path) for checker in checkers)

def file_content_check(file_content, checkers=List[Callable[[str], bool]]):
    """Check if the file content passes all the checkers."""
    return all(checker(file_content) for checker in checkers)


def download_repo(repo_url, output_file, rules):
    """ Download the zip file from the given github URL, unzip it,
      and write the content into a single file """
    # Download the zip file
    zip_url = repo_url + '/archive/master.zip'
    resp = requests.get(zip_url, stream=True)

    total_size = int(resp.headers.get('Content-Length', 0))
    block_size = 4096

    zip_data = io.BytesIO()
    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
        for data in resp.iter_content(block_size):
            pbar.update(len(data))
            zip_data.write(data)

    zip_file = zipfile.ZipFile(zip_data)

    with open(output_file, "w", encoding="utf-8") as out:
        # Extract and write the content of each file
        for file_path in zip_file.namelist():
            if not file_path_check(file_path, rules["file_path_check"]):
                continue

            content = zip_file.read(file_path).decode('utf-8')
            if not file_content_check(content, rules["file_content_check"]):
                continue

            try:
                content = rules["transform"](content)
            except SyntaxError:
                continue

            out.write(f'# File: {file_path}\n')
            out.write(content)
            out.write('\n\n')

def download_python_repo(repo_url, output_file):
    def is_python_file(file_path):
        """ Check if the file is a python file """
        return file_path.endswith('.py')

    def is_likely_useful_file(file_path):
        """Determine if the file is likely to be useful by excluding certain directories 
        and specific file types."""
        excluded_dirs = ["docs", "examples", "tests", "test", "__pycache__", "scripts", "utils", "benchmarks"]
        utility_or_config_files = ["hubconf.py", "setup.py"]
        github_workflow_or_docs = ["stale.py", "gen-card-", "write_model_card"]

        if any(part.startswith('.') for part in file_path.split('/')):
            return False
        
        if 'test' in file_path.lower():
            return False
        
        for excluded_dir in excluded_dirs:
            if f"/{excluded_dir}/" in file_path or file_path.startswith(excluded_dir + "/"):
                return False
        for file_name in utility_or_config_files:
            if file_name in file_path:
                return False
        for doc_file in github_workflow_or_docs:
            if doc_file in file_path:
                return False
        return True

    def is_test_file(file_content):
        """Determine if the file content suggests it is a test file."""
        test_file_indicators = ["import unittest", "import pytest", "import mock", "from unittest", "from pytest"]
        return any(indicator in file_content for indicator in test_file_indicators)

    def remove_comments_and_docstrings(source):
        """Remove comments and docstrings from the Python source code."""
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and ast.get_docstring(node):
                node.body = node.body[1:]  # Remove docstring
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                node.value.s = ""  # Remove comments
        return ast.unparse(tree)

    python_rules = {
        "file_path_check": [is_python_file, is_likely_useful_file],
        "file_content_check": [lambda c: not is_test_file(c), has_sufficient_content],
        "transform": remove_comments_and_docstrings
    }

    download_repo(repo_url, output_file, python_rules)

def download_rust_repo(repo_url, output_file):
    def is_rust_file(file_path):
        """ Check if the file is a rust file """
        return file_path.endswith('.rs')

    def is_likely_useful_file(file_path):
        """Determine if the file is likely to be useful by excluding certain directories 
        and specific file types."""
        excluded_dirs = ["docs", "examples", "tests", "test", "scripts", "utils", "benchmarks"]
        utility_or_config_files = ["Cargo.toml", "Cargo.lock"]
        github_workflow_or_docs = ["stale.py", "gen-card-", "write_model_card"]

        if any(part.startswith('.') for part in file_path.split('/')):
            return False
        
        if 'test' in file_path.lower():
            return False
        
        for excluded_dir in excluded_dirs:
            if f"/{excluded_dir}/" in file_path or file_path.startswith(excluded_dir + "/"):
                return False
        for file_name in utility_or_config_files:
            if file_name in file_path:
                return False
        for doc_file in github_workflow_or_docs:
            if doc_file in file_path:
                return False
        return True

    rust_rules = {
        "file_path_check": [is_rust_file, is_likely_useful_file],
        "file_content_check": [has_sufficient_content],
        "transform": identity_transform
    }

    download_repo(repo_url, output_file, rust_rules)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python gh2file.py <github_url> <output_file>')
        sys.exit(1)

    repo_url = sys.argv[1]
    repo_name = repo_url.split('/')[-1]
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        output_file = repo_name + '.txt'

    if os.path.exists(output_file):
        output_file = output_file.split('.')[0] + '_1.txt'

    download_python_repo(repo_url, output_file)