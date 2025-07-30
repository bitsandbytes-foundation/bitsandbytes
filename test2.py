#!/usr/bin/env python3
"""
AST-based script to extract all classes, functions, and methods from Python files
and save documentation brief to docs/brief/docs_brief.md
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class CodeVisitor(ast.NodeVisitor):
    """AST visitor to extract function and class definitions"""
    
    def __init__(self):
        self.elements = []
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definitions"""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                # Handle cases like module.ClassName
                bases.append(ast.unparse(base))
        
        signature = f"{node.name}({', '.join(bases)})" if bases else f"{node.name}()"
        
        self.elements.append({
            'name': node.name,
            'signature': signature,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'type': 'class'
        })
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definitions"""
        self._process_function(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definitions"""
        self._process_function(node, is_async=True)
    
    def _process_function(self, node: Any, is_async: bool = False):
        """Process function or async function"""
        params = []
        
        # Add positional arguments
        for arg in node.args.args:
            params.append(arg.arg)
        
        # Add *args if present
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        
        # Add keyword-only arguments
        for arg in node.args.kwonlyargs:
            params.append(arg.arg)
        
        # Add **kwargs if present
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        
        signature = f"{node.name}({', '.join(params)})"
        
        self.elements.append({
            'name': node.name,
            'signature': signature,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'type': 'function'
        })
        
        self.generic_visit(node)


def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a Python file using AST"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source, filename=str(file_path))
        
        # Extract elements
        visitor = CodeVisitor()
        visitor.visit(tree)
        
        # Sort by line number
        visitor.elements.sort(key=lambda x: x['start_line'])
        
        return visitor.elements
        
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"⚠️  Error processing {file_path}: {e}", file=sys.stderr)
        return []


def find_python_files(root_path: Path) -> List[Path]:
    """Recursively find all Python files in directory"""
    python_files = []
    
    # Common directories to exclude
    exclude_dirs = {
        '__pycache__', '.git', '.venv', 'venv', 'env', 
        'node_modules', '.tox', '.pytest_cache', '.mypy_cache',
        'build', 'dist', '.eggs', '*.egg-info'
    }
    
    for file_path in root_path.rglob("*.py"):
        # Check if any parent directory should be excluded
        if not any(excluded in file_path.parts for excluded in exclude_dirs):
            python_files.append(file_path)
    
    return sorted(python_files)


def format_file_section(file_path: Path, elements: List[Dict[str, Any]], root_path: Path) -> str:
    """Format the output for a single file"""
    # Get relative path
    try:
        relative_path = file_path.relative_to(root_path)
    except ValueError:
        relative_path = file_path
    
    lines = []
    lines.append(f"## {relative_path}")
    lines.append("")
    
    if not elements:
        lines.append("*No classes or functions found*")
        lines.append("")
        return '\n'.join(lines)
    
    # Group elements by type
    classes = [e for e in elements if e['type'] == 'class']
    functions = [e for e in elements if e['type'] == 'function']
    
    # Add classes
    if classes:
        lines.append("### Classes")
        lines.append("")
        for element in classes:
            start = element['start_line']
            end = element['end_line']
            limit = end - start + 1
            lines.append(f"- `{element['signature']}` # Read(file_path={relative_path}, offset={start}, limit={limit})")
        lines.append("")
    
    # Add functions
    if functions:
        lines.append("### Functions")
        lines.append("")
        for element in functions:
            start = element['start_line']
            end = element['end_line']
            limit = end - start + 1
            lines.append(f"- `{element['signature']}` # Read(file_path={relative_path}, offset={start}, limit={limit})")
        lines.append("")
    
    return '\n'.join(lines)


def generate_docs_brief(root_path: Path, output_path: Path):
    """Generate documentation brief for all Python files"""
    print(f"🔍 Scanning Python files in: {root_path}")
    
    # Find all Python files
    python_files = find_python_files(root_path)
    
    if not python_files:
        print("❌ No Python files found!")
        return
    
    print(f"📁 Found {len(python_files)} Python files")
    
    # Process all files
    all_sections = []
    total_elements = 0
    
    # Add header
    header = [
        "# Documentation Brief",
        "",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Root directory: {root_path.absolute()}",
        f"Total files: {len(python_files)}",
        "",
        "---",
        ""
    ]
    
    all_sections.append('\n'.join(header))
    
    # Process each file
    for i, file_path in enumerate(python_files, 1):
        print(f"📄 Processing ({i}/{len(python_files)}): {file_path.relative_to(root_path)}")
        
        elements = process_file(file_path)
        total_elements += len(elements)
        
        if elements:
            section = format_file_section(file_path, elements, root_path)
            all_sections.append(section)
    
    # Add summary at the end
    summary = [
        "---",
        "",
        "## Summary",
        "",
        f"- **Total files scanned**: {len(python_files)}",
        f"- **Total code elements found**: {total_elements}",
        ""
    ]
    all_sections.append('\n'.join(summary))
    
    # Write to file
    output_content = '\n'.join(all_sections)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✅ Documentation brief written to: {output_path}")
    print(f"📊 Total elements extracted: {total_elements}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate documentation brief using AST analysis'
    )
    parser.add_argument(
        'path', 
        nargs='?', 
        default='.', 
        help='Path to Python project root (default: current directory)'
    )
    parser.add_argument(
        '--output', 
        '-o',
        default='docs/brief/docs_brief.md',
        help='Output file path (default: docs/brief/docs_brief.md)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    root_path = Path(args.path).resolve()
    output_path = Path(args.output)
    
    # Validate root path
    if not root_path.exists():
        print(f"❌ Error: Path '{root_path}' does not exist")
        sys.exit(1)
    
    if not root_path.is_dir():
        print(f"❌ Error: Path '{root_path}' is not a directory")
        sys.exit(1)
    
    # Generate documentation brief
    generate_docs_brief(root_path, output_path)


if __name__ == "__main__":
    main()
