"""
Build both Jupyter Book pages and LaTeX paper sections from unified content.

This ensures perfect content alignment between the two formats.
"""

import re
from pathlib import Path
import yaml


class ContentConverter:
    """Convert markdown content to different output formats."""
    
    def __init__(self, content_dir="content", paper_dir="paper/sections", docs_dir="docs"):
        self.content_dir = Path(content_dir)
        self.paper_dir = Path(paper_dir)
        self.docs_dir = Path(docs_dir)
    
    def md_to_latex(self, content, section_type="section"):
        """Convert markdown to LaTeX format."""
        latex = content
        
        # Convert headers based on section type
        if section_type == "abstract":
            # Abstract doesn't need section header, already wrapped
            latex = re.sub(r'^# Abstract\n\n', '', latex)
        else:
            # Convert markdown headers to LaTeX sections
            latex = re.sub(r'^# (.+)$', r'\\section{\1}', latex, flags=re.MULTILINE)
            latex = re.sub(r'^## (.+)$', r'\\subsection{\1}', latex, flags=re.MULTILINE)
            latex = re.sub(r'^### (.+)$', r'\\subsubsection{\1}', latex, flags=re.MULTILINE)
        
        # Convert emphasis
        latex = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', latex)
        latex = re.sub(r'\*(.+?)\*', r'\\textit{\1}', latex)
        
        # Convert inline code
        latex = re.sub(r'`([^`]+)`', r'\\texttt{\1}', latex)
        
        # Convert lists
        lines = latex.split('\n')
        new_lines = []
        in_list = False
        list_stack = []
        
        for i, line in enumerate(lines):
            # Handle numbered lists
            if re.match(r'^\d+\.\s+', line):
                if not in_list or list_stack[-1] != 'enumerate':
                    new_lines.append('\\begin{enumerate}')
                    list_stack.append('enumerate')
                    in_list = True
                content = re.sub(r'^\d+\.\s+', '', line)
                new_lines.append(f'\\item {content}')
            # Handle unordered lists
            elif re.match(r'^-\s+', line):
                if not in_list or list_stack[-1] != 'itemize':
                    new_lines.append('\\begin{itemize}')
                    list_stack.append('itemize')
                    in_list = True
                content = re.sub(r'^-\s+', '', line)
                new_lines.append(f'\\item {content}')
            else:
                # Close any open lists if we hit a non-list line
                if in_list and line.strip() == '':
                    while list_stack:
                        list_type = list_stack.pop()
                        new_lines.append(f'\\end{{{list_type}}}')
                    in_list = False
                new_lines.append(line)
        
        # Close any remaining open lists
        while list_stack:
            list_type = list_stack.pop()
            new_lines.append(f'\\end{{{list_type}}}')
        
        latex = '\n'.join(new_lines)
        
        # Handle special characters
        latex = latex.replace('$', '\\$')
        latex = latex.replace('%', '\\%')
        latex = latex.replace('&', '\\&')
        latex = latex.replace('#', '\\#')
        latex = latex.replace('_', '\\_')
        
        # Fix dollar amounts
        latex = re.sub(r'\\\$(\d+(?:\.\d+)?[BMK]?)', r'\\$\1', latex)
        
        # Convert links (basic - just show text and URL)
        latex = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', latex)
        
        return latex
    
    def process_content_file(self, content_file):
        """Process a single content file to both formats."""
        # Read content
        with open(content_file, 'r') as f:
            content = f.read()
        
        # Extract metadata if present (YAML frontmatter)
        metadata = {}
        if content.startswith('---'):
            end = content.find('---', 3)
            if end != -1:
                metadata = yaml.safe_load(content[3:end])
                content = content[end+3:].strip()
        
        # Determine output paths and types
        stem = content_file.stem
        
        # LaTeX conversion
        if stem == 'abstract':
            latex_content = self.md_to_latex(content, section_type='abstract')
            latex_content = f"\\begin{{abstract}}\n{latex_content}\n\\end{{abstract}}"
            latex_path = self.paper_dir / "abstract.tex"
        elif stem == 'introduction':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "introduction.tex"
        elif stem == 'data':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "data.tex"
        elif stem == 'methodology':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "methodology.tex"
        elif stem == 'results':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "results.tex"
        elif stem == 'discussion':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "discussion.tex"
        elif stem == 'conclusion':
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "conclusion.tex"
        else:
            print(f"Skipping unknown content file: {stem}")
            return
        
        # Write LaTeX file
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        print(f"Generated LaTeX: {latex_path}")
        
        # Copy to Jupyter Book (markdown stays as-is)
        docs_path = self.docs_dir / f"{stem}.md"
        
        # For Jupyter Book, we might want to skip the abstract
        if stem != 'abstract':
            with open(docs_path, 'w') as f:
                # Add any JB-specific frontmatter if needed
                if metadata:
                    f.write('---\n')
                    yaml.dump(metadata, f)
                    f.write('---\n\n')
                f.write(content)
            print(f"Generated JB page: {docs_path}")
    
    def build_all(self):
        """Process all content files."""
        print("Building paper and Jupyter Book from unified content...")
        print("=" * 60)
        
        # Process each markdown file in content directory
        for content_file in sorted(self.content_dir.glob("*.md")):
            print(f"\nProcessing {content_file.name}...")
            self.process_content_file(content_file)
        
        print("\nContent conversion complete!")
        print("\nNext steps:")
        print("1. Run 'make paper' to build PDF")
        print("2. Run 'make documentation' to build Jupyter Book")


def main():
    """Build both formats from unified content."""
    converter = ContentConverter()
    converter.build_all()


if __name__ == "__main__":
    main()