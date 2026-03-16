"""
Convert Jupyter Book markdown files to LaTeX for paper inclusion.

This script enables maintaining a single source of truth for content
that appears in both the online Jupyter Book and the LaTeX paper.
"""

import re
import os
from pathlib import Path


def convert_markdown_to_latex(markdown_content: str) -> str:
    """
    Convert markdown content to LaTeX format.

    Args:
        markdown_content: Markdown text to convert

    Returns:
        str: LaTeX formatted text
    """
    latex = markdown_content

    # Convert headers
    latex = re.sub(r"^# (.+)$", r"\\section{\1}", latex, flags=re.MULTILINE)
    latex = re.sub(
        r"^## (.+)$", r"\\subsection{\1}", latex, flags=re.MULTILINE
    )
    latex = re.sub(
        r"^### (.+)$", r"\\subsubsection{\1}", latex, flags=re.MULTILINE
    )

    # Convert bold and italic
    latex = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", latex)
    latex = re.sub(r"\*(.+?)\*", r"\\textit{\1}", latex)

    # Convert code blocks
    latex = re.sub(
        r"```python\n(.+?)\n```",
        r"\\begin{lstlisting}[language=Python]\n\1\n\\end{lstlisting}",
        latex,
        flags=re.DOTALL,
    )
    latex = re.sub(
        r"```\n(.+?)\n```",
        r"\\begin{verbatim}\n\1\n\\end{verbatim}",
        latex,
        flags=re.DOTALL,
    )

    # Convert inline code
    latex = re.sub(r"`(.+?)`", r"\\texttt{\1}", latex)

    # Convert lists
    lines = latex.split("\n")
    new_lines = []
    in_list = False
    list_stack = []

    for line in lines:
        # Handle unordered lists
        if re.match(r"^(\s*)-\s+(.+)$", line):
            match = re.match(r"^(\s*)-\s+(.+)$", line)
            indent_level = len(match.group(1)) // 2
            content = match.group(2)

            # Manage list stack
            while len(list_stack) > indent_level + 1:
                new_lines.append(
                    "  " * (len(list_stack) - 1) + "\\end{itemize}"
                )
                list_stack.pop()

            if len(list_stack) <= indent_level:
                new_lines.append("  " * indent_level + "\\begin{itemize}")
                list_stack.append("itemize")

            new_lines.append("  " * (indent_level + 1) + f"\\item {content}")
            in_list = True
        else:
            # Close any open lists
            while list_stack:
                new_lines.append(
                    "  " * (len(list_stack) - 1) + "\\end{itemize}"
                )
                list_stack.pop()
            new_lines.append(line)
            in_list = False

    # Close any remaining lists
    while list_stack:
        new_lines.append("  " * (len(list_stack) - 1) + "\\end{itemize}")
        list_stack.pop()

    latex = "\n".join(new_lines)

    # Convert links
    latex = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\\url{\2})", latex)

    # Escape special characters
    special_chars = ["$", "%", "&", "#", "_"]
    for char in special_chars:
        if char not in ["_"]:  # Don't escape underscore in commands
            latex = re.sub(f"\\{char}", f"\\\\{char}", latex)

    # Handle dollar amounts specially
    latex = re.sub(r"\\\$(\d+(?:\.\d+)?[BMK]?)", r"\\$\1", latex)

    return latex


def extract_content_sections(markdown_file: Path) -> dict:
    """
    Extract specific sections from markdown that should go in the paper.

    Args:
        markdown_file: Path to markdown file

    Returns:
        dict: Sections extracted from the file
    """
    with open(markdown_file, "r") as f:
        content = f.read()

    sections = {}

    # Extract sections based on headers
    section_pattern = r"^(#{1,3})\s+(.+?)$\n(.*?)(?=^#{1,3}\s|\Z)"
    matches = re.finditer(section_pattern, content, re.MULTILINE | re.DOTALL)

    for match in matches:
        level = len(match.group(1))
        title = match.group(2)
        body = match.group(3).strip()

        # Normalize section names
        section_key = title.lower().replace(" ", "_")
        sections[section_key] = {
            "level": level,
            "title": title,
            "content": body,
            "latex": convert_markdown_to_latex(body),
        }

    return sections


def sync_jupyter_book_to_paper():
    """
    Sync content from Jupyter Book markdown files to LaTeX paper sections.
    """
    docs_dir = Path("docs")
    paper_sections_dir = Path("paper/sections")

    # Mapping of Jupyter Book files to paper sections
    file_mappings = {
        "intro.md": "abstract.tex",
        "methodology.md": "methodology/overview.tex",
    }

    for md_file, tex_file in file_mappings.items():
        md_path = docs_dir / md_file
        tex_path = paper_sections_dir / tex_file

        if md_path.exists():
            print(f"Converting {md_file} to {tex_file}")

            with open(md_path, "r") as f:
                markdown_content = f.read()

            # Convert to LaTeX
            latex_content = convert_markdown_to_latex(markdown_content)

            # Add LaTeX document structure if needed
            if "abstract" in tex_file:
                latex_content = (
                    f"\\begin{{abstract}}\n{latex_content}\n\\end{{abstract}}"
                )

            # Save LaTeX file
            tex_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tex_path, "w") as f:
                f.write(latex_content)

            print(f"  Saved to {tex_path}")


def main():
    """Main entry point."""
    print("Syncing Jupyter Book content to LaTeX paper...")
    sync_jupyter_book_to_paper()
    print("Done!")


if __name__ == "__main__":
    main()
