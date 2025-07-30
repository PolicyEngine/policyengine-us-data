"""
Build both Jupyter Book pages and LaTeX paper sections from unified content.

This ensures perfect content alignment between the two formats.
"""

import re
from pathlib import Path
import yaml


class ContentConverter:
    """Convert markdown content to different output formats."""

    def __init__(
        self,
        content_dir="content",
        paper_dir="paper/sections",
        docs_dir="docs",
    ):
        self.content_dir = Path(content_dir)
        self.paper_dir = Path(paper_dir)
        self.docs_dir = Path(docs_dir)

    def md_to_latex(self, content, section_type="section"):
        """Convert markdown to LaTeX format."""
        latex = content

        # Remove MyST admonitions and directives
        # Remove admonition blocks
        latex = re.sub(r"```{admonition}[^`]*```", "", latex, flags=re.DOTALL)
        # Remove card blocks
        latex = re.sub(r":::{card}[^:]*:::", "", latex, flags=re.DOTALL)
        # Remove mermaid diagrams
        latex = re.sub(
            r"```{mermaid}[^`]*```",
            "[Diagram omitted - see online version]",
            latex,
            flags=re.DOTALL,
        )
        # Remove tab-set blocks
        latex = re.sub(r"::::{tab-set}.*?::::", "", latex, flags=re.DOTALL)

        # Convert headers based on section type
        if section_type == "abstract":
            # Abstract doesn't need section header, already wrapped
            latex = re.sub(r"^# Abstract\n\n", "", latex)
        else:
            # Convert markdown headers to LaTeX sections
            latex = re.sub(
                r"^# (.+)$", r"\\section{\1}", latex, flags=re.MULTILINE
            )
            latex = re.sub(
                r"^## (.+)$", r"\\subsection{\1}", latex, flags=re.MULTILINE
            )
            latex = re.sub(
                r"^### (.+)$",
                r"\\subsubsection{\1}",
                latex,
                flags=re.MULTILINE,
            )

        # Convert emphasis
        latex = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", latex)
        latex = re.sub(r"\*(.+?)\*", r"\\textit{\1}", latex)

        # Convert inline code
        latex = re.sub(r"`([^`]+)`", r"\\texttt{\1}", latex)

        # Convert lists
        lines = latex.split("\n")
        new_lines = []
        in_list = False
        list_stack = []

        for i, line in enumerate(lines):
            # Handle numbered lists
            if re.match(r"^\d+\.\s+", line):
                if not in_list or list_stack[-1] != "enumerate":
                    new_lines.append("\\begin{enumerate}")
                    list_stack.append("enumerate")
                    in_list = True
                content = re.sub(r"^\d+\.\s+", "", line)
                new_lines.append(f"\\item {content}")
            # Handle unordered lists
            elif re.match(r"^-\s+", line):
                if not in_list or list_stack[-1] != "itemize":
                    new_lines.append("\\begin{itemize}")
                    list_stack.append("itemize")
                    in_list = True
                content = re.sub(r"^-\s+", "", line)
                new_lines.append(f"\\item {content}")
            else:
                # Close any open lists if we hit a non-list line
                if in_list and line.strip() == "":
                    while list_stack:
                        list_type = list_stack.pop()
                        new_lines.append(f"\\end{{{list_type}}}")
                    in_list = False
                new_lines.append(line)

        # Close any remaining open lists
        while list_stack:
            list_type = list_stack.pop()
            new_lines.append(f"\\end{{{list_type}}}")

        latex = "\n".join(new_lines)

        # Handle special characters
        latex = latex.replace("$", "\\$")
        latex = latex.replace("%", "\\%")
        latex = latex.replace("&", "\\&")
        latex = latex.replace("#", "\\#")
        latex = latex.replace("_", "\\_")

        # Fix dollar amounts
        latex = re.sub(r"\\\$(\d+(?:\.\d+)?[BMK]?)", r"\\$\1", latex)

        # Handle Unicode characters
        latex = latex.replace("∈", " \\in ")
        latex = latex.replace("×", " \\times ")
        latex = latex.replace("Σ", "\\Sigma")

        # Convert citations from (Author, Year) to \citep{author_year}
        # Handle multiple authors and et al.
        def convert_citation(match):
            citation = match.group(1)
            # Remove "and" between authors
            citation = citation.replace(" and ", " ")
            # Extract year
            year_match = re.search(r"(\d{4})", citation)
            if year_match:
                year = year_match.group(1)
                # Extract author(s)
                authors = citation[: citation.rfind(year)].strip().rstrip(",")

                # Handle different citation formats
                if "et al." in authors:
                    # Extract first author only for et al. citations
                    first_author = authors.split()[0]
                    cite_key = f"{first_author.lower()}{year}"
                elif "Congressional Budget Office" in authors:
                    cite_key = f"cbo{year}"
                elif "Joint Committee on Taxation" in authors:
                    cite_key = f"jct{year}"
                elif "Office of Tax Analysis" in authors:
                    cite_key = f"ota{year}"
                elif (
                    "Rothbaum and Bee" in authors
                    or "Rothbaum, Bee" in authors
                    or "Bee," in authors
                ):
                    cite_key = f"rothbaum{year}"
                elif "Sabelhaus" in authors:
                    cite_key = f"sabelhaus{year}"
                elif "Meyer" in authors:
                    cite_key = f"meyer{year}"
                elif "Tax Policy Center" in authors:
                    cite_key = f"tpc{year}"
                elif "Tax Foundation" in authors:
                    cite_key = f"tf{year}"
                elif "Penn Wharton Budget Model" in authors:
                    cite_key = f"pwbm{year}"
                elif "Institute on Taxation and Economic Policy" in authors:
                    cite_key = f"itep{year}"
                elif "Yale Budget Lab" in authors:
                    cite_key = f"budgetlab{year}"
                elif "Policy Simulation Library" in authors:
                    cite_key = f"psl{year}"
                else:
                    # Handle single or multiple authors
                    author_list = authors.split(",")
                    if len(author_list) == 1:
                        # Handle "Author1 and Author2" format
                        if " and " in authors:
                            first_author = (
                                authors.split(" and ")[0].strip().split()[-1]
                            )
                            cite_key = f"{first_author.lower()}{year}"
                        else:
                            # Single author
                            author = (
                                author_list[0].strip().split()[-1]
                            )  # Last name
                            cite_key = f"{author.lower()}{year}"
                    else:
                        # Multiple authors - use first author
                        first_author = author_list[0].strip().split()[-1]
                        cite_key = f"{first_author.lower()}{year}"

                return f"\\citep{{{cite_key}}}"
            return match.group(0)  # Return original if no year found

        latex = re.sub(
            r"\(([^)]+(?:19|20)\d{2}[a-z]?)\)", convert_citation, latex
        )

        # Also handle inline citations like "Author (Year)" or "Author et al. (Year)"
        def convert_inline_citation(match):
            author = match.group(1)
            year = match.group(2)

            # Determine citation key based on author format
            if "et al." in author:
                first_author = author.split()[0]
                cite_key = f"{first_author.lower()}{year}"
            else:
                # Single author case
                cite_key = f"{author.lower()}{year}"

            return f"\\citet{{{cite_key}}}"

        # Handle "Author (Year)" and "Author et al. (Year)" patterns
        latex = re.sub(
            r"(\w+(?:\s+et\s+al\.)?)\s*\((\d{4})\)",
            convert_inline_citation,
            latex,
        )

        # Convert links (basic - just show text and URL)
        latex = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", latex)

        return latex

    def convert_to_myst_citations(self, content):
        """Convert citations to MyST/Jupyter Book 2 format."""
        myst = content

        # Convert parenthetical citations (Author, Year) to {cite}`author_year`
        def convert_myst_citation(match):
            citation = match.group(1)
            # Remove "and" between authors
            citation = citation.replace(" and ", " ")
            # Extract year
            year_match = re.search(r"(\d{4})", citation)
            if year_match:
                year = year_match.group(1)
                # Extract author(s)
                authors = citation[: citation.rfind(year)].strip().rstrip(",")

                # Generate citation key using same logic as LaTeX conversion
                if "et al." in authors:
                    first_author = authors.split()[0]
                    cite_key = f"{first_author.lower()}{year}"
                elif "Congressional Budget Office" in authors:
                    cite_key = f"cbo{year}"
                elif "Joint Committee on Taxation" in authors:
                    cite_key = f"jct{year}"
                elif "Office of Tax Analysis" in authors:
                    cite_key = f"ota{year}"
                elif (
                    "Rothbaum and Bee" in authors
                    or "Rothbaum, Bee" in authors
                    or "Bee," in authors
                ):
                    cite_key = f"rothbaum{year}"
                elif "Sabelhaus" in authors:
                    cite_key = f"sabelhaus{year}"
                elif "Meyer" in authors:
                    cite_key = f"meyer{year}"
                elif "Tax Policy Center" in authors:
                    cite_key = f"tpc{year}"
                elif "Tax Foundation" in authors:
                    cite_key = f"tf{year}"
                elif "Penn Wharton Budget Model" in authors:
                    cite_key = f"pwbm{year}"
                elif "Institute on Taxation and Economic Policy" in authors:
                    cite_key = f"itep{year}"
                elif "Yale Budget Lab" in authors:
                    cite_key = f"budgetlab{year}"
                elif "Policy Simulation Library" in authors:
                    cite_key = f"psl{year}"
                else:
                    # Handle single or multiple authors
                    author_list = authors.split(",")
                    if len(author_list) == 1:
                        # Handle "Author1 and Author2" format
                        if " and " in authors:
                            first_author = (
                                authors.split(" and ")[0].strip().split()[-1]
                            )
                            cite_key = f"{first_author.lower()}{year}"
                        else:
                            # Single author
                            author = (
                                author_list[0].strip().split()[-1]
                            )  # Last name
                            cite_key = f"{author.lower()}{year}"
                    else:
                        # Multiple authors - use first author
                        first_author = author_list[0].strip().split()[-1]
                        cite_key = f"{first_author.lower()}{year}"

                return f"{{cite}}`{cite_key}`"
            return match.group(0)

        myst = re.sub(
            r"\(([^)]+(?:19|20)\d{2}[a-z]?)\)", convert_myst_citation, myst
        )

        # Handle inline citations like "Author (Year)" - convert to {cite:t}`author_year`
        def convert_inline_myst(match):
            author = match.group(1)
            year = match.group(2)

            # Determine citation key based on author format
            if "et al." in author:
                first_author = author.split()[0]
                cite_key = f"{first_author.lower()}{year}"
            else:
                # Single author case
                cite_key = f"{author.lower()}{year}"

            return f"{{cite:t}}`{cite_key}`"

        # Handle "Author (Year)" and "Author et al. (Year)" patterns
        myst = re.sub(
            r"(\w+(?:\s+et\s+al\.)?)\s*\((\d{4})\)", convert_inline_myst, myst
        )

        return myst

    def process_content_file(self, content_file):
        """Process a single content file to both formats."""
        # Read content
        with open(content_file, "r") as f:
            content = f.read()

        # Extract metadata if present (YAML frontmatter)
        metadata = {}
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                metadata = yaml.safe_load(content[3:end])
                content = content[end + 3 :].strip()

        # For Jupyter Book output, convert citations to MyST format
        jb_content = self.convert_to_myst_citations(content)

        # Determine output paths and types
        stem = content_file.stem

        # LaTeX conversion
        if stem == "abstract":
            latex_content = self.md_to_latex(content, section_type="abstract")
            latex_content = (
                f"\\begin{{abstract}}\n{latex_content}\n\\end{{abstract}}"
            )
            latex_path = self.paper_dir / "abstract.tex"
        elif stem == "introduction":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "introduction.tex"
        elif stem == "background":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "background.tex"
        elif stem == "data":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "data.tex"
        elif stem == "methodology":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "methodology.tex"
        elif stem == "results":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "results.tex"
        elif stem == "discussion":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "discussion.tex"
        elif stem == "conclusion":
            latex_content = self.md_to_latex(content)
            latex_path = self.paper_dir / "conclusion.tex"
        else:
            print(f"Skipping unknown content file: {stem}")
            return

        # Write LaTeX file
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(latex_path, "w") as f:
            f.write(latex_content)
        print(f"Generated LaTeX: {latex_path}")

        # Copy to Jupyter Book (markdown stays as-is)
        docs_path = self.docs_dir / f"{stem}.md"

        # For Jupyter Book, we might want to skip the abstract
        if stem != "abstract":
            with open(docs_path, "w") as f:
                # Add any JB-specific frontmatter if needed
                if metadata:
                    f.write("---\n")
                    yaml.dump(metadata, f)
                    f.write("---\n\n")
                f.write(jb_content)
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
