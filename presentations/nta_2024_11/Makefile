.PHONY: all clean

all: main.pdf

main.pdf: main.tex policyengine.sty
	pdflatex -jobname=nta_2024_slides main || (echo "LaTeX failed. Check if all required packages are installed." && exit 1)
	pdflatex -jobname=nta_2024_slides main || (echo "LaTeX failed on second run." && exit 1)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.pdf