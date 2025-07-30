.PHONY: all clean

all: sge_2025_slides.pdf

sge_2025_slides.pdf: main.tex policyengine.sty
	pdflatex -jobname=sge_2025_slides main || (echo "LaTeX failed. Check if all required packages are installed." && exit 1)
	pdflatex -jobname=sge_2025_slides main || (echo "LaTeX failed on second run." && exit 1)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.pdf
	# Keep main.tex and sge_2025_slides.pdf
	touch sge_2025_slides.pdf