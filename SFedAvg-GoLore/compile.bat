@echo off
echo Compiling LaTeX document...
pdflatex SFedAvg-GoLore.tex
bibtex SFedAvg-GoLore
pdflatex SFedAvg-GoLore.tex
pdflatex SFedAvg-GoLore.tex
echo Compilation complete!
pause