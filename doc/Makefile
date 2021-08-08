draft:	src/note.pdf
.PHONY: draft


# ref https://tex.stackexchange.com/questions/40738/how-to-properly-make-a-latex-project
src/note.pdf:	src/note.tex
	./gutenbot.sh latexmk \
		-pdf \
		-pdflatex="pdflatex -interaction=nonstopmode" \
		-use-make note.tex
src/note.dvi:	src/note.tex
	./gutenbot.sh latexmk \
		-dvi \
		-use-make note.tex


clean:
	./gutenbot.sh latexmk -CA
.PHONY: clean
