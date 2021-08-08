draft:	out/note.pdf
.PHONY: draft

out/note.pdf:	src/note.tex
	./gutenbot.sh pdflatex \
		-interaction nonstopmode \
		-output-directory /work/out \
		/work/$<

