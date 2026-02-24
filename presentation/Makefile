build:
	@echo "Building tex"
	@docker run --rm -v '/workspace/presentation:/tex' chaussebenjamin/tex-compiler:latest 'pdflatex -shell-escape 00-main.tex'