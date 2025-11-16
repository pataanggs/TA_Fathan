# Latexmkrc configuration for TeX Live 2025
# This ensures latexmk uses the correct TeX installation

# Set the TeX Live bin directory
$ENV{'PATH'} = '/home/pataangg/texlive/2025/bin/x86_64-linux:' . $ENV{'PATH'};

# PDF generation mode
$pdf_mode = 1;

# Use pdflatex
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Use bibtex for bibliography
$bibtex_use = 2;

# Output directory (optional, comment out if not needed)
# $out_dir = '.';

# Clean up extensions
@generated_exts = qw(aux bbl blg fdb_latexmk fls log out synctex.gz toc lot lof run.xml bcf);
