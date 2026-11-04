# Paper

`main.tex` is the skeleton of the Ising-only preprint and workshop version; `outline.md` is the
section plan it follows. The ICLR 2027 style files (`iclr2027_conference.sty`,
`iclr2027_conference.bst`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex`) are unmodified copies
from the `iclr2027` folder of https://github.com/ICLR/Master-Template at commit `46ed6f4`.

## Building the PDF

With a TeX distribution installed (TeX Live or MiKTeX, both ship `latexmk`), run this from `paper/`:

```sh
latexmk -pdf main.tex
```

`latexmk` runs pdfLaTeX and BibTeX as many times as the references need. The build outputs are
ignored by git. On Overleaf, upload the contents of `paper/` (keeping `figures/`), set the main
document to `main.tex` and the compiler to pdfLaTeX.

The file is in its preprint form. The comment at the top of `main.tex` explains how to switch to the
anonymous submission form.
