# Exploring CNNs through Topological Data Analysis

## Overview

This repository contains the LaTeX source for the bachelor's thesis "Exploring Convolutional Neural Networks through Topological Data Analysis". The thesis investigates the integration of Topological Data Analysis (TDA) with convolutional neural networks (CNNs) to enhance our understanding of how CNNs process and manipulate data.

**NOTE:** The thesis is written in Spanish.

## Author

[Pablo Olivares Martínez](mailto:pablolivares1502@gmail.com)

## Abstract

This Bachelor's Thesis explores the integration of Topological Data Analysis (TDA) with convolutional neural networks (CNNs) to clarify and enhance our understanding of how CNNs manipulate data. By applying persistent homology techniques, a key tool in TDA, this work provides a detailed analysis of the data structure during CNN processing, offering greater transparency and understanding of these networks' internal workings from a topological perspective.

The study demonstrates that topological regularization not only improves the performance of CNNs in image classification and transfer learning tasks but also offers new insights into the data structure throughout the learning process. Implementations are carried out using advanced network architectures such as ResNet, DenseNet, and EfficientNet.

## Repository Structure

- `capitulos/`: Contains individual chapter files
- `img/`: Stores images and diagrams used in the thesis
- `preliminares/`: Includes preliminary sections like introduction and abstract
- `scripts/`: Contains Python scripts for generating plots
- `tfg.tex`: The main LaTeX document
- `library.bib`: Bibliography file

## Compiling the Thesis

Ensure you have a LaTeX distribution installed (e.g., [TeX Live](https://www.tug.org/texlive/), [MiKTeX](https://miktex.org/)). Then run:

```bash
pdflatex tfg.tex
bibtex tfg
pdflatex tfg.tex
pdflatex tfg.tex
```

This will generate `tfg.pdf`, which is the compiled thesis.

## Related Code Repository

The code implementation for this thesis is available in a separate repository: [tda-nn-analysis](https://github.com/pab1s/tda-nn-analysis)

This repository contains the Python code for:
- Implementing TDA techniques
- CNN models (ResNet, EfficientNet, DenseNet)
- Experiments and analysis scripts

Please refer to the README in the code repository for detailed instructions on setting up and running the experiments.

## Contact

For any queries regarding this thesis, please write me to this email: [pablolivares1502@gmail.com](mailto:pablolivares1502@gmail.com).
