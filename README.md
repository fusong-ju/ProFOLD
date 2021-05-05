# ProFOLD
## About The Project
The implementation of the paper "CopulaNet: Learning residue co-evolution
directly from multiple sequence alignment for protein structure prediction".

[Fusong Ju, Jianwei Zhu, Bin Shao, Lupeng Kong, Tie-Yan Liu, Wei-Mou Zheng and Dongbo Bu. CopulaNet: Learning residue co-evolution directly from multiple sequence alignment for protein structure prediction. Nature Communications. https://doi.org/10.1038/s41467-021-22869-8](https://www.nature.com/articles/s41467-021-22869-8)

## Getting Started
### Prerequisites
Install [PyTorch 1.4+](https://pytorch.org/),
[PyRosetta](http://www.pyrosetta.org/), [python
3.7+](https://www.python.org/downloads/)

### Installation

1. Clone the repo
```sh
git clone https://github.com/fusong-ju/ProFOLD.git
```

2. Install python packages
```sh
cd ProFOLD
pip install -r requirements.txt
```

## Usage
1. Generate `aln` format MSA for a given target sequence
2. Run ProFOLD
```sh
run_ProFOLD.sh <MSA> <output_dir>
```

## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
