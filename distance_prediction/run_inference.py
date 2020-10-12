#!/usr/bin/env python
import os
import click
import torch
import numpy as np


def load_models(model_dir):
    models = []
    for path in os.listdir(model_dir):
        if path.endswith(".pt"):
            models.append(torch.jit.load(os.path.join(model_dir, path)))
    return models


def parse_feature(aln_path):
    AMINO = "ACDEFGHIKLMNPQRSTVWY-XBZUOJ"
    msa = [line.strip() for line in open(aln_path) if not line.startswith(">")]
    msa = [[AMINO.index(_) for _ in line if _ in AMINO] for line in msa]
    msa = torch.tensor(msa).long()
    msa[msa >= 21] = 20
    return msa


def predict_single(models, aln_path, output_path):
    feat = parse_feature(aln_path)
    cbcb, omega, theta, phi = [], [], [], []
    with torch.no_grad():
        for model in models:
            a, b, c, d = model(feat)
            cbcb.append(a.cpu().numpy())
            omega.append(b.cpu().numpy())
            theta.append(c.cpu().numpy())
            phi.append(d.cpu().numpy())
    np.savez(
        output_path,
        cbcb=np.mean(cbcb, axis=0),
        omega=np.mean(omega, axis=0),
        theta=np.mean(theta, axis=0),
        phi=np.mean(phi, axis=0),
    )


@click.command()
@click.option("-m", "--model_dir", required=True, type=click.Path())
@click.option("-i", "--aln_path", required=True, type=click.Path())
@click.option("-o", "--output_path", required=True, type=click.Path())
def main(model_dir, aln_path, output_path):
    """
    predict from a *.aln file
    """
    models = load_models(model_dir)
    predict_single(models, aln_path, output_path)


if __name__ == "__main__":
    main()
