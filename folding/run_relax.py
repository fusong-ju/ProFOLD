#!/usr/bin/env python
import os
import click
import pyrosetta
from multiprocessing import Pool
from pyrosetta import pose_from_pdb
from constraints import Constraints
from minimizer import relax


def relex_from_pdb(seq, feature_path, input_pdb, output_pdb):
    pose = pose_from_pdb(input_pdb)
    raw_constraints = Constraints(seq, feature_path)
    constraints = raw_constraints.get_constraint_v1_fix_gly()
    constraints.apply(pose)
    relax(pose)
    pose.dump_pdb(output_pdb)


@click.command()
@click.option("-s", "--fasta_path", required=True, type=click.Path(exists=True))
@click.option("-f", "--feature_path", required=True, type=click.Path(exists=True))
@click.option("-i", "--input_dir", required=True, type=click.Path())
@click.option("-o", "--output_dir", required=True, type=click.Path())
@click.option("-nw", "--n_workers", default=24, type=int)
def main(fasta_path, feature_path, input_dir, output_dir, n_workers):
    pyrosetta.init(
        "-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100"
    )
    os.makedirs(output_dir, exist_ok=True)
    seq = open(fasta_path).readlines()[1].strip()
    with Pool(n_workers) as p:
        args = []
        for path in os.listdir(input_dir):
            args.append(
                (
                    seq,
                    feature_path,
                    os.path.join(input_dir, path),
                    os.path.join(output_dir, path),
                )
            )
        p.starmap(relex_from_pdb, args)


if __name__ == "__main__":
    main()
