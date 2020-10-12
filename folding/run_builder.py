#!/usr/bin/env python
import os
import click
import pyrosetta
from pyrosetta import rosetta
from constraints import Constraints
from score import geo_sf
from minimizer import repeat_minimize


@click.command()
@click.option("-i", "--fasta_path", required=True, type=click.Path(exists=True))
@click.option("-f", "--feature_path", required=True, type=click.Path(exists=True))
@click.option("-o", "--output_dir", required=True, type=click.Path())
@click.option("-nw", "--n_workers", default=24, type=int)
@click.option("-ns", "--n_structs", default=20, type=int)
@click.option("-ni", "--n_iter", default=100, type=int)
def main(fasta_path, feature_path, output_dir, n_workers, n_structs, n_iter):
    pyrosetta.init(
        "-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100"
    )
    os.makedirs(output_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(fasta_path))[0]
    seq = open(fasta_path).readlines()[1].strip()
    seq_no_g = "".join(["A" if _ == "G" else _ for _ in list(seq)])
    raw_constraints = Constraints(seq, feature_path)
    constraints = raw_constraints.get_constraint_v1()
    score_function = geo_sf(dist_weight=5, dihedral_weight=1, angle_weight=1)
    poses = repeat_minimize(
        seq_no_g, constraints, score_function, output_dir, n_workers, n_structs, n_iter,
    )
    for i, a in enumerate(seq):
        if a == "G":
            mutator = rosetta.protocols.simple_moves.MutateResidue(i + 1, "GLY")
            for pose in poses:
                mutator.apply(pose)
    os.makedirs(os.path.join(output_dir, "final"), exist_ok=True)
    for i, pose in enumerate(poses):
        path = os.path.join(output_dir, "final", "%s_%02i.pdb" % (name, i))
        pose.dump_pdb(path)


if __name__ == "__main__":
    main()
