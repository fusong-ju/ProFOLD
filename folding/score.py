import pyrosetta
from pyrosetta import ScoreFunction


def geo_sf(dist_weight=5, dihedral_weight=1, angle_weight=1):
    sf = ScoreFunction()
    sf.set_weight(pyrosetta.rosetta.core.scoring.atom_pair_constraint, dist_weight)
    sf.set_weight(pyrosetta.rosetta.core.scoring.dihedral_constraint, dihedral_weight)
    sf.set_weight(pyrosetta.rosetta.core.scoring.angle_constraint, angle_weight)
    return sf


def score_it(sf, pose):
    return sf(pose)
