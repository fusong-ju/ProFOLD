import os
import threading
import queue
import numpy as np
from pyrosetta import (
    pose_from_sequence,
    MoveMap,
    rosetta,
    create_score_function,
    SwitchResidueTypeSetMover,
)
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from score import score_it


def _random_dihedral():
    phi = [-140, -72, -122, -82, -61, 57]
    psi = [153, 145, 117, -14, -41, 39]
    w = [0.135, 0.155, 0.073, 0.122, 0.497, 0.018]
    p = np.random.choice(range(6), p=w)
    return phi[p], psi[p]


def _set_random_dihedral(pose):
    for i in range(1, pose.total_residue()):
        phi, psi = _random_dihedral()
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)
        pose.set_omega(i, 180)


def _random_pose(seq, constraint):
    pose = pose_from_sequence(seq, "centroid")
    _set_random_dihedral(pose)
    constraint.apply(pose)
    return pose


def _add_noise(pose):
    for i in range(1, pose.total_residue()):
        phi = pose.phi(i) + np.random.normal(0, 60)
        psi = pose.psi(i) + np.random.normal(0, 60)
        pose.set_phi(i, phi)
        pose.set_psi(i, psi)


def _minimize_step(sf, pose):
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)

    min_mover = MinMover(mmap, sf, "lbfgs_armijo_nonmonotone", 0.0001, True)
    min_mover.max_iter(1000)
    min_mover.apply(pose)


def _worker(seq, constraint, sf, run_dir, pose_pool, pool_size, task_queue, mutex):
    while True:
        try:
            idx = task_queue.get(block=False)
            print("Start minimize %i ................." % idx)
            mutex.acquire()
            if len(pose_pool) < pool_size or np.random.random() < 0.1:
                pose = _random_pose(seq, constraint)
            else:
                p = np.random.randint(len(pose_pool))
                pose = pose_pool[p].clone()
                _add_noise(pose)
            mutex.release()
            _minimize_step(sf, pose)
            mutex.acquire()
            pose_pool.append(pose)
            if len(pose_pool) > pool_size:
                pose_pool.sort(key=lambda x: score_it(sf, x))
                del pose_pool[-1]
            print("Score %i: %f" % (idx, score_it(sf, pose)))
            mutex.release()

        except queue.Empty:
            break


def repeat_minimize(seq, constraints, sf, run_dir, n_workers, n_structs, n_iter):
    pose_pool = []
    mutex = threading.Lock()
    q = queue.Queue()
    for i in range(1, n_iter + 1):
        q.put(i)
    threads = []
    for i in range(n_workers):
        thread = threading.Thread(
            target=_worker,
            args=(seq, constraints, sf, run_dir, pose_pool, n_structs, q, mutex),
        )
        thread.start()
        threads.append(thread)
    for x in threads:
        x.join()
    pose_pool.sort(key=lambda x: score_it(sf, x))
    return pose_pool


def relax(pose):
    sf = create_score_function("ref2015")
    sf.set_weight(rosetta.core.scoring.atom_pair_constraint, 5)
    sf.set_weight(rosetta.core.scoring.dihedral_constraint, 1)
    sf.set_weight(rosetta.core.scoring.angle_constraint, 1)

    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.set_movemap(mmap)

    switch = SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    relax.apply(pose)
