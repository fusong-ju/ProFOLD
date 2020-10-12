import tempfile
import numpy as np
import pyrosetta


class Constraints:
    def __init__(self, seq, feat_path):
        self._seq = seq
        self._feat = np.load(feat_path)
        self._tmp_dir = tempfile.TemporaryDirectory(prefix="/dev/shm/")

        self._raw_constraints = self._init_constraints()

    def _init_constraints(self):
        return {
            "cbcb": self._init_cbcb_constraints(),
            "omega": self._init_omega_constraints(),
            "theta": self._init_theta_constraints(),
            "phi": self._init_phi_constraints(),
        }

    def _init_cbcb_constraints(self):
        cbcb = self._feat["cbcb"]
        L = cbcb.shape[0]
        bins = np.linspace(2.25, 19.75, 36)
        contact_prob = np.sum(cbcb[:, :, :-1], axis=-1)
        ref = cbcb[:, :, -2:-1] * np.array((bins / bins[-1]) ** 1.57)[None, None]
        potential = -np.log(cbcb[:, :, :-1] / ref)
        bound_p = np.maximum(potential[:, :, :1], np.zeros((L, L, 1))) + 10
        potential = np.concatenate([bound_p, potential], axis=-1)
        bins = np.concatenate([[0], bins])
        idx, idy = np.where(contact_prob > 0.05)
        step = 0.5
        ret = []
        for i, j, p in zip(idx, idy, contact_prob[idx, idy]):
            if j > i:
                name = self._tmp_dir.name + "/%d.%d.txt" % (i + 1, j + 1)
                with open(name, "w") as f:
                    f.write("x_axis" + "\t%.3f" * len(bins) % tuple(bins) + "\n")
                    f.write(
                        "y_axis" + "\t%.3f" * len(bins) % tuple(potential[i, j]) + "\n"
                    )
                    line = "AtomPair %s %d %s %d SPLINE TAG %s 1.0 %.3f %.5f"
                    line = line % ("CB", i + 1, "CB", j + 1, name, 1.0, step)
                    ret.append([i, j, p, line])
        return ret

    def _init_omega_constraints(self):
        omega = self._feat["omega"]
        L = omega.shape[0]
        STEP = np.deg2rad(15)
        bins = np.linspace(-np.pi - 1.5 * STEP, np.pi + 1.5 * STEP, 24 + 4)
        contact_prob = np.sum(omega[:, :, :-1], axis=-1)
        omega = -np.log((omega[:, :, :-1] + 1e-4) / (omega[:, :, -2:-1] + 1e-4))
        omega = np.concatenate([omega[:, :, -2:], omega, omega[:, :, :2]], axis=-1)
        idx, idy = np.where(contact_prob > 0.05)
        ret = []
        for i, j, p in zip(idx, idy, contact_prob[idx, idy]):
            if j > i:
                name = self._tmp_dir.name + "/%d.%d_omega.txt" % (i + 1, j + 1)
                with open(name, "w") as f:
                    f.write("x_axis" + "\t%.5f" * len(bins) % tuple(bins) + "\n")
                    f.write("y_axis" + "\t%.5f" * len(bins) % tuple(omega[i, j]) + "\n")
                line = (
                    "Dihedral CA %d CB %d CB %d CA %d SPLINE TAG %s 1.0 %.3f %.5f"
                    % (i + 1, i + 1, j + 1, j + 1, name, 1.0, STEP)
                )
                ret.append([i, j, p, line])
        return ret

    def _init_theta_constraints(self):
        theta = self._feat["theta"]
        L = theta.shape[0]
        STEP = np.deg2rad(15)
        bins = np.linspace(-np.pi - 1.5 * STEP, np.pi + 1.5 * STEP, 24 + 4)
        contact_prob = np.sum(theta[:, :, :-1], axis=-1)
        theta = -np.log((theta[:, :, :-1] + 1e-4) / (theta[:, :, -2:-1] + 1e-4))
        theta = np.concatenate([theta[:, :, -2:], theta, theta[:, :, :2]], axis=-1)
        idx, idy = np.where(contact_prob > 0.05)
        ret = []
        for i, j, p in zip(idx, idy, contact_prob[idx, idy]):
            if j != i:
                name = self._tmp_dir.name + "/%d.%d_theta.txt" % (i + 1, j + 1)
                with open(name, "w") as f:
                    f.write("x_axis" + "\t%.5f" * len(bins) % tuple(bins) + "\n")
                    f.write("y_axis" + "\t%.5f" * len(bins) % tuple(theta[i, j]) + "\n")
                line = (
                    "Dihedral N %d CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f "
                    % (i + 1, i + 1, i + 1, j + 1, name, 1.0, STEP,)
                )
                ret.append([i, j, p, line])
        return ret

    def _init_phi_constraints(self):
        phi = self._feat["phi"]
        L = phi.shape[0]
        STEP = np.deg2rad(15)
        bins = np.linspace(-1.5 * STEP, np.pi + 1.5 * STEP, 12 + 4)
        contact_prob = np.sum(phi[:, :, :-1], axis=-1)
        phi = -np.log((phi[:, :, :-1] + 1e-4) / (phi[:, :, -2:-1] + 1e-4))
        phi = np.concatenate(
            [np.flip(phi[:, :, :2], axis=-1), phi, np.flip(phi[:, :, -2:], axis=-1)],
            axis=-1,
        )
        idx, idy = np.where(contact_prob > 0.05)
        ret = []
        for i, j, p in zip(idx, idy, contact_prob[idx, idy]):
            if j != i:
                name = self._tmp_dir.name + "/%d.%d_phi.txt" % (i + 1, j + 1)
                with open(name, "w") as f:
                    f.write("x_axis" + "\t%.5f" * len(bins) % tuple(bins) + "\n")
                    f.write("y_axis" + "\t%.5f" * len(bins) % tuple(phi[i, j]) + "\n")
                line = (
                    "Angle CA %d CB %d CB %d SPLINE TAG %s 1.0 %.3f %.5f         "
                    % (i + 1, i + 1, j + 1, name, 1.0, STEP,)
                )
                ret.append([i, j, p, line])
        return ret

    def _make_constraint(self, a):
        np.random.shuffle(a)
        tmpname = self._tmp_dir.name + "/minimize.cst"
        with open(tmpname, "w") as f:
            for line in a:
                f.write(line + "\n")
        constraints = pyrosetta.rosetta.protocols.constraint_movers.ConstraintSetMover()
        constraints.constraint_file(tmpname)
        constraints.add_constraints(True)
        return constraints

    def get_constraint_v1(self):
        cbcb = [line for _, _, _, line in self._raw_constraints["cbcb"]]
        omega = [line for i, j, p, line in self._raw_constraints["omega"] if p > 0.6]
        theta = [line for i, j, p, line in self._raw_constraints["theta"] if p > 0.6]
        phi = [line for i, j, p, line in self._raw_constraints["phi"] if p > 0.7]
        for x, y in zip(["CBCB", "OMEGA", "THETA", "PHI"], [cbcb, omega, theta, phi]):
            print("%s constraints: %i" % (x, len(y)))
        return self._make_constraint(cbcb + omega + theta + phi)

    def get_constraint_v1_fix_gly(self):
        cbcb = [
            line
            for i, j, _, line in self._raw_constraints["cbcb"]
            if self._seq[i] != "G" and self._seq[j] != "G"
        ]

        omega = [
            line
            for i, j, p, line in self._raw_constraints["omega"]
            if self._seq[i] != "G" and self._seq[j] != "G" and p > 0.6
        ]
        theta = [
            line
            for i, j, p, line in self._raw_constraints["theta"]
            if p > 0.6 and self._seq[i] != "G" and self._seq[j] != "G"
        ]
        phi = [
            line
            for i, j, p, line in self._raw_constraints["phi"]
            if p > 0.7 and self._seq[i] != "G" and self._seq[j] != "G"
        ]
        for x, y in zip(["CBCB", "OMEGA", "THETA", "PHI"], [cbcb, omega, theta, phi]):
            print("%s constraints: %i" % (x, len(y)))
        return self._make_constraint(cbcb + omega + theta + phi)
