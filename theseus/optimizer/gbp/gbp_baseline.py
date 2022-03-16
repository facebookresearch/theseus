import numpy as np
from typing import Any, Dict, Optional, Type
from typing import List, Callable, Optional, Union


"""
Defines squared loss functions that correspond to Gaussians.
Robust losses are implemented by scaling the Gaussian covariance.
"""

class Gaussian:
    def __init__(
        self,
        dim,
        eta=None,
        lam=None,
    ):
        self.dim = dim

        if eta is not None and eta.shape == (dim,):
            self.eta = eta
        else:
            self.eta = np.zeros(dim)

        if lam is not None and lam.shape == (dim, dim):
            self.lam = lam
        else:
            self.lam = np.zeros([dim, dim])

    def mean(self) -> np.ndarray:
        return np.matmul(np.linalg.inv(self.lam), self.eta)

    def cov(self) -> np.ndarray:
        return np.linalg.inv(self.lam)

    def mean_and_cov(self) -> List[np.ndarray]:
        cov = self.cov()
        mean = np.matmul(cov, self.eta)
        return [mean, cov]

    def set_with_cov_form(self, mean: np.ndarray, cov: np.ndarray) -> None:
        self.lam = np.linalg.inv(cov)
        self.eta = np.matmul(self.lam, mean)


class GBPSettings:
    def __init__(
        self,
        damping: float = 0.,
        beta: float = 0.1,
        num_undamped_iters: int = 5,
        min_linear_iters: int = 10,
        dropout: float = 0.,
        reset_iters_since_relin: List[int] = [],
    ):
        # Parameters for damping the eta component of the message
        self.damping = damping
        # Number of undamped iterations after relin before damping is on
        self.num_undamped_iters = num_undamped_iters

        self.dropout = dropout

        # Parameters for just in time factor relinearisation.
        # Threshold absolute distance between linpoint
        # and adjacent belief means for relinearisation.
        self.beta = beta
        # Minimum number of linear iterations before
        # a factor is allowed to realinearise.
        self.min_linear_iters = min_linear_iters
        self.reset_iters_since_relin = reset_iters_since_relin

    def get_damping(self, iters_since_relin: int) -> float:
        if iters_since_relin > self.num_undamped_iters:
            return self.damping
        else:
            return 0.


class SquaredLoss():
    def __init__(
        self,
        dofs: int,
        diag_cov: Union[float, np.ndarray]
    ):
        """
        dofs: dofs of the measurement
        cov: diagonal elements of covariance matrix
        """
        assert diag_cov.shape[0] == dofs
        mat = np.zeros([dofs, dofs])
        mat[range(dofs), range(dofs)] = diag_cov
        self.cov = mat
        self.effective_cov = mat.copy()

    def get_effective_cov(self, residual: np.ndarray) -> None:
        """
        Returns the covariance of the Gaussian (squared loss)
        that matches the loss at the error value.
        """
        self.effective_cov = self.cov.copy()

    def robust(self) -> bool:
        return not np.equal(self.cov, self.effective_cov)


class HuberLoss(SquaredLoss):
    def __init__(
        self,
        dofs: int,
        diag_cov: Union[float, np.ndarray],
        stds_transition: float
    ):
        """
        stds_transition: num standard deviations from minimum at
            which quadratic loss transitions to linear.
        """
        SquaredLoss.__init__(self, dofs, diag_cov)
        self.stds_transition = stds_transition

    def get_effective_cov(self, residual: np.ndarray) -> None:
        energy = residual @ np.linalg.inv(self.cov) @ residual
        mahalanobis_dist = np.sqrt(energy)
        if mahalanobis_dist > self.stds_transition:
            denom = (2 * self.stds_transition * mahalanobis_dist - self.stds_transition ** 2)
            self.effective_cov = self.cov * mahalanobis_dist**2 / denom
        else:
            self.effective_cov = self.cov.copy()


class MeasModel:
    def __init__(
        self,
        meas_fn: Callable,
        jac_fn: Callable,
        loss: SquaredLoss,
        *args,
    ):
        self._meas_fn = meas_fn
        self._jac_fn = jac_fn
        self.loss = loss
        self.args = args
        self.linear = True

    def jac_fn(self, x: np.ndarray) -> np.ndarray:
        return self._jac_fn(x, *self.args)

    def meas_fn(self, x: np.ndarray) -> np.ndarray:
        return self._meas_fn(x, *self.args)


def lin_meas_fn(x):
    length = int(x.shape[0] / 2)
    J = np.concatenate((-np.eye(length), np.eye(length)), axis=1)
    return J @ x


def lin_jac_fn(x):
    length = int(x.shape[0] / 2)
    return np.concatenate((-np.eye(length), np.eye(length)), axis=1)


class LinearDisplacementModel(MeasModel):
    def __init__(self, loss: SquaredLoss) -> None:
        MeasModel.__init__(self, lin_meas_fn, lin_jac_fn, loss)
        self.linear = True


"""
Main GBP functions.
Defines classes for variable nodes, factor nodes and edges and factor graph.
"""


class FactorGraph:
    def __init__(
        self,
        gbp_settings: GBPSettings = GBPSettings(),
    ):
        self.var_nodes = []
        self.factors = []
        self.gbp_settings = gbp_settings

    def add_var_node(
        self,
        dofs: int,
        prior_mean: Optional[np.ndarray] = None,
        prior_diag_cov: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        variableID = len(self.var_nodes)
        self.var_nodes.append(VariableNode(variableID, dofs))
        if prior_mean is not None and prior_diag_cov is not None:
            prior_cov = np.zeros([dofs, dofs])
            prior_cov[range(dofs), range(dofs)] = prior_diag_cov
            self.var_nodes[-1].prior.set_with_cov_form(prior_mean, prior_cov)
            self.var_nodes[-1].update_belief()

    def add_factor(
        self,
        adj_var_ids: List[int],
        measurement: np.ndarray,
        meas_model: MeasModel,
    ) -> None:
        factorID = len(self.factors)
        adj_var_nodes = [self.var_nodes[i] for i in adj_var_ids]
        self.factors.append(
            Factor(factorID, adj_var_nodes, measurement, meas_model))
        for var in adj_var_nodes:
            var.adj_factors.append(self.factors[-1])

    def update_all_beliefs(self) -> None:
        for var_node in self.var_nodes:
            var_node.update_belief()

    def compute_all_messages(self, apply_dropout: bool = True) -> None:
        for factor in self.factors:
            dropout_off = apply_dropout and np.random.rand() > self.gbp_settings.dropout
            if dropout_off or not apply_dropout:
                damping = self.gbp_settings.get_damping(
                    factor.iters_since_relin)
                factor.compute_messages(damping)

    def linearise_all_factors(self) -> None:
        for factor in self.factors:
            factor.compute_factor()

    def robustify_all_factors(self) -> None:
        for factor in self.factors:
            factor.robustify_loss()

    def jit_linearisation(self) -> None:
        """
        Check for all factors that the current estimate
        is close to the linearisation point.
        If not, relinearise the factor distribution.
        Relinearisation is only allowed at a maximum frequency
        of once every min_linear_iters iterations.
        """
        for factor in self.factors:
            if not factor.meas_model.linear:
                adj_belief_means = factor.get_adj_means()
                factor.iters_since_relin += 1
                diff_cond = np.linalg.norm(factor.linpoint - adj_belief_means) > self.gbp_settings.beta
                iters_cond = factor.iters_since_relin >= self.gbp_settings.min_linear_iters
                if diff_cond and iters_cond:
                    factor.compute_factor()

    def synchronous_iteration(self) -> None:
        self.robustify_all_factors()
        self.jit_linearisation()  # For linear factors, no compute is done
        self.compute_all_messages()
        self.update_all_beliefs()

    def random_message(self) -> None:
        """
        Sends messages to all adjacent nodes from a random factor.
        """
        self.robustify_all_factors()
        self.jit_linearisation()  # For linear factors, no compute is done
        ix = np.random.randint(len(self.factors))
        factor = self.factors[ix]
        damping = self.gbp_settings.get_damping(factor.iters_since_relin)
        factor.compute_messages(damping)
        self.update_all_beliefs()

    def gbp_solve(
        self,
        n_iters: Optional[int] = 20,
        converged_threshold: Optional[float] = 1e-6,
        include_priors: bool = True
    ) -> None:
        energy_log = [self.energy()]
        print(f"\nInitial Energy {energy_log[0]:.5f}")

        i = 0
        count = 0
        not_converged = True

        while not_converged and i < n_iters:
            self.synchronous_iteration()
            if i in self.gbp_settings.reset_iters_since_relin:
                for f in self.factors:
                    f.iters_since_relin = 1

            energy_log.append(self.energy(include_priors=include_priors))
            print(
                f"Iter {i+1}  --- "
                f"Energy {energy_log[-1]:.5f} --- "
            )
            i += 1
            if abs(energy_log[-2] - energy_log[-1]) < converged_threshold:
                count += 1
                if count == 3:
                    not_converged = False
            else:
                count = 0

    def energy(
        self,
        eval_point: np.ndarray = None,
        include_priors: bool = True
    ) -> float:
        """
        Computes the sum of all of the squared errors in the graph
        using the appropriate local loss function.
        """
        if eval_point is None:
            energy = sum([factor.get_energy() for factor in self.factors])
        else:
            var_dofs = np.ndarray([v.dofs for v in self.var_nodes])
            var_ix = np.concatenate([np.ndarray([0]), np.cumsum(var_dofs, axis=0)[:-1]])
            energy = 0.
            for f in self.factors:
                local_eval_point = np.concatenate([eval_point[var_ix[v.variableID]: var_ix[v.variableID] + v.dofs] for v in f.adj_var_nodes])
                energy += f.get_energy(local_eval_point)
        if include_priors:
            prior_energy = sum([var.get_prior_energy() for var in self.var_nodes])
            energy += prior_energy
        return energy

    def get_joint_dim(self) -> int:
        return sum([var.dofs for var in self.var_nodes])

    def get_joint(self) -> Gaussian:
        """
        Get the joint distribution over all variables in the information form
        If nonlinear factors, it is taken at the current linearisation point.
        """
        dim = self.get_joint_dim()
        joint = Gaussian(dim)

        # Priors
        var_ix = [0] * len(self.var_nodes)
        counter = 0
        for var in self.var_nodes:
            var_ix[var.variableID] = int(counter)
            joint.eta[counter:counter + var.dofs] += var.prior.eta
            joint.lam[counter:counter + var.dofs, counter:counter + var.dofs] += var.prior.lam
            counter += var.dofs

        # Other factors
        for factor in self.factors:
            factor_ix = 0
            for adj_var_node in factor.adj_var_nodes:
                vID = adj_var_node.variableID
                # Diagonal contribution of factor
                joint.eta[var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor.eta[factor_ix:factor_ix + adj_var_node.dofs]
                joint.lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                    factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
                other_factor_ix = 0
                for other_adj_var_node in factor.adj_var_nodes:
                    if other_adj_var_node.variableID > adj_var_node.variableID:
                        other_vID = other_adj_var_node.variableID
                        # Off diagonal contributions of factor
                        joint.lam[var_ix[vID]:var_ix[vID] + adj_var_node.dofs, var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs] += \
                            factor.factor.lam[factor_ix:factor_ix + adj_var_node.dofs, other_factor_ix:other_factor_ix + other_adj_var_node.dofs]
                        joint.lam[var_ix[other_vID]:var_ix[other_vID] + other_adj_var_node.dofs, var_ix[vID]:var_ix[vID] + adj_var_node.dofs] += \
                            factor.factor.lam[other_factor_ix:other_factor_ix + other_adj_var_node.dofs, factor_ix:factor_ix + adj_var_node.dofs]
                    other_factor_ix += other_adj_var_node.dofs
                factor_ix += adj_var_node.dofs

        return joint

    def MAP(self) -> np.ndarray:
        return self.get_joint().mean()

    def dist_from_MAP(self) -> np.ndarray:
        return np.linalg.norm(self.get_joint().mean() - self.belief_means())

    def belief_means(self) -> np.ndarray:
        """ Get an array containing all current estimates of belief means. """
        return np.concatenate([var.belief.mean() for var in self.var_nodes])

    def belief_covs(self) -> List[np.ndarray]:
        """ Get a list of all belief covariances. """
        covs = [var.belief.cov() for var in self.var_nodes]
        return covs

    def print(self, brief=False) -> None:
        print("\nFactor Graph:")
        print(f"# Variable nodes: {len(self.var_nodes)}")
        if not brief:
            for i, var in enumerate(self.var_nodes):
                print(f"Variable {i}: connects to factors {[f.factorID for f in var.adj_factors]}")
                print(f"    dofs: {var.dofs}")
                print(f"    prior mean: {var.prior.mean()}")
                print(f"    prior covariance: diagonal sigma {np.diag(var.prior.cov())}")
        print(f"# Factors: {len(self.factors)}")
        if not brief:
            for i, factor in enumerate(self.factors):
                if factor.meas_model.linear:
                    print("Linear", end=" ")
                else:
                    print("Nonlinear", end=" ")
                print(f"Factor {i}: connects to variables {factor.adj_vIDs}")
                print(
                    f"    measurement model: {type(factor.meas_model).__name__},"
                    f" {type(factor.meas_model.loss).__name__},"
                    f" diagonal sigma {np.diag(factor.meas_model.loss.effective_cov)}"
                )
                print(f"    measurement: {factor.measurement}")
        print("\n")


class VariableNode:
    def __init__(self, id: int, dofs: int):
        self.variableID = id
        self.dofs = dofs
        self.adj_factors = []
        # prior factor, implemented as part of variable node
        self.prior = Gaussian(dofs)
        self.belief = Gaussian(dofs)

    def update_belief(self) -> None:
        """
        Update local belief estimate by taking product
        of all incoming messages along all edges.
        """
        # message from prior factor
        self.belief.eta = self.prior.eta.copy()
        self.belief.lam = self.prior.lam.copy()
        # messages from other adjacent variables
        for factor in self.adj_factors:
            message_ix = factor.adj_vIDs.index(self.variableID)
            self.belief.eta += factor.messages[message_ix].eta
            self.belief.lam += factor.messages[message_ix].lam

    def get_prior_energy(self) -> float:
        energy = 0.
        if self.prior.lam[0, 0] != 0.:
            residual = self.belief.mean() - self.prior.mean()
            energy += 0.5 * residual @ self.prior.lam @ residual
        return energy


class Factor:
    def __init__(
        self,
        id: int,
        adj_var_nodes: List[VariableNode],
        measurement: np.ndarray,
        meas_model: MeasModel,
    ) -> None:

        self.factorID = id

        self.adj_var_nodes = adj_var_nodes
        self.dofs = sum([var.dofs for var in adj_var_nodes])
        self.adj_vIDs = [var.variableID for var in adj_var_nodes]
        self.messages = [Gaussian(var.dofs) for var in adj_var_nodes]

        self.factor = Gaussian(self.dofs)
        self.linpoint = np.zeros(self.dofs)

        self.measurement = measurement
        self.meas_model = meas_model

        # For smarter GBP implementations
        self.iters_since_relin = 0

        self.compute_factor()

    def get_adj_means(self) -> np.ndarray:
        adj_belief_means = [var.belief.mean() for var in self.adj_var_nodes]
        return np.concatenate(adj_belief_means)

    def get_residual(self, eval_point: np.ndarray = None) -> np.ndarray:
        """ Compute the residual vector. """
        if eval_point is None:
            eval_point = self.get_adj_means()
        return self.meas_model.meas_fn(eval_point) - self.measurement

    def get_energy(self, eval_point: np.ndarray = None) -> float:
        """ Computes the squared error using the appropriate loss function. """
        residual = self.get_residual(eval_point)
        inf_mat = np.linalg.inv(self.meas_model.loss.effective_cov)
        return 0.5 * residual @ inf_mat @ residual

    def robust(self) -> bool:
        return self.meas_model.loss.robust()

    def compute_factor(self) -> None:
        """
            Compute the factor at current adjacente beliefs using robust.
            If measurement model is linear then factor will always be
            the same regardless of linearisation point.
        """
        self.linpoint = self.get_adj_means()
        J = self.meas_model.jac_fn(self.linpoint)
        pred_measurement = self.meas_model.meas_fn(self.linpoint)
        self.meas_model.loss.get_effective_cov(pred_measurement - self.measurement)
        effective_lam = np.linalg.inv(self.meas_model.loss.effective_cov)
        self.factor.lam = J.T @ effective_lam @ J
        self.factor.eta = ((J.T @ effective_lam) @ (J @ self.linpoint + self.measurement - pred_measurement)).flatten()
        self.iters_since_relin = 0

    def robustify_loss(self) -> None:
        """
        Rescale the variance of the noise in the Gaussian
        measurement model if necessary and update the factor
        correspondingly.
        """
        old_effective_cov = self.meas_model.loss.effective_cov[0, 0]
        self.meas_model.loss.get_effective_cov(self.get_residual())
        self.factor.eta *= old_effective_cov / self.meas_model.loss.effective_cov[0, 0]
        self.factor.lam *= old_effective_cov / self.meas_model.loss.effective_cov[0, 0]

    def compute_messages(self, damping: float = 0.) -> None:
        """ Compute all outgoing messages from the factor. """
        messages_eta, messages_lam = [], []

        sdim = 0
        for v in range(len(self.adj_vIDs)):
            eta_factor = self.factor.eta.copy()
            lam_factor = self.factor.lam.copy()

            # Take product of factor with incoming messages
            start = 0
            for var in range(len(self.adj_vIDs)):
                if var != v:
                    var_dofs = self.adj_var_nodes[var].dofs
                    eta_mess = self.adj_var_nodes[var].belief.eta - self.messages[var].eta
                    lam_mess = self.adj_var_nodes[var].belief.lam - self.messages[var].lam
                    eta_factor[start:start + var_dofs] += eta_mess
                    lam_factor[start:start + var_dofs, start:start + var_dofs] += lam_mess
                start += self.adj_var_nodes[var].dofs

            # Divide up parameters of distribution
            dofs = self.adj_var_nodes[v].dofs
            eo = eta_factor[sdim:sdim + dofs]
            eno = np.concatenate((eta_factor[:sdim], eta_factor[sdim + dofs:]))

            loo = lam_factor[sdim:sdim + dofs, sdim:sdim + dofs]
            lono = np.concatenate(
                (lam_factor[sdim:sdim + dofs, :sdim], lam_factor[sdim:sdim + dofs, sdim + dofs:]),
                axis=1)
            lnoo = np.concatenate(
                (lam_factor[:sdim, sdim:sdim + dofs], lam_factor[sdim + dofs:, sdim:sdim + dofs]),
                axis=0)
            lnono = np.concatenate((
                np.concatenate((lam_factor[:sdim, :sdim], lam_factor[:sdim, sdim + dofs:]), axis=1),
                np.concatenate((lam_factor[sdim + dofs:, :sdim], lam_factor[sdim + dofs:, sdim + dofs:]), axis=1)
            ), axis=0)

            new_message_lam = loo - lono @ np.linalg.inv(lnono) @ lnoo
            new_message_eta = eo - lono @ np.linalg.inv(lnono) @ eno
            messages_eta.append((1 - damping) * new_message_eta + damping * self.messages[v].eta)
            messages_lam.append((1 - damping) * new_message_lam + damping * self.messages[v].lam)
            sdim += self.adj_var_nodes[v].dofs

        for v in range(len(self.adj_vIDs)):
            self.messages[v].lam = messages_lam[v]
            self.messages[v].eta = messages_eta[v]


"""
Visualisation function
"""


def draw(i):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.set_tight_layout(True)
    plt.title(i)

    # plot beliefs
    means = fg.belief_means().reshape([size * size, 2])
    plt.scatter(means[:, 0], means[:, 1], color="blue")
    for j, cov in enumerate(fg.belief_covs()):
        circle = plt.Circle(
            (means[j, 0], means[j, 1]),
            np.sqrt(cov[0, 0]), linewidth=0.5, color='blue', fill=False
        )
        ax.add_patch(circle)

    # plot true marginals
    plt.scatter(map_soln[:, 0], map_soln[:, 1], color="g")
    for j, cov in enumerate(marg_covs):
        circle = plt.Circle(
            (map_soln[j, 0], map_soln[j, 1]),
            np.sqrt(marg_covs[j]), linewidth=0.5, color='g', fill=False
        )
        ax.add_patch(circle)

    # draw lines for factors
    for f in fg.factors:
        bels = np.array([means[f.adj_vIDs[0]], means[f.adj_vIDs[1]]])
        plt.plot(bels[:, 0], bels[:, 1], color='black', linewidth=0.3)

    # draw lines for belief error
    for i in range(len(means)):
        xs = [means[i, 0], map_soln[i, 0]]
        ys = [means[i, 1], map_soln[i, 1]]
        plt.plot(xs, ys, color='grey', linewidth=0.3, linestyle='dashed')

    plt.axis('scaled')
    plt.xlim([-1, size])
    plt.ylim([-1, size])

    # convert to image
    ax.axis('off')
    fig.tight_layout(pad=0)
    ax.margins(0)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return img

if __name__ == "__main__":

    np.random.seed(1)

    size = 3
    dim = 2

    prior_noise_std = 0.2

    gbp_settings = GBPSettings(
        damping=0.,
        beta=0.1,
        num_undamped_iters=1,
        min_linear_iters=10,
        dropout=0.0,
    )

    # GBP library soln ------------------------------------------

    noise_cov = np.array([0.01, 0.01])

    prior_sigma = np.array([1.3**2, 1.3**2])
    prior_noise_std = 0.2

    fg = FactorGraph(gbp_settings)

    init_noises = np.random.normal(np.zeros([size*size, 2]), prior_noise_std)
    meas_noises = np.random.normal(np.zeros([100, 2]), np.sqrt(noise_cov[0]))

    for i in range(size):
        for j in range(size):
            init = np.array([j, i])
            noise_init = init_noises[j + i * size]
            init = init + noise_init
            sigma = prior_sigma
            if i == 0 and j == 0:
                init = np.array([j, i])
                sigma = np.array([0.0001, 0.0001])
            print(init, sigma)
            fg.add_var_node(2, init, sigma)

    m = 0
    for i in range(size):
        for j in range(size):
            if j < size - 1:
                meas = np.array([1., 0.])
                meas += meas_noises[m]
                fg.add_factor(
                    [i * size + j, i * size + j + 1],
                    meas,
                    LinearDisplacementModel(SquaredLoss(dim, noise_cov))
                )
                m += 1
            if i < size - 1:
                meas = np.array([0., 1.])
                meas += meas_noises[m]
                fg.add_factor(
                    [i * size + j, (i + 1) * size + j],
                    meas,
                    LinearDisplacementModel(SquaredLoss(dim, noise_cov))
                )
                m += 1

    fg.print(brief=True)

    # # for vis ---------------

    joint = fg.get_joint()
    marg_covs = np.diag(joint.cov())[::2]
    map_soln = fg.MAP().reshape([size * size, 2])

    # # run gbp ---------------

    gbp_settings = GBPSettings(
        damping=0.,
        beta=0.1,
        num_undamped_iters=1,
        min_linear_iters=10,
        dropout=0.0,
    )

    # fg.compute_all_messages()

    import ipdb; ipdb.set_trace()

    # i = 0
    n_iters = 5
    while i <= n_iters:
        # img = draw(i)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        print(f"Iter {i}  --- Energy {fg.energy():.5f}")

        # fg.random_message()
        fg.synchronous_iteration()
        i += 1

        for f in fg.factors:
            for m in f.messages:
                print(np.linalg.inv(m.lam) @ m.eta)

        print(fg.belief_means())

        import ipdb; ipdb.set_trace()


        # time.sleep(0.05)
