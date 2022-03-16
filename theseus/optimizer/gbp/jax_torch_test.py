import numpy as np
import torch
import jax
import jax.numpy as jnp

import time


def pass_fac_to_var_messages(
    potentials_eta,
    potentials_lam,
    vtof_msgs_eta,
    vtof_msgs_lam,
    adj_var_dofs_nested,
):
    ftov_msgs_eta = [None] * len(vtof_msgs_eta)
    ftov_msgs_lam = [None] * len(vtof_msgs_eta)

    start = 0
    for i in range(len(adj_var_dofs_nested)):
        adj_var_dofs = adj_var_dofs_nested[i]
        num_optim_vars = len(adj_var_dofs)


        inp_msgs_eta = vtof_msgs_eta[start: start + num_optim_vars]
        inp_msgs_lam = vtof_msgs_lam[start: start + num_optim_vars]

        num_optim_vars = len(adj_var_dofs)
        ftov_eta, ftov_lam = [], []

        sdim = 0
        for v in range(num_optim_vars):
            eta_factor = potentials_eta[i].clone()[0]
            lam_factor = potentials_lam[i].clone()[0]

            # Take product of factor with incoming messages
            start_in = 0
            for var in range(num_optim_vars):
                var_dofs = adj_var_dofs[var]
                if var != v:
                    eta_mess = vtof_msgs_eta[var]
                    lam_mess = vtof_msgs_lam[var]
                    eta_factor[start_in:start_in + var_dofs] += eta_mess
                    lam_factor[start_in:start_in + var_dofs, start_in:start_in + var_dofs] += lam_mess
                start_in += var_dofs

            # Divide up parameters of distribution
            dofs = adj_var_dofs[v]
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

            ftov_eta.append(new_message_eta[None, :])
            ftov_lam.append(new_message_lam[None, :])

            sdim += dofs

        ftov_msgs_eta[start: start + num_optim_vars] = ftov_eta
        ftov_msgs_lam[start: start + num_optim_vars] = ftov_lam

        start += num_optim_vars

    return ftov_msgs_eta, ftov_msgs_lam


@jax.jit
def pass_fac_to_var_messages_jax(
    potentials_eta,
    potentials_lam,
    vtof_msgs_eta,
    vtof_msgs_lam,
    adj_var_dofs_nested,
):
    ftov_msgs_eta = [None] * len(vtof_msgs_eta)
    ftov_msgs_lam = [None] * len(vtof_msgs_eta)

    start = 0
    for i in range(len(adj_var_dofs_nested)):
        adj_var_dofs = adj_var_dofs_nested[i]
        num_optim_vars = len(adj_var_dofs)


        inp_msgs_eta = vtof_msgs_eta[start: start + num_optim_vars]
        inp_msgs_lam = vtof_msgs_lam[start: start + num_optim_vars]

        num_optim_vars = len(adj_var_dofs)
        ftov_eta, ftov_lam = [], []

        sdim = 0
        for v in range(num_optim_vars):
            eta_factor = potentials_eta[i][0]
            lam_factor = potentials_lam[i][0]

            # Take product of factor with incoming messages
            start_in = 0
            for var in range(num_optim_vars):
                var_dofs = adj_var_dofs[var]
                if var != v:
                    eta_mess = vtof_msgs_eta[var]
                    lam_mess = vtof_msgs_lam[var]
                    eta_factor = eta_factor.at[start_in:start_in + var_dofs].add(eta_mess)
                    lam_factor = lam_factor.at[start_in:start_in + var_dofs, start_in:start_in + var_dofs].add(lam_mess)
                start_in += var_dofs

            # Divide up parameters of distribution
            dofs = adj_var_dofs[v]
            eo = eta_factor[sdim:sdim + dofs]
            eno = jnp.concatenate((eta_factor[:sdim], eta_factor[sdim + dofs:]))

            loo = lam_factor[sdim:sdim + dofs, sdim:sdim + dofs]
            lono = jnp.concatenate(
                (lam_factor[sdim:sdim + dofs, :sdim], lam_factor[sdim:sdim + dofs, sdim + dofs:]),
                axis=1)
            lnoo = jnp.concatenate(
                (lam_factor[:sdim, sdim:sdim + dofs], lam_factor[sdim + dofs:, sdim:sdim + dofs]),
                axis=0)
            lnono = jnp.concatenate((
                jnp.concatenate((lam_factor[:sdim, :sdim], lam_factor[:sdim, sdim + dofs:]), axis=1),
                jnp.concatenate((lam_factor[sdim + dofs:, :sdim], lam_factor[sdim + dofs:, sdim + dofs:]), axis=1)
            ), axis=0)

            new_message_lam = loo - lono @ jnp.linalg.inv(lnono) @ lnoo
            new_message_eta = eo - lono @ jnp.linalg.inv(lnono) @ eno

            ftov_eta.append(new_message_eta[None, :])
            ftov_lam.append(new_message_lam[None, :])

            sdim += dofs

        ftov_msgs_eta[start: start + num_optim_vars] = ftov_eta
        ftov_msgs_lam[start: start + num_optim_vars] = ftov_lam

        start += num_optim_vars

    return ftov_msgs_eta, ftov_msgs_lam





if __name__ == "__main__":

    adj_var_dofs_nested = [[2], [2], [2], [2], [2], [2], [2], [2], [2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

    potentials_eta = [torch.tensor([[0., 0.]]), torch.tensor([[ 0.5292, -0.1270]]), torch.tensor([[ 1.2858, -0.2724]]), torch.tensor([[0.2065, 0.5016]]), torch.tensor([[0.6295, 0.5622]]), torch.tensor([[1.3565, 0.3479]]), torch.tensor([[-0.0382,  1.1380]]), torch.tensor([[0.7259, 1.0533]]), torch.tensor([[1.1630, 1.0795]]), torch.tensor([[-100.4221,   -5.8282,  100.4221,    5.8282]]), torch.tensor([[  11.0062, -111.4472,  -11.0062,  111.4472]]), torch.tensor([[-109.0159,   -5.0249,  109.0159,    5.0249]]), torch.tensor([[ -9.0086, -93.1627,   9.0086,  93.1627]]), torch.tensor([[  1.2289, -90.6423,  -1.2289,  90.6423]]), torch.tensor([[-97.3211,  -5.3036,  97.3211,   5.3036]]), torch.tensor([[  6.9166, -96.0325,  -6.9166,  96.0325]]), torch.tensor([[-93.1283,   8.4521,  93.1283,  -8.4521]]), torch.tensor([[  6.7125, -99.8733,  -6.7125,  99.8733]]), torch.tensor([[  11.1731, -102.3442,  -11.1731,  102.3442]]), torch.tensor([[-116.5980,   -7.4204,  116.5980,    7.4204]]), torch.tensor([[-98.0816,   8.8763,  98.0816,  -8.8763]])]
    potentials_lam = [torch.tensor([[[10000.,     0.],
         [    0., 10000.]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[0.5917, 0.0000],
         [0.0000, 0.5917]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]]), torch.tensor([[[ 100.,    0., -100.,    0.],
         [   0.,  100.,    0., -100.],
         [-100.,    0.,  100.,    0.],
         [   0., -100.,    0.,  100.]]])]

    vtof_msgs_eta = [torch.tensor([[ 0.8536, -1.5929]]), torch.tensor([[182.3461,  16.7745]]), torch.tensor([[222.8854,  13.1250]]), torch.tensor([[-10.1678, 202.9393]]), torch.tensor([[200.4927, 213.6843]]), torch.tensor([[264.5976, 132.6887]]), torch.tensor([[-17.9007, 222.3988]]), torch.tensor([[127.5813, 277.0478]]), torch.tensor([[191.0187, 201.1600]]), torch.tensor([[ 5.6620, -4.6983]]), torch.tensor([[83.3856, 10.9277]]), torch.tensor([[-4.8085,  3.1053]]), torch.tensor([[ 0.9854, 93.0631]]), torch.tensor([[153.1307,  16.3761]]), torch.tensor([[98.1263,  3.3349]]), torch.tensor([[129.7635,   5.8644]]), torch.tensor([[140.2319, 158.5661]]), torch.tensor([[127.3308,   9.2454]]), torch.tensor([[187.8824,  92.8337]]), torch.tensor([[-16.5414, 145.2973]]), torch.tensor([[152.8149, 148.6686]]), torch.tensor([[ -4.1601, 169.0230]]), torch.tensor([[-12.0344,  99.1287]]), torch.tensor([[153.7062, 168.3496]]), torch.tensor([[149.0974,  72.7772]]), torch.tensor([[157.2429, 167.7175]]), torch.tensor([[ 70.8858, 152.1307]]), torch.tensor([[196.2848, 100.8102]]), torch.tensor([[ 99.5512, 100.5530]]), torch.tensor([[ -5.9426, 125.5461]]), torch.tensor([[ 87.5787, 197.8408]]), torch.tensor([[ 98.8758, 207.2840]]), torch.tensor([[ 93.7936, 102.7661]])]
    vtof_msgs_lam = [torch.tensor([[95.7949,  0.0000],
        [ 0.0000, 95.7949]]), torch.tensor([[190.3769,   0.0000],
        [  0.0000, 190.3769]]), torch.tensor([[109.9605,   0.0000],
        [  0.0000, 109.9605]]), torch.tensor([[190.3769,   0.0000],
        [  0.0000, 190.3769]]), torch.tensor([[197.8604,   0.0000],
        [  0.0000, 197.8604]]), torch.tensor([[132.5915,   0.0000],
        [  0.0000, 132.5915]]), torch.tensor([[109.9605,   0.0000],
        [  0.0000, 109.9605]]), torch.tensor([[132.5915,   0.0000],
        [  0.0000, 132.5915]]), torch.tensor([[99.8496,  0.0000],
        [ 0.0000, 99.8496]]), torch.tensor([[10047.8975,     0.0000],
        [    0.0000, 10047.8975]]), torch.tensor([[91.9540,  0.0000],
        [ 0.0000, 91.9540]]), torch.tensor([[10047.8975,     0.0000],
        [    0.0000, 10047.8975]]), torch.tensor([[91.9540,  0.0000],
        [ 0.0000, 91.9540]]), torch.tensor([[158.0642,   0.0000],
        [  0.0000, 158.0642]]), torch.tensor([[49.3043,  0.0000],
        [ 0.0000, 49.3043]]), torch.tensor([[132.5106,   0.0000],
        [  0.0000, 132.5106]]), torch.tensor([[141.4631,   0.0000],
        [  0.0000, 141.4631]]), torch.tensor([[61.8396,  0.0000],
        [ 0.0000, 61.8396]]), torch.tensor([[94.9975,  0.0000],
        [ 0.0000, 94.9975]]), torch.tensor([[132.5106,   0.0000],
        [  0.0000, 132.5106]]), torch.tensor([[141.4631,   0.0000],
        [  0.0000, 141.4631]]), torch.tensor([[158.0642,   0.0000],
        [  0.0000, 158.0642]]), torch.tensor([[49.3043,  0.0000],
        [ 0.0000, 49.3043]]), torch.tensor([[156.5110,   0.0000],
        [  0.0000, 156.5110]]), torch.tensor([[72.2502,  0.0000],
        [ 0.0000, 72.2502]]), torch.tensor([[156.5110,   0.0000],
        [  0.0000, 156.5110]]), torch.tensor([[72.2502,  0.0000],
        [ 0.0000, 72.2502]]), torch.tensor([[99.7104,  0.0000],
        [ 0.0000, 99.7104]]), torch.tensor([[50.5165,  0.0000],
        [ 0.0000, 50.5165]]), torch.tensor([[61.8396,  0.0000],
        [ 0.0000, 61.8396]]), torch.tensor([[94.9975,  0.0000],
        [ 0.0000, 94.9975]]), torch.tensor([[99.7104,  0.0000],
        [ 0.0000, 99.7104]]), torch.tensor([[50.5165,  0.0000],
        [ 0.0000, 50.5165]])]
    vtof_msgs_eta = torch.cat(vtof_msgs_eta)
    # vtof_msgs_lam = torch.cat([m[None, ...] for m in vtof_msgs_lam])

    t1 = time.time()
    times = []
    for i in range(100):
        t_start = time.time()
        ftov_msgs_eta, ftov_msgs_lam = pass_fac_to_var_messages(
            potentials_eta,
            potentials_lam,
            vtof_msgs_eta,
            vtof_msgs_lam,
            adj_var_dofs_nested,
        )
        times.append(time.time() - t_start)

    t2 = time.time()
    print("-------------- TORCH --------------")
    print("elapsed", t2 - t1)
    print("min max mean", np.min(times), np.max(times), np.mean(times))

    # print(ftov_msgs_eta)
    # print(ftov_msgs_lam)


    potentials_eta_jax = [jnp.array(pe) for pe in potentials_eta]
    potentials_lam_jax = [jnp.array(pe) for pe in potentials_lam]
    vtof_msgs_eta_jax = jnp.array(vtof_msgs_eta)
    vtof_msgs_lam_jax = [jnp.array(pe) for pe in vtof_msgs_lam]

    t1 = time.time()
    times = []
    for i in range(10):
        t_start = time.time()
        ftov_msgs_eta_jax, ftov_msgs_lam_jax = pass_fac_to_var_messages_jax(
            potentials_eta_jax,
            potentials_lam_jax,
            vtof_msgs_eta_jax,
            vtof_msgs_lam_jax,
            adj_var_dofs_nested,
        )
        times.append(time.time() - t_start)

    t2 = time.time()
    print("\n\n")
    print("-------------- JAX --------------")
    print("elapsed", t2 - t1)
    print("min max mean", np.min(times), np.max(times), np.mean(times))

    # print(ftov_msgs_eta_jax)
    # print(ftov_msgs_lam_jax)
