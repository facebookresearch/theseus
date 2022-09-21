python examples/homography_estimation.py autograd_mode=dense outer_optim.batch_size=64 outer_optim.num_epochs=1 inner_optim.max_iters=10
python examples/homography_estimation.py autograd_mode=loop_batch outer_optim.batch_size=64 outer_optim.num_epochs=1 inner_optim.max_iters=10
python examples/homography_estimation.py autograd_mode=vmap outer_optim.batch_size=64 outer_optim.num_epochs=1 inner_optim.max_iters=10
