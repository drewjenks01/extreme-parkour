import torch


tvm = torch.tensor([[ 9.9999e-01, -2.7312e-04],
        [ 9.9999e-01, -2.9291e-04],
        [ 9.9999e-01, -3.6673e-04],
        [ 9.9999e-01,  2.5181e-04],
        [ 9.9999e-01,  1.8673e-04],
        [ 9.9999e-01,  2.3703e-05],
        [ 9.9998e-01,  4.8760e-04],
        [ 9.9998e-01,  4.9661e-05],
        [ 9.9999e-01, -9.2095e-05],
        [ 9.9999e-01,  3.3397e-04]], device='cuda:0')

norm = torch.tensor([[0.9995],
        [1.1005],
        [1.4510],
        [1.4505],
        [1.4504],
        [1.4484],
        [0.5007],
        [0.4993],
        [1.4497],
        [1.4506]], device='cuda:0')

curr_vel = torch.tensor([[ 0.0223,  0.0088],
        [-0.0334,  0.0167],
        [-0.0792,  0.0200],
        [-0.0254, -0.0067],
        [-0.0342, -0.0040],
        [ 0.1129, -0.0007],
        [-0.0468, -0.0051],
        [ 0.0521, -0.0012],
        [ 0.0166,  0.0036],
        [-0.0304, -0.0105]], device='cuda:0')

tpr = torch.tensor([[ 9.9952e-01, -2.7299e-04],
        [ 1.1005e+00, -3.2234e-04],
        [ 1.4510e+00, -5.3215e-04],
        [ 1.4505e+00,  3.6526e-04],
        [ 1.4504e+00,  2.7084e-04],
        [ 1.4484e+00,  3.4332e-05],
        [ 5.0069e-01,  2.4414e-04],
        [ 4.9929e-01,  2.4796e-05],
        [ 1.4497e+00, -1.3351e-04],
        [ 1.4506e+00,  4.8447e-04]], device='cuda:0')

root_states = torch.tensor([[ 1.9000e+01,  2.0003e+00,  4.1906e-01, -2.5626e-03,  1.1804e-02,
          5.6425e-04,  9.9993e-01,  2.2324e-02,  8.7600e-03, -1.2016e-01,
         -1.6271e-01,  1.0593e+00,  5.2252e-02],
        [ 1.9000e+01,  6.0003e+00,  4.1903e-01, -7.2263e-03,  1.9461e-03,
         -4.4981e-03,  9.9996e-01, -3.3408e-02,  1.6740e-02, -1.1832e-01,
         -8.6475e-01,  3.9744e-01, -5.0800e-01],
        [ 1.8999e+01,  1.0001e+01,  4.1704e-01, -7.2315e-03,  6.0736e-03,
         -2.5816e-03,  9.9995e-01, -7.9178e-02,  1.9980e-02, -1.8267e-01,
         -5.5791e-01,  8.8699e-01, -1.5838e-01],
        [ 1.8999e+01,  1.4000e+01,  4.1892e-01,  9.1985e-03,  9.5638e-05,
         -4.4297e-03,  9.9995e-01, -2.5351e-02, -6.6804e-03, -1.0503e-01,
          1.4615e-01,  7.4076e-02, -6.5490e-01],
        [ 1.9000e+01,  1.8000e+01,  4.1724e-01,  3.2452e-03,  4.2842e-03,
          9.0343e-04,  9.9999e-01, -3.4242e-02, -4.0005e-03, -1.7569e-01,
         -2.2351e-01,  5.2621e-01, -8.8344e-02],
        [ 1.9002e+01,  2.2000e+01,  4.1872e-01,  5.7853e-04, -1.1568e-02,
         -8.2050e-04,  9.9993e-01,  1.1295e-01, -7.2744e-04, -1.5840e-01,
          1.6946e-03, -1.3932e+00, -2.0949e-01],
        [ 1.8999e+01,  2.6000e+01,  4.1731e-01, -1.7889e-03,  8.1726e-03,
         -2.0405e-03,  9.9996e-01, -4.6772e-02, -5.1337e-03, -1.8854e-01,
         -3.1253e-01,  9.5988e-01, -3.5795e-01],
        [ 1.9001e+01,  3.0000e+01,  4.1864e-01,  6.2503e-03, -3.1548e-03,
          2.2447e-04,  9.9998e-01,  5.2062e-02, -1.2240e-03, -1.5760e-01,
          4.0082e-01, -4.7959e-01, -6.9575e-02],
        [ 1.9000e+01,  3.4000e+01,  4.1771e-01, -4.8969e-03, -3.7853e-03,
          9.6200e-04,  9.9998e-01,  1.6606e-02,  3.5974e-03, -1.6291e-01,
         -2.7105e-01, -2.0041e-01,  1.0323e-01],
        [ 1.8999e+01,  3.8000e+01,  4.1832e-01,  7.2529e-03,  8.6865e-03,
         -3.2251e-03,  9.9993e-01, -3.0352e-02, -1.0466e-02, -1.4657e-01,
         -9.0732e-03,  1.0440e+00, -5.0102e-01]], device='cuda:0')[:, 7:9]

commands = torch.tensor([[0.8531, 0.0000, -0.0000, 0.0000],
        [0.2316, 0.0000, 0.0000, 0.0000],
        [1.1790, 0.0000, 0.0000, 0.0000],
        [0.5519, 0.0000, 0.0000, 0.0000],
        [0.5069, 0.0000, -0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, -0.0000, 0.0000],
        [0.9323, 0.0000, -0.0000, 0.0000],
        [0.4992, 0.0000, 0.0000, 0.0000]], device='cuda:0')[: , 0]

norm = torch.norm(tpr, dim=-1, keepdim=True)
target_vec_norm = tpr / (norm + 1e-5)
cur_vel = root_states
t_sum = torch.sum(target_vec_norm * cur_vel, dim=-1)
t_min = torch.minimum(t_sum, commands)
rew = t_min / (commands + 1e-5)

for n,t,c,ts,tm,comms, r in zip(norm, target_vec_norm, cur_vel, t_sum, t_min, commands, rew):
    print(f'norm: {n}')
    print(f'target_vec_norm: {t}')
    print(f'cur_vel: {c}')
    print(f't_sum: {ts}')
    print(f't_min: {tm}')
    print(f'commands: {comms}')
    print(f'rew: {r}')
    print('-----------------')
