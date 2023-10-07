# Unet_pytorch
Unet_Model is the file where the the actual structure of the Unet along with data generator and training method is defined
network_blocks contain the building blocks of the U_net(expansion, and redution methods)
Data_transform contains data augmentation transformations

I am getting the following error, I would really grateful if you could help me out on this. Thanks
Traceback (most recent call last):
  File "/NFSHOME/mjawaid/U_net/pytorch_moded/Unet_Model.py", line 282, in <module>
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  File "/NFSHOME/mjawaid/miniconda3/envs/PY39-CUDA/lib/python3.9/site-packages/torch/optim/adam.py", line 33, in __init__
    super().__init__(params, defaults)
  File "/NFSHOME/mjawaid/miniconda3/envs/PY39-CUDA/lib/python3.9/site-packages/torch/optim/optimizer.py", line 187, in __init__
    raise ValueError("optimizer got an empty parameter list")
ValueError: optimizer got an empty parameter list
