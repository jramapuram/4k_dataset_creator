# GPU Resizer

This simple program uses pytorch to resize images to whatever size you want using either bilinear, trilinear or nearest-neighbor upsampling.
Functionality is provided by pytorch upsample call. Batch size allows to parallel upsample the dataset.

## Custom datasets

Add your dataset to https://github.com/jramapuram/datasets (or your own fork of it) and reparameterize loader.py


## Parameters

```python
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--data-dir', type=str, default='./original_dataset',
                    help='directory which contains input data')
parser.add_argument('--dest-dir', type=str, default='./resized_dataset',
                    help='directory which will hold the 4k dataset')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size (default: 64)')
parser.add_argument('--w', type=int, default=4000,
                    help='width to upsample to (default: 4000)')
parser.add_argument('--h', type=int, default=4000,
                    help='height to upsample to (default: 4000)')
parser.add_argument('--mode', type=str, default='bilinear',
                    help='upsample mode (default: bilinear)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA')
```
