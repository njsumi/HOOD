#!/bin/bash
#SBATCH -A rayyeh
#SBATCH -p a10
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH --job-name=p3d-install
#SBATCH --output=p3d-install-%j.out

module reset
module load gcc/13.3.0
module load cuda/12.6.0
module load external conda/2025.09
source activate hood
export PYTHONNOUSERSITE=1

nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export MAX_JOBS=8
pip install iopath fvcore
pip install --no-build-isolation -v "git+https://github.com/facebookresearch/pytorch3d.git@stable"

python - <<'EOF'
import torch, pytorch3d
from pytorch3d.ops import knn_points
print('pytorch3d', pytorch3d.__version__)
p1 = torch.randn(1, 5000, 3, device='cuda'); p2 = torch.randn(1, 13776, 3, device='cuda')
_, idx, _ = knn_points(p1, p2, return_nn=True)
print('knn matches brute force:', (idx[..., 0] == torch.cdist(p1, p2).argmin(-1)).float().mean().item())
EOF
