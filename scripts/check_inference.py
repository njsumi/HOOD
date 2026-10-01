import os
import sys
import torch
import numpy as np

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_making import convert_vto_to_pkl
from utils.defaults import DEFAULTS
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump

VTO_DATASET_PATH = Path(os.environ['RCAC_SCRATCH']) / 'vto-dataset/'
vto_sequence_path = Path(VTO_DATASET_PATH) / 'tshirt/simulations/tshirt_shape00_01_01.pkl'
target_pkl_path =  Path(os.environ['HOOD_DATA']) / 'temp/01_01.pkl'
out_path = Path(DEFAULTS.data_root) / 'temp' / 'output.pkl'

# convert one VTO sequence to HOOD format
convert_vto_to_pkl(vto_sequence_path, target_pkl_path, n_zeropose_interpolation_steps=30)
print(f'Pose sequence saved into {target_pkl_path}')

garment_name = 'tshirt'

config_dict = dict()
config_dict['density'] = 0.20022
config_dict['lame_mu'] = 23600.0
config_dict['lame_lambda'] = 44400
config_dict['bending_coeff'] = 3.962e-05
config_dict['separate_arms'] = True
config_dict['garment_dict_file'] = 'garments_dict.pkl'
config_dict['smpl_model'] = 'smpl/SMPL_FEMALE.pkl'

validation_config = ValidationConfig(**config_dict)

config_name = 'postcvpr'
checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'

# load pretrained model
modules, experiment_config = load_params(config_name)
experiment_config = update_config_for_validation(experiment_config, validation_config)

runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)
print('model loaded')

dataloader = create_one_sequence_dataloader(target_pkl_path, garment_name, modules, experiment_config)

with torch.no_grad():
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')
    trajectories_dict = runner.valid_rollout(sequence, bare=True)

pickle_dump(dict(trajectories_dict), out_path)
print(f"Rollout saved into {out_path}")

pred, body = np.asarray(trajectories_dict['pred']), np.asarray(trajectories_dict['obstacle'])
print('pred shape (frames, cloth verts, 3):', pred.shape)
print('all finite:', np.isfinite(pred).all())
step = np.linalg.norm(np.diff(pred, axis=0), axis=-1)
print(f'per-frame motion: median {np.median(step)*100:.2f} cm, max {step.max()*100:.2f} cm')
print(f'cloth-body center distance (last frame): {np.linalg.norm(pred[-1].mean(0) - body[-1].mean(0))*100:.1f} cm')
