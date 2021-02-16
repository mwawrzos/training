
import argparse
import os

import torch
from common.data.dataset import AudioDataset, get_data_loader
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.text import Tokenizer
from common.data import features
from common.helpers import print_once
from rnnt import config


def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--model_config', default='configs/baseline_v3-1023sp.yaml',
                    type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    args.train_manifests = [os.path.join(args.dataset_dir, train_manifest)
                            for train_manifest in ['librispeech-train-clean-100-wav.json',
                                                   'librispeech-train-clean-360-wav.json',
                                                   'librispeech-train-other-500-wav.json']]

    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    print_once('Setting up datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
    ) = config.input(cfg, 'train')

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    train_feat_proc = torch.nn.Sequential(
        features.FilterbankFeatures(optim_level=0, **train_features_kw),
        train_specaugm_kw and features.SpecAugment(optim_level=0, **train_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=0, **train_splicing_kw),
        features.FillPadding(optim_level=0, ),
    )

    use_dali = True

    if use_dali:

        sampler = dali_sampler.SimpleSampler()

        train_loader = DaliDataLoader(gpu_id=None,
                                      dataset_path=args.dataset_dir,
                                      config_data=train_dataset_kw,
                                      config_features=train_features_kw,
                                      json_names=args.train_manifests,
                                      batch_size=4,
                                      sampler=sampler,
                                      grad_accumulation_steps=1,
                                      pipeline_type='train',
                                      device_type="cpu",
                                      tokenizer=tokenizer)
    else:
        world_size = 1
        batch_size = 4
        args.local_rank = 0
        args.num_buckets = 1
        train_dataset = AudioDataset(args.dataset_dir,
                                     args.train_manifests,
                                     tokenizer=tokenizer,
                                     **train_dataset_kw)
        train_loader = get_data_loader(train_dataset,
                                       batch_size,
                                       world_size,
                                       args.local_rank,
                                       num_buckets=args.num_buckets,
                                       shuffle=True,
                                       num_workers=4)

    for batch in train_loader:
        audio, audio_lens, txt, txt_lens = batch
        feats, feat_lens = train_feat_proc([audio, audio_lens])
        print("Audio shape: {}".format(audio.shape))
        print("Audio-lens shape: {}".format(audio_lens.shape))
        print("Txt shape: {}".format(txt.shape))
        print("Txt-lens shape: {}".format(txt_lens.shape))
        print("Feat shape: {}".format(feats.shape))
        print("Feat-lens shape: {}".format(feat_lens.shape))

