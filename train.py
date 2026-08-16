import sys
sys.path.append('droid_slam')

import socket
import time
from datetime import timedelta
from loguru import logger
import cv2
import numpy as np
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_readers.factory import dataset_factory

from lietorch import SO3, SE3, Sim3
from geom import losses
from geom.losses import geodesic_loss, residual_loss, flow_loss
from geom.graph_utils import build_frame_graph

# network
from droid_net import DroidNet
from logger import Logger, SUM_FREQ

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(gpu, args):
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=args.world_size,
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    logger.info(f"[rank {dist.get_rank()}/{dist.get_world_size()}] "
                f"host={socket.gethostname()} gpu={gpu} ({torch.cuda.get_device_name(gpu)}) "
                f"backend={dist.get_backend()}")

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey()

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)

    N = args.n_frames
    model = DroidNet()
    model.cuda()
    model.train()

    model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location=f'cuda:{gpu}')
        for key in ["module.update.weight.2.weight", "module.update.weight.2.bias",
                    "module.update.delta.2.weight", "module.update.delta.2.bias"]:
            if key in state_dict and state_dict[key].shape[0] > 2:
                logger.warning(f"Slicing checkpoint tensor '{key}' from "
                               f"{tuple(state_dict[key].shape)} to first 2 output channels.")
                state_dict[key] = state_dict[key][:2]
        model.load_state_dict(state_dict)

    # fetch dataloader
    db = dataset_factory(args.datasets, datapath=args.datapath, n_frames=args.n_frames, fmin=args.fmin, fmax=args.fmax,
                         aug_photo=not args.no_aug_photo, aug_crop=not args.no_aug_crop)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        db, shuffle=True, num_replicas=args.world_size, rank=gpu)

    train_loader = DataLoader(db, batch_size=args.batch, sampler=train_sampler, num_workers=2)

    if args.epochs is not None:
        args.steps = int(len(db) * args.epochs / (args.world_size * args.batch))
        logger.info(f"[rank {gpu}] --epochs={args.epochs} => steps={args.steps} "
                    f"(M={len(db)} anchors, world_size={args.world_size}, batch={args.batch})")

    # fetch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        args.lr, args.steps, pct_start=0.01, cycle_momentum=False)

    train_logger = Logger(args.name, scheduler)
    should_keep_training = True
    total_steps = 0
    epoch = 0

    train_start_time = time.time()
    avg_step_time = 0.0

    logger.info("=" * 60)
    logger.info(f"[train] start train loop")
    logger.info(f"[rank {gpu}] dataset: {len(db)} anchors total, "
                f"{len(train_sampler)} assigned to this rank, "
                f"{len(train_loader)} batches/pass (batch_size={args.batch})")
    logger.info("=" * 60)
    while should_keep_training:
        train_sampler.set_epoch(epoch)
        epoch += 1
        for i_batch, item in enumerate(train_loader):
            step_start_time = time.time()
            optimizer.zero_grad()

            images, poses, disps, intrinsics = [x.to('cuda') for x in item]

            # convert poses w2c -> c2w
            Ps = SE3(poses).inv()
            Gs = SE3.IdentityLike(Ps)

            # randomize frame graph
            if rng.random() < 0.5:
                graph = build_frame_graph(poses, disps, intrinsics, num=args.edges)

            else:
                graph = OrderedDict()
                for i in range(N):
                    graph[i] = [j for j in range(N) if i!=j and abs(i-j) <= 2]

            # fix first to camera poses
            Gs.data[:,0] = Ps.data[:,0].clone()
            Gs.data[:,1:] = Ps.data[:,[1]].clone()
            disp0 = torch.ones_like(disps[:,:,3::8,3::8])

            # perform random restarts (always runs at least once, then repeats with
            # probability restart_prob)
            while True:
                intrinsics0 = intrinsics / 8.0
                poses_est, disps_est, residuals = model(Gs, images, disp0, intrinsics0,
                    graph, num_steps=args.iters, fixedp=2)

                geo_loss, geo_metrics = losses.geodesic_loss(Ps, poses_est, graph, do_scale=False)
                res_loss, res_metrics = losses.residual_loss(residuals)
                flo_loss, flo_metrics = losses.flow_loss(Ps, disps, poses_est, disps_est, intrinsics, graph)

                loss = args.w1 * geo_loss + args.w2 * res_loss + args.w3 * flo_loss

                # consistency branch: second forward on perturbed images from the
                # same (Gs, disp0) initialization, penalize pose disagreement
                if args.w4 > 0:
                    images_pert = (images + args.noise_sigma * torch.randn_like(images)).clamp(0.0, 255.0)
                    poses_est2, _, _ = model(Gs, images_pert, disp0, intrinsics0,
                        graph, num_steps=args.iters, fixedp=2)

                    con_loss, con_metrics = losses.consistency_loss(poses_est, poses_est2, graph)
                    loss = loss + args.w4 * con_loss

                loss.backward()

                Gs = poses_est[-1].detach()
                disp0 = disps_est[-1][:,:,3::8,3::8].detach()

                if rng.random() >= args.restart_prob:
                    break

            metrics = {}
            metrics.update(geo_metrics)
            metrics.update(res_metrics)
            metrics.update(flo_metrics)
            if args.w4 > 0:
                metrics.update(con_metrics)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            total_steps += 1

            step_time = time.time() - step_start_time
            avg_step_time += (step_time - avg_step_time) / total_steps

            if gpu == 0:
                train_logger.push(metrics)

                if total_steps % SUM_FREQ == 0:
                    elapsed = time.time() - train_start_time
                    steps_left = args.steps - total_steps
                    eta = avg_step_time * steps_left
                    logger.info(f"[step {total_steps}/{args.steps}] "
                                f"time/step={avg_step_time:.3f}s "
                                f"elapsed={timedelta(seconds=int(elapsed))} "
                                f"eta={timedelta(seconds=int(eta))} "
                                f"steps_left={steps_left}")

            if total_steps % 10000 == 0 and gpu == 0:
                PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
                torch.save(model.state_dict(), PATH)

            if total_steps >= args.steps:
                should_keep_training = False
                break

    if gpu == 0:
        PATH = 'checkpoints/%s_%06d.pth' % (args.name, total_steps)
        torch.save(model.state_dict(), PATH)
        logger.info(f"Saved final checkpoint to '{PATH}'")

    dist.destroy_process_group()


if __name__ == '__main__':
    logger.info("Executing train.py ...")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datasets', nargs='+', help='lists of datasets for training')
    parser.add_argument('--datapath', default='datasets/TartanAir', help="path to dataset directory")
    parser.add_argument('--gpus', type=int, default=None, help='number of GPUs to use (default: all available)')

    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=15)
    parser.add_argument('--steps', type=int, default=250000)
    parser.add_argument('--epochs', type=float, default=None,
                         help='if set, overrides --steps: steps = len(dataset) * epochs / (world_size * batch)')
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--wd', type=float, default=1e-5, help='Adam weight decay')
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--n_frames', type=int, default=7)

    parser.add_argument('--w1', type=float, default=10.0)
    parser.add_argument('--w2', type=float, default=0.01)
    parser.add_argument('--w3', type=float, default=0.05)
    parser.add_argument('--w4', type=float, default=0.0,
                         help='consistency loss weight: if > 0, adds a second forward pass on '
                              'noise-perturbed images and penalizes pose disagreement between the '
                              'two branches (roughly doubles training memory/compute)')
    parser.add_argument('--noise_sigma', type=float, default=5.0,
                         help='stddev of the Gaussian image noise for the consistency branch, '
                              'in 0-255 intensity units (only used when --w4 > 0)')

    parser.add_argument('--fmin', type=float, default=8.0)
    parser.add_argument('--fmax', type=float, default=96.0)
    parser.add_argument('--no_aug_photo', action='store_true',
                         help='disable photometric augmentation (color jitter / random grayscale)')
    parser.add_argument('--no_aug_crop', action='store_true',
                         help='fixed minimal resize before the center crop instead of a random scale')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--edges', type=int, default=24)
    parser.add_argument('--restart_prob', type=float, default=0.2)

    args = parser.parse_args()

    available_gpus = torch.cuda.device_count()
    if args.gpus is None:
        args.gpus = available_gpus
        logger.info(f"--gpus not set; using all {available_gpus} available GPU(s).")
    elif args.gpus > available_gpus:
        logger.error(f"Requested --gpus={args.gpus} but only {available_gpus} GPU(s) are available.")
        raise SystemExit(1)
    elif args.gpus < available_gpus:
        logger.warning(f"--gpus={args.gpus} but {available_gpus} GPU(s) are available; "
                       f"{available_gpus - args.gpus} will be left unused.")

    args.world_size = args.gpus

    logger.info("Training arguments:")
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest)
        logger.info(f"  {action.dest:<15} {str(value):<20} {action.help or ''}")
    logger.info(f"  {'world_size':<15} {str(args.world_size):<20} number of GPUs used for distributed training")

    import os
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

