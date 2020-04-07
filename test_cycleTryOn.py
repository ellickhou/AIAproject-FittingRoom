# coding=utf-8
import argparse
import os
import os.path as osp
from PIL import Image
import time
import torch.nn.functional as F


from dataset import CPDataset, CPDataLoader
from utils import *
from networks import *

from tensorboardX import SummaryWriter
from visualization import board_add_image, board_add_images


def get_opt():
    # basic
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "cycleTryOn")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--uselimbs", action='store_true', help='use lambs or not')
    parser.add_argument("--useSCHP", action='store_true', help='use SCHP for human parsing result or not')
    # dir
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM", choices=['GMM', 'cycleTryOn'], help='stage you want')
    parser.add_argument("--data_list", default = "train_pair.txt")
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    # model condition
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    # train 
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--gan_mode", type=str, default='lsgan')
    opt = parser.parse_args()
    return opt


def test_gmm(opt, test_loader, model, board):
    # criterion
    model.cuda()
    model.eval()
    
    cp_name = osp.dirname(opt.checkpoint).split('/')[-1]
    save_dir = osp.join(opt.result_dir, cp_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)

    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)
        
    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()
        # inputs
        pose = inputs['pose'].cuda()
        target_mask = inputs['target_mask'].cuda()
        cloth = inputs['cloth'].cuda()
        # eval
        agnostic = torch.cat([target_mask, pose],1)
        grid, theta = model(agnostic, cloth)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        
        # ground_truth
        t_upper_cloth = inputs['im_upper_cloth'].cuda()
        # for vis
        cloth_names = inputs['cloth_name']
        human_parts = inputs['human_parts'].cuda()
        image = inputs['image'].cuda()
        im_g = inputs['grid_image'].cuda()
        cloth_mask = inputs['cloth_mask'].cuda()
        pose_image = inputs['pose_image'].cuda()

        warped_mask = F.grid_sample(cloth_mask, grid, padding_mode='zeros')
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
    
        visuals = [[human_parts, target_mask, pose_image],
                   [cloth, warped_cloth, t_upper_cloth],
                   [warped_grid, (warped_cloth+image)*0.5, image]]

        
        save_images(warped_cloth, cloth_names, warp_cloth_dir)
        save_images(warped_mask*2-1, cloth_names, warp_mask_dir)
        
        
        if (step+1) % opt.display_count == 0:
            board_add_images(board, f'combine{step+1}', visuals, step+1)
            t = time.time() - iter_start_time
#             print('step: %8d, time: %.3f' % (step+1, t), flush=True)
    
def test_cycleTryOn(opt, train_loader, model, board):

    model.cuda()
    model.eval()
    
    cp_name = osp.dirname(opt.checkpoint).split('/')[-1]
    save_dir = osp.join(opt.result_dir, cp_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gen_img_dir = os.path.join(save_dir, 'Gen_images')
    
    if not os.path.exists(gen_img_dir):
        os.makedirs(gen_img_dir)
     
    for step, inputs in enumerate(train_loader.data_loader):
        iter_start_time = time.time()
        
        inputs = train_loader.next_batch()
        
        model.set_input(inputs)
        model.optimize_parameters()
        results = model.current_results()
        
        im_name = inputs["im_name"]
        gen_B = results['gen_B']
        save_images(gen_B, im_name, gen_img_dir)
        
        # for vis
        im_name = inputs["im_name"]
        human_parts = inputs['human_parts']
        im_shape = inputs['im_shape'].cuda()
        pose_image = inputs['pose_image']

        cloth = results['cloth']
        warp_cloth = inputs["warp_cloth"].cuda()
        im_upper_cloth = inputs['im_upper_cloth'].cuda()
        

        image = results['img']
        gen_A = results['gen_A']

        visuals = [[human_parts, im_shape, pose_image],
                     [cloth, warp_cloth, im_upper_cloth],
                   [image, gen_B, gen_A]] 
        board_add_images(board, f'combine{step+1}', visuals, step+1)   
        
        t = time.time() - iter_start_time
        print('step: %8d, time: %.3f' % (step + 1, t), flush=True)
        
#         if (step + 1) % opt.save_count == 0:


def main():
    opt = get_opt()
    print(opt)
    print("named: %s!" % (opt.name))

    # create dataset
    train_dataset = CPDataset(opt)
    generator = cyclegan(opt)
    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        test_gmm(opt, train_loader, model, board)
        
        
    elif opt.stage == 'cycleTryOn':
        generator = cyclegan(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(generator, opt.checkpoint)
        test_cycleTryOn(opt, train_loader, generator, board)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()

