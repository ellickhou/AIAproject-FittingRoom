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

def train_gmm(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        
        
        # inputs
        pose = inputs['pose'].cuda()
        target_mask = inputs['target_mask'].cuda()
        cloth = inputs['cloth'].cuda()
        
        # ground_truth
        im_upper_cloth = inputs['im_upper_cloth'].cuda()
        
        agnostic = torch.cat([target_mask, pose],1)
        grid, theta = model(agnostic, cloth)
        
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        
        loss = criterionL1(warped_cloth, im_upper_cloth)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if (step+1) % opt.display_count == 0:
            # save board
            board.add_scalar('L1_warped_truth', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        
        if (step+1) % opt.save_count == 0:
            # save checkpoint
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))
            
            # for vis
            human_parts = inputs['human_parts'].cuda()
            image = inputs['image'].cuda()
            im_g = inputs['grid_image'].cuda()
            cloth_mask = inputs['cloth_mask'].cuda()
            pose_image = inputs['pose_image'].cuda()

            warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')

            visuals = [[human_parts, target_mask, pose_image],
                       [cloth, warped_cloth, im_upper_cloth],
                       [warped_grid, (warped_cloth+image)*0.5, image]]
            board_add_images(board, f'visuals{step+1}', visuals, step+1)

    
    
def train_cycleTryOn(opt, train_loader, model, board):

    model.cuda()
    model.train()



    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        
        inputs = train_loader.next_batch()
        
        model.set_input(inputs)
        model.optimize_parameters()
        results = model.current_results()
        
        gen_B = results['gen_B']
        
        
        if (step+1) % opt.display_count == 0:
 
            
            # loss
            board.add_scalar('total_G_loss', results['total_G_loss'].item(), step+1)
            board.add_scalar('attention_loss', results['attention_loss'].item(), step+1)
            board.add_scalar('content_lossB', results['content_lossB'].item(), step+1)
            board.add_scalar('content_lossA', results['content_lossA'].item(), step+1)
            board.add_scalar('G_B_loss', results['G_B_loss'].item(), step+1)
            board.add_scalar('G_A_loss', results['G_A_loss'].item(), step+1)
            board.add_scalar('total_D_loss', results['total_D_loss'].item(), step+1)
            board.add_scalar('D_B_loss', results['D_B_loss'].item(), step+1)
            board.add_scalar('D_A_loss', results['D_A_loss'].item(), step+1)
            
            t = time.time() - iter_start_time
            print('step: %8d, total_G_loss: %4f, total_D_loss: %4f, time: %.3f' % \
                  (step + 1, results['total_G_loss'].item(),\
                   results['total_D_loss'].item(), t), flush=True)
            
            
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
                               
            # for vis
            im_name = inputs['im_name']
            
            human_parts = inputs['human_parts']
            im_shape = inputs['im_shape'].cuda()
            pose_image = inputs['pose_image']

            cloth = results['cloth']
            warp_cloth = inputs['warp_cloth'].cuda()
            im_upper_cloth = inputs['im_upper_cloth'].cuda()
            
            
            image = results['img']
            gen_A = results['gen_A']
            
            visuals = [[human_parts, im_shape, pose_image],
                       [cloth, warp_cloth, im_upper_cloth],
                       [image, gen_B, gen_A]]
            
            board_add_images(board, f'combine{step+1}', visuals, step+1)





def main():
    opt = get_opt()
    print(opt)
    print("named: %s!" % (opt.name))
    # create dataset
    train_dataset = CPDataset(opt)
    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))
    
    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
        
        
    elif opt.stage == 'cycleTryOn':
        generator = cyclegan(opt)
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_cycleTryOn(opt, train_loader, generator, board)
        save_checkpoint(generator, os.path.join(opt.checkpoint_dir, opt.name, 'cycleTryOn_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)


    print('Finished training %s, nameed: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()
