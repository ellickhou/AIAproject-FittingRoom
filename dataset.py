#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
from PIL import Image
from PIL import ImageDraw
import numpy as np
import json
import pickle

class CPDataset(data.Dataset):

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode
        self.stage = opt.stage # GMM or cycleTryOn
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.uselimbs = opt.uselimbs
        self.useSCHP = opt.useSCHP
        
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        self.transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        
        # load data list
        im_names = []
        cloth_names = []


        with open(opt.data_list, 'r') as f:
            for line in f.readlines():
                im_name, cloth_name = line.strip().split()
                im_names.append(im_name)
                cloth_names.append(cloth_name)


        self.im_names = im_names
        self.cloth_names = cloth_names
        
    def name(self):
        return "CPDataset_new"

    def __getitem__(self, index):

        im_name = self.im_names[index]  # the index of the pair
        cloth_name = self.cloth_names[index]  # condition person index

        
        # cloth image & cloth mask
        
        if self.stage == 'cycleTryOn':
            warp_cloth_path = osp.join(self.data_path, 'warp-cloth', cloth_name)
            warp_cloth = Image.open(warp_cloth_path)
            warp_cloth = self.transform(warp_cloth)  # [-1,1]
         
        
        # cloth you want
        cloth_path = osp.join(self.data_path, 'cloth', cloth_name)
        cloth = Image.open(cloth_path)
        cloth = self.transform(cloth)  # [-1,1]
        
        # cloth_mask you want
        cloth_mask_path = osp.join(self.data_path, 'cloth-mask', cloth_name)
        cloth_mask = Image.open(cloth_mask_path)
        cloth_mask_array = np.array(cloth_mask)
        cloth_mask_array = (cloth_mask_array > 0).astype(np.float32)
        
        cloth_mask = torch.from_numpy(cloth_mask_array) # [0,1]
        cloth_mask = cloth_mask.unsqueeze_(0)
        
        
        # person image 
        im_path = osp.join(self.data_path, 'image', im_name)
        
        im = Image.open(im_path)
        im = self.transform(im) # [-1,1]

        # parsing image
        if self.useSCHP:
            parse_name = im_name.split('.')[0] + 'SCHP.png'
        else:
            parse_name = im_name.replace('.jpg', '.png')
        

        parse_path = osp.join(self.data_path, 'image-parse', parse_name)
        
        im_parse = Image.open(parse_path)

        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + \
                      (parse_array == 6).astype(np.float32) + \
                      (parse_array == 7).astype(np.float32)
        
        
        
        if self.uselimbs:
            parse_human_parts = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 4).astype(np.float32) + \
                                (parse_array == 13).astype(np.float32) + \
                                (parse_array == 14).astype(np.float32) + \
                                (parse_array == 15).astype(np.float32) + \
                                (parse_array == 16).astype(np.float32) + \
                                (parse_array == 17).astype(np.float32) + \
                                (parse_array == 9).astype(np.float32) + \
                                (parse_array == 12).astype(np.float32)
        else:
            parse_human_parts = (parse_array == 1).astype(np.float32) + \
                                (parse_array == 2).astype(np.float32) + \
                                (parse_array == 4).astype(np.float32) + \
                                (parse_array == 13).astype(np.float32)

        ## shape downsample
        parse_shape = Image.fromarray((parse_shape*255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.fine_width//16, self.fine_height//16),Image.BILINEAR)
        parse_shape = parse_shape.resize((self.fine_width, self.fine_height),Image.BILINEAR)

        ## done
        im_shape = self.transform(parse_shape) # [-1,1]
        parse_cloth_mask = torch.from_numpy(parse_cloth) # [0,1]

        # upper cloth
        ## done
        im_cloth = im * parse_cloth_mask + (1 - parse_cloth_mask) # [-1,1], fill 1 for other parts
        
        
        
        p_hp = torch.from_numpy(parse_human_parts) # [0,1]
        im_hp = im * p_hp - (1 - p_hp) # [-1,1], fill 0 for other parts


        # load pose points

        if 's' in im_name or 'a' in im_name:
            pose_name = im_name.replace('.jpg', '.pkl')
            pose_path = osp.join(self.data_path, 'pose', pose_name)

            pose_data = - np.ones((18, 2), dtype=int)
            with open(pose_path, 'rb') as f:
                pose_label = pickle.load(f)
                for i in range(18):
                    if pose_label['subset'][0, i] != -1:
                        pose_data[i, :] = pose_label['candidate'][int(pose_label['subset'][0, i]), :2]
                pose_data = np.asarray(pose_data)         
            
        else:
            pose_name = im_name.replace('.jpg', '_keypoints.json')
            pose_path = osp.join(self.data_path, 'pose', pose_name)
            
            with open(pose_path, 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1,3))

            
            
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        pose_image = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(pose_image)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = self.transform(one_map)
            pose_map[i] = one_map[0]


        # just for visualization
        pose_image = self.transform(pose_image)

        # cloth-agnostic representation
        agnostic = torch.cat([im_shape, pose_map], 0)


        if self.stage == 'GMM':
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g)
        else:
            im_g = ''
#         c_cloth = c_cloth.view(1, 256, 192)


        if self.opt.stage == "GMM":
            result = {
                #### input
                'cloth': cloth,
                'target_mask': im_shape,
                'pose': pose_map,
                #### ground_truth
                'im_upper_cloth': im_cloth,
                #### for vis
                'cloth_name': cloth_name,
                'im_name' : im_name,
                'cloth_mask': cloth_mask,
                'human_parts': im_hp,
                'grid_image': im_g,
                'image': im,
                'pose_image': pose_image,

            }
            return result


        result = {
            'im_name': im_name,
            'cloth_name': cloth_name,
            # cycleGAN input
            # cloth you want
            'cloth': cloth,
            'warp_cloth': warp_cloth,
            'parse_human_parts': p_hp,
            'human_parts': im_hp,
            # img
            'image': im,  # c
            # cloth on img
            'im_upper_cloth': im_cloth,
            # body shape of img
            'im_shape': im_shape,
            # pose of img
            'pose': pose_map,
            # cloth mask on img
            'im_upper_mask': parse_cloth_mask,
            # vis
            'pose_image': pose_image,
                        }
        
        return result


    def __len__(self):
        return len(self.im_names)

class CPDataLoader(object):
    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        if opt.shuffle :
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()
       
    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM", choices=['GMM', 'cycleTryOn'], help='stage you want')
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1)

    
    
    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()

