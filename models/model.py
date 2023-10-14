import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
import pdb
import data.util.panorama as panorama


class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data, label=None):
        ''' must use set_device in tensor '''
        if not label is None:
            self.label = self.set_device(label)
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])

    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        #if self.task in ['inpainting','uncropping']:
        #    ret_path.extend(['Mask_{}'.format(name) for name in self.path])
        #    ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()
    
    def save_iter_results(self, iter):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('{}_GT'.format(iter))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('{}_Process'.format(iter))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('{}_Out'.format(iter))
            ret_result.append(self.visuals[idx - self.batch_size].detach().float().cpu())

        #if self.task in ['inpainting', 'uncropping']:
        #    ret_path.extend(['{}_Mask'.format(name) for name in self.path])
        #    ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data, labels in tqdm.tqdm(self.phase_loader, desc="epoch {}".format(self.epoch)):
            self.set_input(train_data, labels)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, label=self.label)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
 
        #self.sample(train_data, self.get_cond(theme=0, difficulty=2))
        #self.gen_panorama(input=panorama.gen_rand_input(), iter=5, cond=self.get_cond(theme=0))

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
   
    def get_cond(self, theme=None, gamestyle=None, difficulty=None):
        #cond_theme = torch.zeros((10,), dtype=torch.long)
        #cond_gs = torch.zeros((5,), dtype=torch.long)
        #cond_df = torch.zeros((4,), dtype=torch.long)
        cond_theme = theme if theme is not None else -1
        cond_gs = gamestyle if gamestyle is not None else -1
        cond_df = difficulty if difficulty is not None else -1
        #cond = torch.cat((cond_theme, cond_gs, cond_df)) # 1x19
        # cond = (torch.ones(1)*theme).long()
        cond = torch.tensor([[cond_theme], [cond_df], [cond_gs]]).long()
        #cond = (torch.ones(1)*theme).long()
        return cond

    def get_cond_dict(self, cond):
        dic = {}
        theme, gamestyle, difficulty = cond[0].item(), cond[2].item(), cond[1].item()
        if theme != -1:
            dic['theme'] = theme
        if gamestyle != -1:
            dic['gamestyle'] = gamestyle
        if difficulty != -1:
            dic['difficulty'] = difficulty
 
        return dic

    def sample(self, data, label):
        count = len(data['gt_image'])
        self.logger.info("\n\n\n--------------------------Sampling {} images------------------------------".format(count))

        dict_list = []
        for i in range(count):
            dic = {'gt_image': data['gt_image'][i][None], 'cond_image': data['cond_image'][i][None],
                   'mask_image': data['mask_image'][i][None], 'mask': data['mask'][i][None], 'path': [data['path'][i]]}

            dict_list.append(dic)

        self.netG.eval()
        with torch.no_grad():
            for dic in dict_list:
                self.set_input(dic, label)
                if self.opt['distributed']:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals, _ = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                                                                                    y_0=self.gt_image, mask=self.mask,
                                                                                    sample_num=self.sample_num)
                    else:
                        self.output, self.visuals, _ = self.netG.module.restoration(self.cond_image,
                                                                                    sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting', 'uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                                                                          y_0=self.gt_image, mask=self.mask,
                                                                          sample_num=self.sample_num, label=self.label)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                #suffix = ''
                #if label is not None:
                #    cond = label
                #    print(cond.shape)
                #    print(cond)
                #    suffix = '_' 
                #    dic = self.get_cond_dict(cond) 
                #    for item in dic:
                #        if item == 'theme':
                #            suffix += f"t{dic[item]}"
                #        elif item == 'difficulty':
                #            suffix += f"d{dic[item]}"
                #        elif item == 'gamestyle':
                #            suffix += f"g{dic[item]}"
                self.writer.save_images(self.save_current_results(), self.tileset, self.spritesheet, filename=self.path[0])
        self.logger.info("\n\n\n------------------------------Sampling End------------------------------")


    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(),filename=self.path[0])

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
            self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
    
    def uncrop(self, data, iter, label=None):
        self.set_input(data, label)
        self.output, self.visuals = self.netG.restoration(self.cond_image,
                                                      y_t=self.cond_image,
                                                      y_0=self.gt_image, mask=self.mask,
                                                      sample_num=self.sample_num, label=self.label)
        self.writer.save_images(self.save_iter_results(iter), self.tileset, self.spritesheet)
        return self.output

    # return (16, 0.5+0.5*iter*width) tensor
    def left_uncrop(self, start, iter, filename, cond=None):
        res = torch.squeeze(start['gt_image'])[:, 8:]
        for i in range(iter):
            ret = torch.squeeze(self.uncrop(start, i, cond)).detach().float().cpu()
            res = torch.cat((ret[:, 0:8], res), 1)
            self.cur_panorama[:, self.center_start-(i+1)*8:self.center_start-i*8] = ret[:, 0:8]
            self.progress = torch.cat((self.progress, self.cur_panorama[None, None]), dim=0)
            start = panorama.gen_starting_point((res[:, 0:16], filename))
        return res

    # return (16, 0.5+0.5*iter*width) tensor
    def right_uncrop(self, start, iter, filename, cond=None):
        res = torch.squeeze(start['gt_image'])[:, 0:8]
        for i in range(iter):
            ret = torch.squeeze(self.uncrop(start, i+iter, cond)).detach().float().cpu()
            res = torch.cat((res, ret[:, 8:]), 1)
            self.cur_panorama[:, self.center_end+i*8:self.center_end+(i+1)*8] = ret[:, 0:8]
            self.progress = torch.cat((self.progress, self.cur_panorama[None, None]), dim=0)
            start = panorama.gen_starting_point((res[:, -16:], filename), is_left=False)
        return res
        
    def panorama(self, cond=None):
        self.gen_panorama(input=panorama.gen_rand_input(), iter=5, cond=cond)


    '''
    input: tuple(16x16 ndarray, filename)
    iter: number of iteration      
    '''
    def gen_panorama(self, input, iter=5, cond=None):
        ndarray = input[0]  # (16,16,1) ndarray
        filename = input[1].replace(".txt", "")
        size = ndarray.shape              # WARNING: only works when height and width are the same and even
        h, w = size[0], size[1]

        # dicts
        raw_input = panorama.gen_starting_point(input, is_origin=True)    # input data
        lstart = panorama.gen_starting_point((raw_input['gt_image'], filename), is_left=True)         # left uncrop start
        rstart = panorama.gen_starting_point((raw_input['gt_image'], filename), is_left=False)        # right uncrop start

        # store panorama generation progress, should have 2*iter elements
        self.cur_panorama = torch.ones(h, (iter+1)*w) * (-1.0)

        # initialze input at center
        self.center_start = int(0.5*w*iter)
        self.center_end = self.center_start + w
        self.cur_panorama[:, self.center_start:self.center_end] = torch.squeeze(raw_input['gt_image'])[:]
        self.progress = self.cur_panorama[None, None]

        self.netG.eval()
        with torch.no_grad():
            left = self.left_uncrop(lstart, iter, filename, cond)
            right = self.right_uncrop(rstart, iter, filename, cond)
        res = torch.torch.cat((left, right), 1)
        suffix = ''
        if cond is not None:
            suffix = '_'
            dic = self.get_cond_dict(cond)
            for item in dic:
                if item == 'theme':
                    suffix += f"t{dic[item]}"
                elif item == 'difficulty':
                    suffix += f"d{dic[item]}"
                elif item == 'gamestyle':
                    suffix += f"g{dic[item]}"
        self.writer.save_panorama(raw_input, res[None], self.progress, self.tileset, self.spritesheet, suffix=suffix)      
