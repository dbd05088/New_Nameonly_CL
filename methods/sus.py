# When we make a new one, we should inherit the Finetune class.
import logging
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from methods.zero_shot_clip import ZeroShotClip

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")




class SUS(ZeroShotClip):
    def __init__(self,  train_datalist, test_datalist, device, **kwargs):
        # if kwargs["temp_batchsize"] is None:
        #     kwargs["temp_batchsize"] = kwargs["batchsize"]//2
        super().__init__(train_datalist, test_datalist, device, **kwargs)
    
    def online_evaluate(self, domain_name, cur_task, test_list, sample_num, batch_size, n_worker, few_shot_list, val_list):
        self.cur_task = cur_task
        
        test_df = pd.DataFrame(test_list)
        few_shot_df = pd.DataFrame(few_shot_list)
        val_df = pd.DataFrame(val_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        exp_few_shot_df = few_shot_df[few_shot_df['klass'].isin(self.exposed_classes)]
        exp_val_df = val_df[val_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageTextDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            # prompt_cls_list=self.text_class_prompts
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        fewshot_dataset = ImageTextDataset(
            exp_few_shot_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            # prompt_cls_list=self.text_class_prompts
        )
        fewshot_loader = DataLoader(
            fewshot_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        val_dataset = ImageTextDataset(
            exp_val_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            # prompt_cls_list=self.text_class_prompts
        )
        val_loader = DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        
        test_features, test_labels = self.get_features(test_loader)
        support_features, support_labels = self.get_features(fewshot_loader)
        val_features, val_labels = self.get_features(val_loader)
        text_classifier_weights = self.get_text_classifier_weights()
        
        train_kl_divs_sims, test_kl_divs_sims, val_kl_divs_sims = self.get_kl_div_sims(args, test_features, val_features, support_features, text_classifier_weights)

        tipx_acc, best_alpha_tipx, best_beta_tipx, best_gamma_tipx = self.hparam_search(val_features, val_labels, test_features, test_labels, support_features, support_labels, text_classifier_weights, val_kl_divs_sims, test_kl_divs_sims)
        
        self.report_test(domain_name, sample_num, tipx_acc)
            
        del test_loader
        return sample_num, eval_dict
    
    def get_text_classifier_weights(self):
        text_features = self.model.encode_text(self.text_class_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0) # take mean over all text embeddings for all prompts
        class_embedding /= class_embedding.norm() # L2 normalise mean embedding
        return class_embedding
    
    def get_features(self, loader):
        feature_list = []
        label_list = []
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                imgs = data["image"]
                labels = data["label"]
                imgs = imgs.to(self.device)
                
                image_features = self.model.encode_image(imgs)
                text_features = self.model.encode_text(self.text_class_tokens)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                feature_list.append(image_features.cpu())
                label_list.append(labels)
            
            feature_list = torch.cat(feature_list)
            label_list = torch.cat(label_list)
        
        return feature_list, label_list
    
    def compute_image_text_distributions(temp, train_images_features_agg, test_features, val_features, vanilla_zeroshot_weights):
        train_image_class_distribution = train_images_features_agg.T @ vanilla_zeroshot_weights
        train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution/temp)

        test_image_class_distribution = test_features @ vanilla_zeroshot_weights
        test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution/temp)

        val_image_class_distribution = val_features @ vanilla_zeroshot_weights
        val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution/temp)

        return train_image_class_distribution, test_image_class_distribution, val_image_class_distribution

    def get_kl_divergence_sims(self, train_image_class_distribution, test_image_class_distribution):
        bs = 100
        kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

        for i in tqdm(range(test_image_class_distribution.shape[0]//bs)):
            curr_batch = test_image_class_distribution[i*bs : (i+1)*bs]
            repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)    
            q = train_image_class_distribution
            q_repeated = torch.cat([q]*bs)
            kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
            kl = kl.sum(dim=-1)
            kl = kl.view(bs, -1)
            kl_divs_sim[ i*bs : (i+1)*bs , : ] = kl  

        return kl_divs_sim

    def get_kl_div_sims(self, args, test_features, val_features, train_features, clip_weights):

        train_image_class_distribution, test_image_class_distribution, val_image_class_distribution = compute_image_text_distributions(args.temperature, train_features, test_features, val_features, clip_weights)

        train_kl_divs_sim = self.get_kl_divergence_sims(train_image_class_distribution, train_image_class_distribution)
        test_kl_divs_sim = self.get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)
        val_kl_divs_sim = self.get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)

        return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

    def scale_(self, x, target):
        
        y = (x - x.min()) / (x.max() - x.min())
        y *= target.max() - target.min()
        y += target.min()
        
        return y

    def hparam_search(self, val_features, val_labels, test_features, test_labels, train_images_features_agg, train_images_targets, zeroshot_weights, val_kl_divs_sim, test_kl_divs_sim):

        search_scale = [50, 50, 30]
        search_step = [200, 20, 50]

        alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
        beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
        gamma_list = [i * (search_scale[2] - 0.1) / search_step[2] + 0.1 for i in range(search_step[2])]

        best_tipx_acc = 0 

        best_gamma_tipx, best_alpha_tipx, best_beta_tipx = 0, 0, 0

        for alpha in alpha_list:
            for beta in beta_list:
                n = 0.
                batch_idx = 0
    
                new_knowledge = val_features @ train_images_features_agg
                cache_logits = ((-1) * (beta - beta * new_knowledge)).exp() @ (train_images_targets)
                clip_logits = 100. * val_features @ zeroshot_weights

                batch_idx += 1
                n += val_features.size(0)

                neg_affs = self.scale_((val_kl_divs_sim).cuda(), new_knowledge)
                affinities = -neg_affs
                kl_logits = affinities.half() @ train_images_targets

                for gamma in gamma_list:  
                    tipx_top1, tipx_top5 = 0., 0.

                    tipx_logits = clip_logits + kl_logits * gamma + cache_logits * alpha
                    tipx_acc1, tipx_acc5 = self.accuracy(tipx_logits, val_labels, topk=(1, 5))
                    tipx_top1 += tipx_acc1
                    tipx_top5 += tipx_acc5
                    tipx_top1 = (tipx_top1 / n) * 100
                    tipx_top5 = (tipx_top5 / n) * 100

                    if tipx_top1 > best_tipx_acc:
                        best_tipx_acc = tipx_top1
                        best_alpha_tipx = alpha
                        best_gamma_tipx = gamma
                        best_beta_tipx = beta

        n = test_features.size(0)

        clip_logits = 100. * test_features @ zeroshot_weights

        neg_affs = self.scale_((test_kl_divs_sim).cuda(), new_knowledge)
        affinities = -neg_affs
        kl_logits = affinities.half() @ train_images_targets

        tipx_top1, tipx_top5 = 0., 0.

        new_knowledge = test_features @ train_images_features_agg
        cache_logits = ((-1) * (best_beta_tipx - best_beta_tipx * new_knowledge)).exp() @ train_images_targets    
        tipx_logits = clip_logits + kl_logits * best_gamma_tipx + cache_logits * best_alpha_tipx
        tipx_acc1, tipx_acc5 = self.accuracy(tipx_logits, test_labels, topk=(1, 5))
        tipx_top1 += tipx_acc1
        tipx_top5 += tipx_acc5
        tipx_top1 = (tipx_top1 / n) * 100
        tipx_top5 = (tipx_top5 / n) * 100

        return tipx_top1, best_alpha_tipx, best_beta_tipx, best_gamma_tipx


        
        

            


