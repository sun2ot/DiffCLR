import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import numpy as np
import os
from scipy.sparse import coo_matrix
from typing import Literal, Optional
from safetensors.torch import save_file

from data import DataHandler
from utils import *
from utils.log import Log
from utils.conf import Config
from utils.loss import l2_reg_loss, InfoNCE, bpr_loss
from utils.adj import torch_sparse_adj, build_knn_adj
from models import DiffCLR, DiffusionModel, DenoiseModel

class Trainer:
    def __init__(self, handler: DataHandler, config: Config, logger: Log):
        self.handler = handler
        self.config = config
        self.logger = logger
        self.device = torch.device(f"cuda:{self.config.base.gpu}" if torch.cuda.is_available() else "cpu")
        self.best_user_emb: Optional[Tensor] = None
        self.best_item_emb: Optional[Tensor] = None
        self.adjs: list[Optional[Tensor]] = [None] * 3
        self.init_model()

    def init_model(self):
        """Init DiffMM, Diffusion, Denoise Models"""
        self.model = DiffCLR(self.config, self.handler).cuda(self.device)

        self.opt = Adam(self.model.parameters(), lr=self.config.train.lr, weight_decay=0)
        self.model_scheduler = CosineAnnealingLR(self.opt, T_max=self.config.train.epoch, eta_min=1e-4)

        self.diffusion_model = DiffusionModel(self.config).cuda(self.device)
        
        out_dims = [self.config.base.hidden_dim ,self.config.data.item_num] # [denoise_dim, item_num]
        in_dims = out_dims[::-1]  # [item_num, denoise_dim]
        self.image_denoise_model = DenoiseModel(in_dims, out_dims, self.config).cuda(self.device)
        self.image_denoise_opt = Adam(self.image_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
        self.image_scheduler = CosineAnnealingLR(self.image_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)

        self.text_denoise_model = DenoiseModel(in_dims, out_dims, self.config).cuda(self.device)
        self.text_denoise_opt = Adam(self.text_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
        self.text_scheduler = CosineAnnealingLR(self.text_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)

        if self.config.data.name == 'tiktok':
            self.audio_denoise_model = DenoiseModel(in_dims, out_dims, self.config).cuda(self.device)
            self.audio_denoise_opt = Adam(self.audio_denoise_model.parameters(), lr=self.config.train.lr, weight_decay=0)
            self.audio_scheduler = CosineAnnealingLR(self.audio_denoise_opt, T_max=self.config.train.epoch, eta_min=1e-4)

    def save(self, mode: Literal['embs', 'graph', 'model']):
        save_path = os.path.join('persist', self.config.data.name)
        file_name = f"{self.config.base.timestamp}_{mode}.safetensors"

        if mode == 'embs':
            if self.best_user_emb is not None and self.best_item_emb is not None:
                embs_dict = {'user': self.best_user_emb, 'item': self.best_item_emb}
                save_file(embs_dict, os.path.join(save_path, file_name))
                self.logger.info(f"Save model to {save_path} as {mode} format")
            else:
                print("⚠️ No embeddings to save.")

        elif mode == 'model':
            # Save the model state dict
            torch.save(self.model.state_dict(), os.path.join(save_path, file_name))
            self.logger.info(f"Save model to {save_path} as {mode} format")

        else:
            raise NotImplementedError(f"Unsupported save mode: {mode}")

    def run(self):
        self.logger.info('Model Initialized ✅')

        max_recall, max_ndcg, max_precision = 0, 0, 0
        his_max = [0, 0, 0]
        bestEpoch = 0
        total_epoch = self.config.train.epoch

        self.logger.info('Start training 🚀')
        try:
            for epoch in range(0, self.config.train.epoch):
                tstFlag = (epoch % self.config.train.test_epoch == 0)
                result = self.train()

                if self.config.train.use_lr_scheduler:
                    self.model_scheduler.step()
                    # ----------- Ablation3: KNN -----------
                    self.image_scheduler.step()
                    self.text_scheduler.step()
                    if self.config.data.name == 'tiktok':
                        self.audio_scheduler.step()
                    # ----------- Ablation3: KNN -----------
                
                self.logger.info(format_epoch('⏩ Train', epoch, total_epoch, result))
                if tstFlag:
                    result, user_emb, item_emb = self.testEpoch()
                    his_max = update_max([result['Recall'], result['NDCG'], result['Precision']], his_max)
                    if result['Recall'] > max_recall:
                        max_recall = result['Recall']
                        max_ndcg = result['NDCG']
                        max_precision = result['Precision']
                        bestEpoch = epoch
                        self.best_user_emb, self.best_item_emb = user_emb, item_emb
                    self.logger.info(format_epoch('🧪 Test', epoch, total_epoch, result))
                self.logger.info(format_best(bestEpoch, max_recall, his_max[0], max_ndcg, his_max[1], max_precision, his_max[2]))
        except KeyboardInterrupt:
            self.logger.info('🈲 Training interrupted by user!')
            self.logger.info(format_best(bestEpoch, max_recall, his_max[0], max_ndcg, his_max[1], max_precision, his_max[2]))
            if self.config.base.enable_save:
                self.logger.warning("⚠️ Waiting for saving model... Please do not press Ctrl+C continuously.")
        finally:
            if self.config.base.enable_save:
                self.save('embs')

    def train(self):
        self.handler.train_data.neg_sampling()

        train_steps = len(self.handler.train_data) // self.config.train.batch
        diffusion_steps = len(self.handler.diffusion_data) // self.config.train.batch

        image_diff_loss, text_diff_loss, audio_diff_loss = self.diffusion_train()

        self.rebuild_matrix()
        # self.knn_rebuild_matrix()

        ep_loss, ep_rec_loss, ep_reg_loss, ep_cl_loss = self.joint_train()

        result = dict()
        result['Loss'] = ep_loss / train_steps
        result['BPR Loss'] = ep_rec_loss / train_steps
        result['reg loss'] = ep_reg_loss / train_steps
        result['CL loss'] = ep_cl_loss / train_steps
        result['image loss'] = image_diff_loss / diffusion_steps
        result['text loss'] = text_diff_loss / diffusion_steps
        if self.config.data.name == 'tiktok':
            result['audio loss'] = audio_diff_loss / diffusion_steps
        return result

    def testEpoch(self):
        testData = self.handler.test_data
        testLoader = self.handler.test_loader
        epRecall, epNdcg, epPrecision = [0] * 3
        i = 0
        data_length = len(testData)

        if self.config.data.name == 'tiktok':
            gcn_output = self.model.forward(self.handler.torchBiAdj, self.adjs[0], self.adjs[1], self.adjs[2]) # type: ignore
        else:
            gcn_output = self.model.forward(self.handler.torchBiAdj, self.adjs[0], self.adjs[1]) # type: ignore
        user_emb, item_emb = gcn_output.u_final_embs, gcn_output.i_final_embs

        for usr, trainMask in testLoader:
            i += 1
            usr: Tensor = usr.long().cuda(self.device)
            trainMask: Tensor = trainMask.cuda(self.device)
            predict = torch.mm(user_emb[usr], torch.transpose(item_emb, 1, 0)) * (1 - trainMask) - trainMask * 1e8
            topk = self.config.base.topk
            _, top_idxs = torch.topk(predict, topk)  # (batch, topk)
            recall, ndcg, precision = cal_metrics(topk, top_idxs.cpu().numpy(), testData.test_user_its, usr)
            epRecall += recall
            epNdcg += ndcg
            epPrecision += precision
        ret = dict()
        ret['Recall'] = epRecall / data_length
        ret['NDCG'] = epNdcg / data_length
        ret['Precision'] = epPrecision / data_length
        return ret, user_emb, item_emb

    def diffusion_train(self):
        image_diff_loss, text_diff_loss, audio_diff_loss = 0, 0, 0
        self.logger.info('Diffusion model training')
        for i, batch_data in enumerate(self.handler.diffusion_loader):
            # batch: list(tensor), batch[0]: (batch_size, item_num), batch[1]: (batch_size, )
            batch_u_items = batch_data[0]

            i_embs = self.model.i_embs
            image_feats = self.model.get_image_feats().detach()
            text_feats = self.model.get_text_feats().detach()

            batch_image_loss: Tensor = self.diffusion_model.train_loss(self.image_denoise_model, batch_u_items, i_embs, image_feats)
            loss_image = batch_image_loss.mean()
            image_diff_loss += loss_image.item()

            batch_text_loss: Tensor = self.diffusion_model.train_loss(self.text_denoise_model, batch_u_items, i_embs, text_feats)
            loss_text = batch_text_loss.mean()
            text_diff_loss += loss_text.item()

            # optimizer
            self.image_denoise_opt.zero_grad()
            self.text_denoise_opt.zero_grad()

            if self.config.data.name == 'tiktok':
                audio_feats = self.model.get_audio_feats()
                assert audio_feats is not None
                audio_feats = audio_feats.detach()
                self.audio_denoise_opt.zero_grad()
                batch_audio_loss: Tensor = self.diffusion_model.train_loss(self.audio_denoise_model, batch_u_items, i_embs, audio_feats)
                loss_audio = batch_audio_loss.mean()
                audio_diff_loss += loss_audio.item()

                # Normalize the losses before summing
                total_loss = loss_image.item() + loss_text.item() + loss_audio.item()
                batch_diff_loss = (loss_image + loss_text + loss_audio)/total_loss
                image_diff_loss /= total_loss
                text_diff_loss /= total_loss
                audio_diff_loss /= total_loss
            else:
                # Normalize the losses before summing
                total_loss = loss_image.item() + loss_text.item()
                batch_diff_loss = (loss_image + loss_text)/total_loss
                image_diff_loss /= total_loss
                text_diff_loss /= total_loss

            batch_diff_loss.backward()

            self.image_denoise_opt.step()
            self.text_denoise_opt.step()
            if self.config.data.name == 'tiktok':
                self.audio_denoise_opt.step()
        return image_diff_loss, text_diff_loss, audio_diff_loss

    def rebuild_matrix(self):
        self.logger.info('Re-build multimodal UI matrix')
        with torch.no_grad():
            # every modal's u_list/i_list/edge_list for creating adjacency matrix
            modality_names = ['image', 'text']
            if self.config.data.name == 'tiktok':
                modality_names.append('audio')
            u_list_dict = {m: [] for m in modality_names}
            i_list_dict = {m: [] for m in modality_names}
            edge_list_dict = {m: [] for m in modality_names}
            denoise_model_dict = {
                'image': self.image_denoise_model,
                'text': self.text_denoise_model,
            }
            if self.config.data.name == 'tiktok':
                denoise_model_dict['audio'] = self.audio_denoise_model

            for batch_data in self.handler.diffusion_loader:
                batch_u_items: Tensor = batch_data[0]
                batch_u_idxs: np.ndarray = batch_data[1].cpu().numpy()

                user_degrees = self.handler.getUserDegrees()
                topk_values = user_degrees[batch_u_idxs]

                for m in modality_names:
                    denoised_batch = self.diffusion_model.generate_view(
                        denoise_model_dict[m],
                        batch_u_items,
                        self.config.hyper.sampling_step
                    )
                    for i in range(batch_u_idxs.shape[0]):
                        user_topk = topk_values[i]
                        _, indices = torch.topk(denoised_batch[i], k=user_topk)  # (batch_size, topk)
                        for j in range(indices.shape[0]):
                            u_list_dict[m].append(batch_u_idxs[i])
                            i_list_dict[m].append(int(indices[j]))
                            edge_list_dict[m].append(1.0)

            # make torch sparse adjacency matrix: (user_num, topk)
            shape = (self.config.data.user_num, self.config.data.item_num)
            for i, m in enumerate(modality_names):
                mat = coo_matrix(
                    (np.array(edge_list_dict[m]), (np.array(u_list_dict[m]), np.array(i_list_dict[m]))),
                    shape=shape, dtype=np.float32
                )
                self.adjs[i] = torch_sparse_adj(mat, self.config.data.user_num, self.config.data.item_num, self.device) # type: ignore

    def knn_rebuild_matrix(self):
        """Ablation3: Use this to replace `rebuild_matrix()`"""
        self.logger.info('Rebuild multimodal UI matrix (KNN)')
        with torch.no_grad():
            # image
            u_i, i_i, v_i = build_knn_adj(
                self.handler.train_data.user_pos_items,
                self.handler.image_feats.detach().cpu().numpy(),
                self.config.hyper.knn_topk
            )
            mat0 = coo_matrix((v_i, (u_i, i_i)), shape=(self.config.data.user_num, self.config.data.item_num), dtype=np.float32)
            self.adjs[0] = torch_sparse_adj(mat0, self.config.data.user_num, self.config.data.item_num, self.device)

            # text
            u_t, i_t, v_t = build_knn_adj(
                self.handler.train_data.user_pos_items,
                self.handler.text_feats.detach().cpu().numpy(),
                self.config.hyper.knn_topk
            )
            mat1 = coo_matrix((v_t, (u_t, i_t)), shape=(self.config.data.user_num, self.config.data.item_num), dtype=np.float32)
            self.adjs[1] = torch_sparse_adj(mat1, self.config.data.user_num, self.config.data.item_num, self.device)
            
            # audio
            if self.config.data.name == 'tiktok':
                u_a, i_a, v_a = build_knn_adj(
                    self.handler.train_data.user_pos_items,
                    self.handler.audio_feats.detach().cpu().numpy(),
                    self.config.hyper.knn_topk
                )
                mat2 = coo_matrix((v_a, (u_a, i_a)), shape=(self.config.data.user_num, self.config.data.item_num), dtype=np.float32)
                self.adjs[2] = torch_sparse_adj(mat2, self.config.data.user_num, self.config.data.item_num, self.device)

    def joint_train(self):
        self.logger.info('Joint training 🤝')
        ep_loss, ep_rec_loss, ep_reg_loss, ep_cl_loss = 0, 0, 0, 0
        for batch_data in self.handler.train_loader:
            users, pos_items, neg_items = batch_data
            users: Tensor = users.long().cuda(self.device)
            pos_items: Tensor = pos_items.long().cuda(self.device)
            neg_items: Tensor = neg_items.long().cuda(self.device)

            if self.config.data.name == 'tiktok':
                model_output = self.model.forward(self.handler.torchBiAdj, self.adjs[0], self.adjs[1], self.adjs[2]) # type: ignore
                final_user_embs, final_item_embs = model_output.u_final_embs, model_output.i_final_embs
            else:
                model_output = self.model.forward(self.handler.torchBiAdj, self.adjs[0], self.adjs[1]) # type: ignore
                final_user_embs, final_item_embs = model_output.u_final_embs, model_output.i_final_embs
            
            u_embs = final_user_embs[users]
            pos_embs = final_item_embs[pos_items]
            neg_embs = final_item_embs[neg_items]

            rec_loss = bpr_loss(u_embs, pos_embs, neg_embs)
            reg_loss = l2_reg_loss(self.config.train.reg, [self.model.u_embs, self.model.i_embs], self.device)
            ep_rec_loss += rec_loss.item()
            ep_reg_loss += reg_loss.item()

            #* Cross layer CL
            ego_embs = torch.cat([self.model.u_embs, self.model.i_embs], dim=0)
            all_embs = []
            all_embs_cl = ego_embs
            for k in range(3): # GCN Layers = 3
                ego_embs = torch.sparse.mm(self.handler.torchBiAdj, ego_embs)
                random_noise = torch.rand_like(ego_embs)
                ego_embs += torch.sign(ego_embs) * F.normalize(random_noise) * self.config.hyper.noise_degree
                all_embs.append(ego_embs)
                if k == 0: # which layer to CL
                    all_embs_cl = ego_embs
            final_embs = torch.mean(torch.stack(all_embs), dim=0)
            
            cl1_user_embs = final_embs[:self.config.data.user_num]
            cl1_item_embs = final_embs[self.config.data.user_num:]
            cl2_user_embs = all_embs_cl[:self.config.data.user_num]
            cl2_item_embs = all_embs_cl[self.config.data.user_num:]

            #* Cross CL Loss
            cross_cl_loss = (InfoNCE(cl1_user_embs, cl2_user_embs, users, self.config.hyper.cross_cl_temp) + InfoNCE(cl1_item_embs, cl2_item_embs, pos_items, self.config.hyper.cross_cl_temp)) * self.config.hyper.cross_cl_rate
            cl_loss = cross_cl_loss

            # Ablation1
            # cl_loss = 0

            # ----------- Ablation2 -----------
            if self.config.data.name == 'tiktok':
                u_image_embs, i_image_embs = model_output.u_image_embs, model_output.i_image_embs
                u_text_embs, i_text_embs = model_output.u_text_embs, model_output.i_text_embs
                u_audio_embs, i_audio_embs = model_output.u_audio_embs, model_output.i_audio_embs
                assert u_audio_embs is not None and i_audio_embs is not None
                if self.config.base.cl_method == 1:
                    # pairwise CL: image-text, image-audio, text-audio
                    cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cross_modal_cl_loss += (InfoNCE(u_image_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cross_modal_cl_loss += (InfoNCE(u_text_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_text_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cl_loss += cross_modal_cl_loss
                else:
                    # only one CL: image-text
                    main_cl_loss = (InfoNCE(final_user_embs, u_image_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_image_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    main_cl_loss += (InfoNCE(final_user_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    main_cl_loss += (InfoNCE(final_user_embs, u_audio_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_audio_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cl_loss += main_cl_loss
            else:
                u_image_embs, i_image_embs = model_output.u_image_embs, model_output.i_image_embs
                u_text_embs, i_text_embs = model_output.u_text_embs, model_output.i_text_embs
                if self.config.base.cl_method == 1:
                    cross_modal_cl_loss = (InfoNCE(u_image_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(i_image_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cl_loss += cross_modal_cl_loss
                else:
                    #* Main view as the anchor
                    main_cl_loss = (InfoNCE(final_user_embs, u_image_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_image_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    main_cl_loss += (InfoNCE(final_user_embs, u_text_embs, users, self.config.hyper.modal_cl_temp) + InfoNCE(final_item_embs, i_text_embs, pos_items, self.config.hyper.modal_cl_temp)) * self.config.hyper.modal_cl_rate
                    cl_loss += main_cl_loss
            # ----------- Ablation2 -----------

            ep_cl_loss += cl_loss.item()

            batch_joint_loss =  rec_loss + reg_loss + cl_loss
            ep_loss += batch_joint_loss.item()
            
            self.opt.zero_grad()
            batch_joint_loss.backward()
            self.opt.step()
        return ep_loss, ep_rec_loss, ep_reg_loss, ep_cl_loss