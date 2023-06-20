import torch
from torch.nn import functional as F
import logging
import numpy as np

from .model_base import ModelBase
from core.net_bank.joint_estimator import Joint_estimator
from core.net_bank.parts_classifier import Parts_classifier

from .utils.chamfer_loss import classification_chamfer, align_chamfer
from .utils.joint_loss import joint_decoder_loss, supervised_axis_loss, theta_range_loss, segmentation_suppress_loss
from .utils.align import rotation_vec_2_matrix, binary_split, multi_frame_align
from .utils.viz_cdc_render import viz_cdc
from .utils.evaluation import eval_atc_all, eval_segmentation_acc

class Model(ModelBase):
    def __init__(self, cfg):
        network = CaDeX_PM(cfg)
        super().__init__(cfg, network)

        self.input_num = cfg["dataset"]["input_num"]
        self.num_atc = cfg["dataset"]["num_atc"]
        self.input_type = cfg["dataset"]["input_type"]

        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_theta_range",
                "loss_supervised_axis",
                "loss_joint",
                "loss_classifier",
                "loss_align",
                "loss_axis",
                "loss_confidence",
                "loss_suppress",
                "segmentation",
                "results_observed"
            ]
            + ["loss_reg_shift_len"],
            "image": ["mesh_viz_image", "query_viz_image"],
            "video": ["flow_video"],
            "hist": ["loss_recon_i", "loss_corr_i", "cdc_shift"],
            "xls": ["running_metric_report", "results_observed"],
        }

    def _postprocess_after_optim(self, batch):
        # eval iou
        if batch["phase"].startswith("val"):
            report = {}
            report["segmentation"], _, _ = eval_segmentation_acc(
                batch['seq_pc_mask'][:, 0].detach().cpu().numpy(),
                batch['model_input']['label'][:,:,0,:].to(torch.int64).detach().cpu().numpy()
                )
            batch["running_metric_report"] = report
            batch['segmentation'] = report["segmentation"]

        if "c_trans" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            viz_flag = batch["model_input"]["viz_flag"]
            TEST_RESULT_OBS = {}
            B, P, T, _, _ = batch["c_rotation"].shape
            with torch.no_grad():
                # prepare viz mesh lists
                for t in range(T):
                    batch["mesh_t%d" % t] = []
                batch["cdc_mesh"] = []
                rendered_fig_list, video_list = [], []
                for bid in range(B):
                    if phase.startswith("test"):
                        # evaluate the generated mesh list
                        # logging.warning("Start eval")
                        eval_dict_mean_gt_observed = eval_atc_all(
                            joint_o_pred=batch['c_joint'][bid, :-1, :3].detach().cpu().numpy(),
                            joint_t_pred=batch['c_joint'][bid, :-1, 3:].detach().cpu().numpy(),
                            segmentation_pred=batch['seq_pc_mask'][bid].detach().cpu().numpy(),
                            joint_t_gt=batch['model_input']['axis_t'][bid].detach().cpu().numpy(),
                            joint_o_gt=batch['model_input']['axis_o'][bid].detach().cpu().numpy(),
                            segmentation_gt=batch['model_input']['label'][bid].permute(1,0,2).to(torch.int64).detach().cpu().numpy(),
                            pc = batch['model_input']['inputs'][bid].detach().cpu() if T != 1 else None,
                            rotation = batch['c_rotation'][bid].detach().cpu() if T != 1 else None,
                        )
                        # logging.warning("End eval")
                        # record the batch results
                        for k, v in eval_dict_mean_gt_observed.items():
                            _k = f"{k}(O)"
                            if _k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[_k] = [v.item()]
                            else:
                                TEST_RESULT_OBS[_k].append(v.item())

                    # render an image of the mesh
                    if viz_flag:
                        logging.warning("Start visualization")
                        scale_cdc = True
                        if "viz_cdc_scale" in self.cfg["logging"].keys():
                            scale_cdc = self.cfg["logging"]["viz_cdc_scale"]
                        fig_t_list, fig_query_list = viz_cdc(
                            input_pc=batch["seq_pc"][bid].detach().cpu().numpy(),
                            input_pc_mask=batch["seq_pc_mask"][bid].detach().cpu().numpy(),
                            input_pc_joint=batch["c_joint"][bid, :-1].detach().cpu().numpy(),
                            object_T=None,
                            scale_cdc=scale_cdc,
                            interval=self.cfg["logging"]["mesh_viz_interval"],
                            align_cdc=False,
                            cam_dst_default=1.7,
                        )
                        cat_fig = np.concatenate(fig_t_list, axis=0).transpose(2, 0, 1)
                        cat_fig = np.expand_dims(cat_fig, axis=0).astype(np.float) / 255.0
                        rendered_fig_list.append(cat_fig)

                        video = np.concatenate(
                            [i.transpose(2, 0, 1)[np.newaxis, ...] for i in fig_t_list], axis=0
                        )  # T,3,H,W
                        video = np.expand_dims(video, axis=0).astype(np.float) / 255.0
                        video_list.append(video)
                        logging.warning("End visualization")

                if viz_flag:
                    batch["mesh_viz_image"] = torch.Tensor(
                        np.concatenate(rendered_fig_list, axis=0)
                    )  # B,3,H,W
                    batch["flow_video"] = torch.Tensor(
                        np.concatenate(video_list, axis=0)
                    )  # B,T,3,H,W
            if phase.startswith("test"):
                batch["results_observed"] = TEST_RESULT_OBS
        del batch["model_input"]
        return batch


class CaDeX_PM(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_atc = cfg["dataset"]["num_atc"]
        self.input_num = cfg["dataset"]["input_num"]
        self.input_type = cfg["dataset"]["input_type"]
        self.use_axis_loss = cfg["training"]["loss_axis"]
        self.refinement_threhold = cfg["training"]["refinement_point_n"]
        self.dcd_alpha = cfg["training"]["dcd_alpha"]
        self.dcd_threshold = cfg["training"]["dcd_threshold"]
        self.ablation = cfg['training']['ablation']

        self.network_dict = torch.nn.ModuleDict(
            {
                "joint_estimator": Joint_estimator(
                    #**cfg["model"]["joint_estimator"],
                    atc_num=self.num_atc,
                ),
                "parts_classifier": Parts_classifier(
                    num_t=self.input_num,
                    num_p=self.num_atc + 1
                ),
            }
        )

        for k in self.network_dict:
            logging.info(
                "{} params in {}".format(
                    sum(param.numel() for param in self.network_dict[k].parameters()),
                    k
                )
            )

    def forward(self, input_pack, viz_flag):

        output = {}
        output["phase"] = input_pack["phase"]
        phase = input_pack["phase"]
        set_pc = input_pack["inputs"][:, : self.input_num]
        axis_o = input_pack["axis_o"]
        B, _, _, _ = input_pack["inputs"].shape
        P = self.num_atc+1
        T_in = self.input_num
        N = set_pc.shape[2]
        refine_T = 1

        for refine_t in range(refine_T):
            c_joint, theta_hat, c_confidence, c_joint_frame = self.network_dict["joint_estimator"](
                set_pc, input_pack["category"][0]
                )
            axis = c_joint[:, :, :3]  # B,P,3
            c_axis = axis / torch.norm(axis, dim=-1).unsqueeze(-1)
            c_trans = c_joint[:, :, 3:]
            c_trans = torch.cat(
                [c_trans[:, :-1, :], c_trans[:, :1, :]], dim=1
                )  # B,P,3
            c_length = torch.cat(
                [theta_hat, torch.zeros_like(theta_hat[:, :, :1])], dim=-1
                ).permute(0, 2, 1)  # B,P,T
            c_rotation = rotation_vec_2_matrix(c_axis, c_length)

            input_multi_w, _ = self.network_dict["parts_classifier"](
                set_pc.reshape(B*T_in, N, -1), c_joint_frame, theta_hat
                )  # BT,P,N
            input_multi_label = input_multi_w.reshape(B, T_in, P, N).permute(0, 2, 1, 3)  # B,P,T,N

        input_multi_label_hard = F.one_hot(
            torch.argmax(input_multi_label, dim=1), num_classes=P
            ).permute(0, 3, 1, 2)  # B,P,T,N
        static_label_mask = input_multi_label_hard[:, -1, :, :]  # B,T,N
        input_multi = multi_frame_align(
            set_pc, c_rotation[:, :, :T_in, :, :], c_trans, cat_frame_label=True)  # B,P,T,N,(3+T)
        input_multi_query, input_multi_others = binary_split(input_multi)  # BPT,N,3, BPT,(t-1)N,3

        # visualize
        if viz_flag:
            output["c_joint"] = c_joint.detach()
            output["c_trans"] = c_trans.detach()
            output["c_rotation"] = c_rotation.detach()
            output["seq_pc"] = input_pack["inputs"]
            output["seq_pc_mask"] = input_multi_label_hard.permute(0, 2, 3, 1).detach()

        # if test, direct return
        if phase.startswith("test"):
            c_joint[:, :, :3] = c_joint[:, :, :3] / torch.norm(c_joint[:, :, :3], dim=-1).unsqueeze(-1)
            output["c_joint"] = c_joint.detach()
            output["c_trans"] = c_trans.detach()
            output["c_rotation"] = c_rotation.detach()
            output["seq_pc_mask"] = input_multi_label_hard.permute(0, 2, 3, 1).detach()
            output["theta_hat"] = theta_hat.detach()
            output["viz_query"] = input_multi_query
            output["viz_weights"] = input_multi_label
            output["viz_trans"] = c_trans
            output["viz_axis"] = c_axis
            output["viz_length"] = c_length
            return output

        # Flag of using segmentation result to refine joint parameter
        active_point_cnt = static_label_mask.shape[-1] - torch.count_nonzero(static_label_mask.reshape(B*T_in,-1), dim=-1)
        double_align = torch.all(
            active_point_cnt > (P-1) * self.refinement_threhold
            )
        # Align loss by using the average values of each frame joint estimation
        '''
        align_loss_mean, _, _ = self.align_loss(
            input_multi_query, input_multi_others, B, P, T_in, c_trans, c_axis,
            threshold=self.dcd_threshold, dcd_alpha=self.dcd_alpha)
        if double_align:
            align_loss_2_mean, _, _ = self.align_loss(
                input_multi_query, input_multi_others,
                B, P, T_in, c_trans, c_axis,
                threshold=self.dcd_threshold, mask=static_label_mask, dcd_alpha=self.dcd_alpha)
            align_loss_mean += align_loss_2_mean
            align_loss_mean /= 2
        '''

        # Align loss by using joint estimation from each frame seperately
        align_loss_list = []
        for t in range(T_in):
            c_joint_t = c_joint_frame[:, t, :, :]
            axis_t = c_joint_t[:, :, :3]  # B,P,3
            c_axis_t = axis_t / torch.norm(axis_t, dim=-1).unsqueeze(-1)
            c_trans_t = c_joint_t[:, :, 3:]
            c_trans_t = torch.cat(
                [c_trans_t[:, :-1, :], c_trans_t[:, :1, :]], dim=1
                )  # B,P,3
            c_length_t = torch.cat(
                [theta_hat, torch.zeros_like(theta_hat[:, :, :1])], dim=-1
                ).permute(0, 2, 1)  # B,P,T
            c_rotation_t = rotation_vec_2_matrix(c_axis_t, c_length_t)
            input_multi_t = multi_frame_align(
                set_pc, c_rotation_t[:, :, :T_in, :, :], c_trans_t, cat_frame_label=True)  # B,P,T,N,(3+T)
            input_multi_query_t, input_multi_others_t = binary_split(input_multi_t)  # BPT,N,3, BPT,(t-1)N,3
            align_loss_t, _, _ = self.align_loss(
                input_multi_query_t, input_multi_others_t,
                B, P, T_in, c_trans_t, c_axis_t,
                threshold=self.dcd_threshold, dcd_alpha=self.dcd_alpha)
            if double_align:
                align_loss_2_t, _, _ = self.align_loss(
                    input_multi_query_t, input_multi_others_t,
                    B, P, T_in, c_trans_t, c_axis_t,
                    threshold=self.dcd_threshold, mask=static_label_mask, dcd_alpha=self.dcd_alpha)
                align_loss_t += align_loss_2_t
                align_loss_t /= 2
            align_loss_list.append(align_loss_t)

        align_loss = torch.sum(torch.stack(align_loss_list))
        align_loss /= T_in
        
        joint_loss = joint_decoder_loss(
            c_joint_frame[:,:,:,:3], c_length, c_joint_frame[:,:,:,3:], set_pc,
            input_multi_label_hard,
            refine_threshold = self.refinement_threhold,
            ablation = False,
            )
        
        suppress_loss = segmentation_suppress_loss(
            input_multi_label, P
            )


        range_loss = theta_range_loss(
            c_length[:, :-1, :T_in], input_pack["theta_range"]
            )
        
        supervised_loss = supervised_axis_loss(
            c_axis, axis_o,
            )
        
        classifier_loss = self.classification_loss(
            input_multi_query, input_multi_others,
            input_multi_label, dcd_alpha=self.dcd_alpha
            )

        total_loss = joint_loss + 10*classifier_loss + 10*align_loss + range_loss + suppress_loss
        if self.use_axis_loss:
            total_loss += supervised_loss

        output["batch_loss"] = total_loss
        output["loss_supervised_axis"] = supervised_loss.detach()
        output["loss_joint"] = joint_loss.detach()
        output["loss_theta_range"] = range_loss.detach()
        output["loss_suppress"] = suppress_loss.detach()
        output["loss_classifier"] = 10*classifier_loss.detach()
        output["loss_align"] = 10*align_loss.detach()
        output["viz_query"] = input_multi_query
        output["viz_weights"] = input_multi_label
        output["viz_trans"] = c_trans
        output["viz_axis"] = c_axis
        output["viz_length"] = c_length
        output["seq_pc_mask"] = input_multi_label_hard.permute(0, 2, 3, 1).detach()
        return output

    def classification_loss(
        self,
        query, others,
        label=None, dcd_alpha=50
    ):
        if label is not None:
            label = label.reshape(-1, label.shape[-1])  # BPT,N
        loss_sum = classification_chamfer(
            query, others,
            weights=label,
            pcl=self.input_type == 'pcl',
            dcd_alpha=dcd_alpha
            )
        return loss_sum

    def align_loss(
        self,
        query, others,
        B, P, T,
        c_trans, c_axis,
        threshold=1, mask=None, dcd_alpha=50
    ):
        loss_sum, static_mask, active_mask = align_chamfer(
            query, others, B, P, T, c_trans, c_axis,
            threshold_weight=threshold, weights=mask, dcd_alpha=dcd_alpha)
        return loss_sum, static_mask, active_mask
