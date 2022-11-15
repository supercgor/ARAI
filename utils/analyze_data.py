import os
import torch


class Analyzer(torch.nn.Module):
    def __init__(self, cfg):
        super(Analyzer, self).__init__()
        # mark the config
        self.cfg = cfg

        # some initialization
        self.elem = cfg.DATA.ELE_NAME
        self.elem_num = len(self.elem)
        self.ele_diameter = [0.740, 0.528]

        self.space_size = [cfg.MODEL.CHANNELS, cfg.MODEL.CHANNELS, cfg.DATA.Z]
        self.output_shape = self.space_size + [self.elem_num, 4]
        self.split = cfg.OTHER.SPLIT

        self.register_buffer('real_size', torch.tensor(cfg.DATA.REAL_SIZE))

        self.register_buffer(
            'lattice_expand', self.real_size/torch.tensor(self.space_size))

        self.do_nms = cfg.OTHER.NMS

        self.P_threshold = cfg.OTHER.THRESHOLD
        self.T_threshold = 0.5

        # Construct the pit-tensor
        # Used to Caculate the absulute position of offset, this tensor fulfilled that t[x,y,z] = [x,y,z], pit refers to position index tensor
        self.register_buffer('pit', torch.ones(
            self.space_size).nonzero().view(self.space_size + [3]))

        # In short: ele * (T,P,TP,FP,FN,Acc,Suc) #

    # def forward(self, predictions, targets):
    #     device = self.pit.device
    #     """_summary_

    #     Args:
    #         prediction (_type_): _description_
    #         target (_type_): _description_

    #         return (tuple): TP,FP,FN
    #     """
    #     # Store the previous information, the list contains
    #     total_P = []
    #     total_T = []
    #     total_TP_index = []
    #     total_P_pos = []
    #     total_T_pos = []

    #     # ------------------------------------
    #     # pre-process
    #     # ------------------------------------
    #     # Reshape to batch,X,Y,Z,4,ele
    #     predictions = predictions.view([-1] + self.output_shape)
    #     targets = targets.view([-1] + self.output_shape)
    #     batch_size = predictions.size(0)
    #     # Change to ele,X,Y,Z,4
    #     predictions = predictions.permute(0, 4, 1, 2, 3, 5)
    #     targets = targets.permute(0, 4, 1, 2, 3, 5)

    #     # ------------------------------------
    #     # Statistics
    #     # Find T and P (sometimes P will contain repeat elemenets for two or more near box) #
    #     # mask out for those bigger than the thrushold and turn them into [...[x,y,z]]
    #     # ------------------------------------

    #     for batch in range(batch_size):
    #         P = []
    #         T = []
    #         TP_index = []
    #         P_pos = []
    #         T_pos = []
    #         for ele in range(self.elem_num):
    #             prediction = predictions[batch, ele].detach()
    #             target = targets[batch, ele].detach()
    #             # take all fulfilled elements
    #             mask_t = target[..., 3] > self.T_threshold
    #             ele_T = (target[..., :3] + self.pit)[mask_t] * \
    #                 self.lattice_expand

    #             mask_p = prediction[..., 3] > self.P_threshold
    #             mask_nms = self.nms(prediction, self.ele_diameter[ele])

    #             ele_P = (prediction[..., :3] +
    #                      self.pit)[mask_p] * self.lattice_expand
    #             ele_P = ele_P[mask_nms]

    #             # Matching the nearest
    #             TP_dist = torch.cdist(ele_T, ele_P, p=2)
    #             # Figure out the indices of TP in T and P tensor
    #             TP_T_index = (TP_dist < self.ele_diameter[ele]).sum(
    #                 1).nonzero()
    #             TP_P_index = torch.tensor([], device=device)
    #             if TP_T_index.shape[0] >= 1:
    #                 TP_T_index = TP_T_index.squeeze(1)
    #                 TP_P_index = TP_dist[TP_T_index].min(1).indices

    #             # ---------------------------
    #             # Mark raw elements of the prediction and targets
    #             # P and T can do gradient descend; TP_index and P_pos,T_pos are detached
    #             # ---------------------------

    #             P.append(predictions[batch, ele][mask_p][mask_nms])
    #             T.append(targets[batch, ele][mask_t])
    #             TP_index.append([TP_T_index, TP_P_index])
    #             P_pos.append(ele_P)
    #             T_pos.append(ele_T)
    #         total_P.append(P)
    #         total_T.append(T)
    #         total_TP_index.append(TP_index)
    #         total_P_pos.append(P_pos)
    #         total_T_pos.append(T_pos)

    #     return {"P": total_P, "T": total_T, "TP_index": total_TP_index, "P_pos": total_P_pos, "T_pos": total_T_pos}

    def forward(self, predictions, targets):
        device = self.pit.device
        """_summary_

        Args:
            prediction (_type_): _description_
            target (_type_): _description_

            return (tuple): TP,FP,FN
        """
        # Store the previous information, the list contains
        total_P_nms = []
        total_T = []
        total_TP_index_nms = []
        total_P_pos_nms = []
        total_T_pos = []
        total_P_pos = []
        total_TP_index = []
        total_P = []

        # ------------------------------------
        # pre-process
        # ------------------------------------
        # Reshape to batch,X,Y,Z,4,ele
        predictions = predictions.view([-1] + self.output_shape)
        targets = targets.view([-1] + self.output_shape)
        batch_size = predictions.size(0)
        # Change to ele,X,Y,Z,4
        predictions = predictions.permute(0, 4, 1, 2, 3, 5)
        targets = targets.permute(0, 4, 1, 2, 3, 5)

        # ------------------------------------
        # Statistics
        # Find T and P (sometimes P will contain repeat elemenets for two or more near box) #
        # mask out for those bigger than the thrushold and turn them into [...[x,y,z]]
        # ------------------------------------

        for batch in range(batch_size):
            try:
                P = []
                P_nms = []
                TP_index = []
                TP_index_nms = []
                P_pos = []
                P_pos_nms = []
                T = []
                T_pos = []
                for ele in range(self.elem_num):
                    prediction = predictions[batch, ele].detach()
                    target = targets[batch, ele].detach()
                    # take all fulfilled elements
                    mask_t = target[..., 3] > self.T_threshold
                    ele_T = (target[..., :3] + self.pit)[mask_t] * \
                        self.lattice_expand

                    mask_p = prediction[..., 3] > self.P_threshold
                    mask_nms = self.nms(prediction, self.ele_diameter[ele])

                    ele_P = (prediction[..., :3] +
                             self.pit)[mask_p] * self.lattice_expand
                    ele_P_nms = ele_P[mask_nms]

                    # Matching the nearest
                    TP_dist_nms = torch.cdist(ele_T, ele_P_nms, p=2)
                    # Figure out the indices of TP in T and P tensor
                    TP_T_index_nms = (TP_dist_nms < self.ele_diameter[ele]).sum(
                        1).nonzero()
                    TP_P_index_nms = torch.tensor([], device=device)
                    if TP_T_index_nms.shape[0] >= 1:
                        TP_T_index_nms = TP_T_index_nms.squeeze(1)
                        TP_P_index_nms = TP_dist_nms[TP_T_index_nms].min(
                            1).indices

                    TP_dist = torch.cdist(ele_T, ele_P, p=2)
                    # Figure out the indices of TP in T and P tensor
                    TP_T_index = (TP_dist < self.ele_diameter[ele]).sum(
                        1).nonzero()
                    TP_P_index = torch.tensor([], device=device)
                    if TP_T_index.shape[0] >= 1:
                        TP_T_index = TP_T_index.squeeze(1)
                        TP_P_index = TP_dist[TP_T_index].min(1).indices
                        
                # ---------------------------
                # Mark raw elements of the prediction and targets
                # P and T can do gradient descend; TP_index and P_pos,T_pos are detached
                # ---------------------------

                    P.append(predictions[batch, ele][mask_p])
                    P_nms.append(predictions[batch, ele][mask_p][mask_nms])
                    T.append(targets[batch, ele][mask_t])
                    TP_index.append([TP_T_index, TP_P_index])
                    TP_index_nms.append([TP_T_index_nms, TP_P_index_nms])
                    P_pos.append(ele_P)
                    P_pos_nms.append(ele_P_nms)
                    T_pos.append(ele_T)
            except:
                torch.save(predictions, "../log/error_p.npy")
                torch.save(targets, "../log/error_t.npy")
                raise ("I dont know what is error")
                
            total_P.append(P)
            total_P_nms.append(P_nms)
            total_T.append(T)
            total_TP_index.append(TP_index)
            total_TP_index_nms.append(TP_index_nms)
            total_P_pos.append(P_pos)
            total_P_pos_nms.append(P_pos_nms)
            total_T_pos.append(T_pos)

        return {"P_nms": total_P_nms,
                "P": total_P,
                "T": total_T,
                "TP_index": total_TP_index,
                "TP_index_nms": total_TP_index_nms,
                "P_pos": total_P_pos,
                "P_pos_nms": total_P_pos_nms,
                "T_pos": total_T_pos, }

    def nms(self, position: torch.Tensor, ele_d: float) -> torch.Tensor:
        """_summary_

        Args:
            position (torch.Tensor): 2D tensor with (offset)x, y, z, confitdence
            ele_d (float): elements diameter

        Returns:
            torch.Tensor: 1D mask
        """
        mask = position[..., 3] >= self.P_threshold

        pos_P = (position[..., :3] + self.pit)[mask] * self.lattice_expand
        position = position[mask]
        sort_index = torch.argsort(position[..., 3], descending=True)
        pos_P = pos_P[sort_index]
        PP_dist = torch.cdist(pos_P, pos_P, p=2)

        # which show that whether the points should be restrained.
        # improve great performance ~ 100 times need 0.058s
        PP_dist = torch.triu(PP_dist < ele_d, diagonal=1).float()
        restrain_tensor = PP_dist.sum(0)
        restrain_one = (restrain_tensor != 0).unsqueeze(0).float()
        correct = restrain_one.mm(PP_dist)
        restrain_tensor = restrain_tensor - correct
        selection = restrain_tensor[0] == 0

        mask = torch.sort(sort_index[selection]).values

        return mask

    def count(self, info):

        TP_index, P_pos, T_pos = info["TP_index_nms"], info["P_pos_nms"], info["T_pos"]
        batch_size = len(TP_index)
        dic = {}
        for batch in range(batch_size):
            for i, ele in enumerate(self.elem):

                TP_num = TP_index[batch][i][0].size(0)

                T_z, P_z = T_pos[batch][i][..., 2], P_pos[batch][i][..., 2]

                TP_T_z = T_z[TP_index[batch][i][0]]

                split_past = 0
                for split in self.split[1:]:
                    TP_num = torch.logical_and(
                        TP_T_z > split_past, TP_T_z < split).sum().float()
                    FP_num = torch.logical_and(
                        P_z > split_past, P_z < split).sum().float() - TP_num
                    FN_num = torch.logical_and(
                        T_z > split_past, T_z < split).sum().float() - TP_num
                    acc = TP_num / (TP_num + FP_num + FN_num)
                    suc = (FP_num == 0 and FN_num == 0).float()
                    key = f"{ele}-{split_past:3.1f}-{split:3.1f}"

                    if f"{key}-TP" in dic:
                        dic[f"{key}-TP"] += TP_num
                        dic[f"{key}-FP"] += FP_num
                        dic[f"{key}-FN"] += FN_num
                        dic[f"{key}-ACC"] += acc/batch_size
                        dic[f"{key}-SUC"] += suc/batch_size
                    else:
                        dic[f"{key}-TP"] = TP_num
                        dic[f"{key}-FP"] = FP_num
                        dic[f"{key}-FN"] = FN_num
                        dic[f"{key}-ACC"] = acc/batch_size
                        dic[f"{key}-SUC"] = suc/batch_size
        return dic

    def to_poscar(self, predictions, filenames, out_dir, nms=True, npy=True):

        batch_size = predictions.size(0)

        if npy:
            for batch in range(batch_size):
                filename = filenames[batch]
                file_path = os.path.join(out_dir, self.cfg.TRAIN.CHECKPOINT.split(
                    "/")[-1][:-4], filename + '.npy')
                torch.save(predictions[batch], file_path)

        predictions = predictions.view([-1] + self.output_shape)

        predictions = predictions.permute(0, 4, 1, 2, 3, 5)

        P_pos = []
        for batch in range(batch_size):
            _P_pos = []
            for ele in range(self.elem_num):
                prediction = predictions[batch, ele]
                mask_p = prediction[..., 3] > self.P_threshold
                mask_nms = self.nms(prediction, self.ele_diameter[ele])
                ele_P = (prediction[..., :3] +
                         self.pit)[mask_p] * self.lattice_expand
                ele_P = ele_P[mask_nms]
                ele_P = ele_P / self.real_size
                _P_pos.append(ele_P)
            P_pos.append(_P_pos)

        for batch in range(batch_size):
            for position_list, filename in zip(P_pos, filenames):
                output = ""
                output += f"{' '.join(self.elem)}\n"
                output += f"{1.0:3.1f}" + "\n"
                output += f"\t{self.real_size[0].item():.8f} {0:.8f} {0:.8f}\n"
                output += f"\t{0:.8f} {self.real_size[1].item():.8f} {0:.8f}\n"
                output += f"\t{0:.8f} {0:.8f} {self.real_size[2].item():.8f}\n"
                output += f"\t{' '.join(self.elem)}\n"
                output += f"\t{' '.join([f'{ele_pos.size(0):d}' for ele_pos in position_list])}\n"
                output += f"Selective dynamics\n"
                output += f"Direct\n"
                for i, ele in enumerate(self.elem):
                    P_ele = position_list[i].tolist()
                    for atom in P_ele:
                        output += f" {atom[0]:.8f} {atom[1]:.8f} {atom[2]:.8f} T T T\n"

                try:
                    os.mkdir(os.path.join(
                        out_dir, self.cfg.TRAIN.CHECKPOINT.split("/")[-1][:-4]))
                except:
                    pass
                file_path = os.path.join(out_dir, self.cfg.TRAIN.CHECKPOINT.split(
                    "/")[-1][:-4], filename + '.poscar')
                with open(file_path, 'w') as f:
                    f.write(output)
