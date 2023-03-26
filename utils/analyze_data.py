import os
import torch
from numpy import pi
from .tools import metStat

class Analyzer(torch.nn.Module):
    def __init__(self, cfg):
        super(Analyzer, self).__init__()
        # mark the config
        self.cfg = cfg

        # some initialization
        self.elem = cfg.data.elem_name
        self.elem_num = len(self.elem)
        self.ele_diameter = [0.740 * 1.4, 0.528 * 1.4]

        self.out_size = cfg.model.out_size[::-1]
        self.output_shape = self.out_size + (self.elem_num, 4)
        self.split = cfg.setting.split

        self.register_buffer('real_size', torch.tensor(cfg.data.real_size[::-1]))

        self.register_buffer(
            'lattice_expand', self.real_size/torch.tensor(self.out_size))

        self.do_nms = cfg.model.nms

        self.P_threshold = cfg.model.threshold
        self.T_threshold = 0.5

        # Construct the pit-tensor
        # Used to Caculate the absulute position of offset, this tensor fulfilled that t[x,y,z] = [x,y,z], pit refers to position index tensor
        self.register_buffer('pit', torch.ones(
            self.out_size).nonzero().view(self.out_size + (3,)))
        self.register_buffer('empty_tensor', torch.tensor([]))

    def forward(self, predictions, targets):
        assert predictions.shape == targets.shape, f"prediction shape {predictions.shape} doesn't match {targets.shape}"
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
        predictions = predictions.view((-1,) + self.output_shape)
        targets = targets.view((-1,) + self.output_shape)
        batch_size = predictions.size(0)
        # Change to ele,X,Y,Z,4
        predictions = predictions.permute(0, 4, 1, 2, 3, 5)
        targets = targets.permute(0, 4, 1, 2, 3, 5)

        # ------------------------------------
        pit = self.pit
        lattice_expand = self.lattice_expand
        # ------------------------------------

        for batch in range(batch_size):

            T, P, P_nms, T_pos, P_pos, P_pos_nms, TP_index, TP_index_nms = [],[],[],[],[],[],[],[]

            for ele, diameter in enumerate(self.ele_diameter):
                prediction = predictions[batch, ele]
                target = targets[batch, ele]

                mask_t = target[..., 3] > self.T_threshold
                mask_p = prediction[..., 3] > self.P_threshold

                T_position = (target[..., :3] + pit)[mask_t] * lattice_expand
                P_position = (prediction[..., :3] + pit)[mask_p] * lattice_expand

                prediction_nms, P_nmspos = self.nms(prediction, diameter)

                # Matching the nearest
                TP_dist_nms = torch.cdist(T_position, P_nmspos)
                TP_T_index_nms = (TP_dist_nms < diameter).sum(1).nonzero()

                if TP_T_index_nms.nelement() != 0:
                    TP_T_index_nms = TP_T_index_nms.squeeze(1)
                    TP_P_index_nms = TP_dist_nms[TP_T_index_nms].min(
                        1).indices
                else:
                    TP_P_index_nms = self.empty_tensor

                TP_dist = torch.cdist(T_position, P_position)
                TP_T_index = (TP_dist < diameter).sum(1).nonzero()

                if TP_T_index.nelement() != 0:
                    TP_T_index = TP_T_index.squeeze(1)
                    TP_P_index = TP_dist[TP_T_index].min(1).indices
                else:
                    TP_P_index = self.empty_tensor

                P.append(prediction[mask_p])
                P_nms.append(prediction_nms)
                T.append(target[mask_t])
                TP_index.append([TP_T_index, TP_P_index])
                TP_index_nms.append([TP_T_index_nms, TP_P_index_nms])
                P_pos.append(P_position)
                P_pos_nms.append(P_nmspos)
                T_pos.append(T_position)

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

    def nms(self, prediction, diameter) -> torch.Tensor:
        
        # ------------------------------------
        pit = self.pit
        lattice_expand = self.lattice_expand
        # ------------------------------------
        
        mask_p = prediction[..., 3] > self.P_threshold
        P_position = (prediction[..., :3] + pit)[mask_p] * lattice_expand

        index = torch.argsort(prediction[...,3][mask_p])

        sorted_prediction = P_position[index]

        dist_matrix = torch.cdist(sorted_prediction, sorted_prediction)

        dist_matrix = torch.triu(dist_matrix < diameter, diagonal=1).float()

        restrain_tensor = dist_matrix.sum(0)
        restrain_one = (restrain_tensor != 0).unsqueeze(0).float()
        correct = restrain_one.mm(dist_matrix)
        restrain_tensor = restrain_tensor - correct
        selection = restrain_tensor[0] == 0

        return prediction[mask_p][index[selection]], sorted_prediction[selection] # prediction, position
        # which show that whether the points should be restrained.
        # improve great performance ~ 100 times need 0.058s

    def count(self, info):
        device = self.pit.device
        TP_index, P_pos, T_pos = info["TP_index_nms"], info["P_pos_nms"], info["T_pos"]
        batch_size = len(TP_index)
        dic = {}
        for batch in range(batch_size):
            for i, ele in enumerate(self.elem):

                TP_num = TP_index[batch][i][0].size(0)

                T_z, P_z = T_pos[batch][i][..., 2], P_pos[batch][i][..., 2]

                if TP_index[batch][i][0].nelement() != 0:
                    TP_T_z = T_z[TP_index[batch][i][0]]
                else:
                    TP_T_z = self.empty_tensor
                if TP_index[batch][i][1].nelement() != 0:
                    TP_P_z = P_z[TP_index[batch][i][1]]
                else:
                    TP_P_z = self.empty_tensor

                split_past = 0
                for split in self.split[1:]:
                    TP_num = torch.logical_and(
                        TP_P_z >= split_past, TP_P_z < split).sum().float()
                    FP_num = torch.logical_and(
                        P_z >= split_past, P_z < split).sum().float() - TP_num
                    TP_num = torch.logical_and(
                        TP_T_z >= split_past, TP_T_z < split).sum().float()
                    FN_num = torch.logical_and(
                        T_z >= split_past, T_z < split).sum().float() - TP_num
                    if (TP_num + FP_num + FN_num) == 0:
                        acc = torch.ones(1, device=device).squeeze()
                        suc = acc
                    else:
                        acc = TP_num / (TP_num + FP_num + FN_num)
                        suc = (FP_num == 0 and FN_num == 0).float()
                    key = f"{ele}-{split_past:3.1f}-{split:3.1f}"

                    if f"{key}-TP" in dic:
                        dic[f"{key}-TP"].add(TP_num)
                        dic[f"{key}-FP"].add(FP_num)
                        dic[f"{key}-FN"].add(FN_num)
                        dic[f"{key}-ACC"].add(acc)
                        dic[f"{key}-SUC"].add(suc)
                    else:
                        dic[f"{key}-TP"] = metStat(TP_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-FP"] = metStat(FP_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-FN"] = metStat(FN_num, mode= "sum", dtype=torch.long)
                        dic[f"{key}-ACC"] = metStat(acc)
                        dic[f"{key}-SUC"] = metStat(suc)

                    split_past = split
        return dic

    def to_poscar(self, predictions, filenames, out_dir, nms=True, npy=False):

        batch_size = predictions.size(0)

        if npy:
            for batch in range(batch_size):
                filename = filenames[batch]
                file_path = os.path.join(out_dir, self.cfg.model.checkpoint.split(
                    "/")[-1][:-4], filename + '.npy')
                torch.save(predictions[batch], file_path)

        predictions = predictions.view([-1] + self.output_shape)

        predictions = predictions.permute(0, 4, 1, 2, 3, 5)

        P_pos = []
        for batch in range(batch_size):
            _P_pos = []
            for ele, diameter in enumerate(self.ele_diameter):
                prediction = predictions[batch, ele]
                prediction_nms, ele_P = self.nms(prediction, diameter)
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

                save_dir = os.path.join(
                    out_dir, self.cfg.TRAIN.CHECKPOINT.split("/")[-1][:-4])
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                file_path = os.path.join(save_dir, filename + '.poscar')
                with open(file_path, 'w') as f:
                    f.write(output)


class RDF(torch.nn.Module):
    def __init__(self, cfg, delta=0.05, stop=10):
        super(RDF, self).__init__()
        self.cfg = cfg

        # some initialization
        self.elem = cfg.data.elem_name

        self.stop = stop

        self.inver_delta = 1 / delta

        self.total = int(stop * self.inver_delta)

        self.register_buffer('real_size', torch.tensor(cfg.data.real_size))
        self.register_buffer(
            'vol', self.real_size[0] * self.real_size[1] * self.real_size[2])
        self.register_buffer('slice_vol', torch.arange(
            delta/2, stop, delta) ** 1 * delta * 2 * pi * 3)

        self.register_buffer("number", torch.zeros(1))
        self.register_buffer("count", torch.zeros(
            len(self.elem), len(self.elem), self.total))

    def forward(self, pos: list):
        device = self.number.device
        for i, ele in enumerate(self.elem):
            for j, o_ele in enumerate(self.elem):
                ele_pos = pos[i]
                other_pos = pos[j]
                density = len(other_pos) / self.vol
                dist = (torch.cdist(ele_pos * self.real_size,
                        other_pos * self.real_size) * self.inver_delta).int()

                if i == j:
                    dist.fill_diagonal_(9999)

                dist_count = torch.unique(
                    dist, return_counts=True, sorted=True)
                mask = dist_count[0] < self.stop * self.inver_delta
                dist_count = (dist_count[0][mask].tolist(),
                              dist_count[1][mask]/density)
                buf = torch.zeros(self.total, device=device)
                buf[dist_count[0]] = dist_count[1]
                buf = buf / self.slice_vol / len(ele_pos)
                self.count[i, j] += buf
        self.number += 1

    def save(self):
        pass
        #TODO
        # self.count = self.count / self.number
        # path = os.path.join(LOG_DIR, f'{DATE}-RDF.npy')
        # torch.save(self.count, path)
        # self.count.zero_()
        # self.number.zero_()
