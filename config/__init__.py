from yacs.config import CfgNode as CN

class CfgNode(CN):
    def __init__(self):
        super(CfgNode, self).__init__()

    def merge_from_dict(self, dic):
        for key in self:
            for sub_key in self[key]:
                if sub_key in dic and dic[sub_key] is not None and dic[sub_key] != "":
                    self.merge_from_list(
                        [f"{key}.{sub_key}", f"{dic[sub_key]}"])
