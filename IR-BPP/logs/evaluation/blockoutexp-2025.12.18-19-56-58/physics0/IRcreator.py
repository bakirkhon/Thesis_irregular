import numpy as np
import copy
import torch


# 如果想把dict改掉的话，主要是修改这里的逻辑
class ItemCreator(object):
    # 存一个查shape的字典， 再存一个记录编号的list
    def __init__(self):
        self.item_dict = {}   # 根据编号查shape的字典
        self.item_list = []   # 已经存放的item的编号

    def reset(self, index=None):
        self.item_list.clear()

    def generate_item(self, **kwargs):
        pass

    def preview(self, length):
        while len(self.item_list) < length:
            self.generate_item()
        return copy.deepcopy(self.item_list[:length])

    def update_item_queue(self, index):
        assert len(self.item_list) >= 0
        self.item_list.pop(index)


class RandomItemCreator(ItemCreator):
    def __init__(self, item_set):
        super().__init__()
        self.item_set = item_set
        print(self.item_set)

    def generate_item(self):
        self.item_list.append(np.random.choice(self.item_set))


class RandomInstanceCreator(ItemCreator):
    def __init__(self, item_set, dicPath):
        super().__init__()
        self.inverseDict = {}

        for k in dicPath.keys():
            key_name = dicPath[k][0:-6]
            if key_name not in self.inverseDict.keys():
                self.inverseDict[key_name] = [k]
            else:
                self.inverseDict[key_name].append(k)

        self.item_set = item_set
        print(self.item_set)
        print(self.inverseDict)

    def generate_item(self):
        name = np.random.choice(list(self.inverseDict.keys()))
        self.item_list.append(np.random.choice(self.inverseDict[name]))


class RandomCateCreator(ItemCreator):
    def __init__(self, item_set, dicPath):
        super().__init__()
        self.categories = {
            'objects': 0.34,
            'concave': 0.33,
            'board': 0.33
        }
        self.objCates = {}

        for key in self.categories.keys():
            self.objCates[key] = []

        for k, item in zip(dicPath.keys(), dicPath.values()):
            cate, item = item.split('/')
            self.objCates[cate].append(k)

        self.item_set = item_set
        print(self.item_set)
        print(self.objCates)

    def generate_item(self):
        name = np.random.choice(list(self.categories.keys()))
        self.item_list.append(np.random.choice(self.objCates[name]))


class LoadItemCreator(ItemCreator):
    def __init__(self, data_name=None, infoDict=None):
        super().__init__()
        self.data_name = data_name
        self.traj_index = 0
        self.item_index = 0
        self.infoDict = infoDict  # store volume lookup table

        print("Load dataset set: {}".format(data_name))
        self.item_trajs = torch.load(self.data_name)
        self.traj_nums = len(self.item_trajs)

    def reset(self, traj_index=None):
        self.item_list.clear()

        if traj_index is None:
            self.traj_index += 1
        else:
            self.traj_index = traj_index

        # Load trajectory
        self.traj = self.item_trajs[self.traj_index]

        if self.infoDict is not None:
            def score(item):
                if item is None or item == -1:
                    return -1  # smallest possible

                # get extents for the FIRST rotation (approx. shape)
                dx, dy, dz = self.infoDict[item][0]['extents']
                vol = self.infoDict[item][0]['volume']
                max_dim = max(dx, dy, dz)
                footprint = dx * dy  # base area

                # weighted hybrid score:
                return (0 * max_dim + 1 * footprint + 0.5 * vol)

            self.traj = sorted(self.traj, key=score, reverse=True)

            self.traj_2quarter = self.traj[11:15]
            self.traj_3quarter = self.traj[15:19]
            self.traj_4quarter = self.traj[-4:]

            self.traj[11:15] = self.traj_4quarter
            self.traj[15:19] = self.traj_2quarter
            self.traj[-4:] = self.traj_3quarter

        self.item_index = 0
        self.item_set = self.traj.copy()
        self.item_set.append(None)

    def generate_item(self, **kwargs):
        if self.item_index < len(self.item_set):
            self.item_list.append(self.item_set[self.item_index])
            self.item_index += 1
        else:
            self.item_list.append(-1)
            self.item_index += 1


class FixedListCreator(ItemCreator):
    """
    Loads a trajectory, 
    and uses lists_by_bin to specify which *indices* from the sorted 
    trajectory belong to each bin.
    """

    def __init__(self, data_name, infoDict, lists_by_bin):
        super().__init__()

        # Load trajectory file (same as LoadItemCreator)
        self.data_name = data_name
        self.infoDict = infoDict
        self.lists_by_bin = lists_by_bin  # list of index-lists
        self.current_list = []
        self.index = 0
        self.traj_index = 0

        self.item_trajs = torch.load(self.data_name)
        self.traj_nums = len(self.item_trajs)

    def reset(self, bin_idx, traj_index):

        self.item_list.clear()
        self.index = 0

        # KEEP THIS (as you requested)
        trajs = torch.load(
            "/home/bakirkhon/Thesis/3D-bin-packing-master/dataset/inference_dataset_irregular.pt"
        )
        sorted_traj = trajs[traj_index]["traj"]

        # ---- scoring function (define ONCE) ----
        def score(item):
            if item is None or item == -1:
                return -1e12

            dx, dy, dz = self.infoDict[item][0]['extents']
            vol = self.infoDict[item][0]['volume']
            footprint = dx * dy

            return (1.0 * footprint + 1.5 * vol)

        if bin_idx < len(self.lists_by_bin):
            indices = self.lists_by_bin[bin_idx]

            # 1️⃣ collect items
            self.current_list = [
                sorted_traj[idx]
                for idx in indices
                if idx < len(sorted_traj)
            ]

            # 2️⃣ sort ONCE
            self.current_list = sorted(self.current_list, key=score, reverse=True)

            # 3️⃣ append single terminator
            self.current_list.append(None)

        else:
            self.current_list = [None]

    # def reset(self, bin_idx, traj_index):

    #     self.item_list.clear()
    #     self.index = 0

    #     # Load next trajectory
    #     trajs = torch.load("/home/bakirkhon/Thesis/3D-bin-packing-master/dataset/inference_dataset_irregular.pt")
    #     sorted_traj = trajs[traj_index]["traj"]
    #     # ----- SELECT ITEMS FOR THIS BIN -----
    #     if bin_idx < len(self.lists_by_bin):
    #         indices = self.lists_by_bin[bin_idx]
    #         # extract items at these indices
    #         self.current_list = []
    #         for idx in indices:
    #             if idx < len(sorted_traj):
    #                 self.current_list.append(sorted_traj[idx])
    #             else:
    #                 self.current_list.append(None)
    #     else:
    #         # No more bins -> produce only None
    #         self.current_list = [None]

    #     # print(f"\n Bin {bin_idx+1}: fixed indices = {self.lists_by_bin[bin_idx]}")
    #     # print("Retrieved items:", self.current_list)

    def preview(self, length):
        """
        Return next <length> items from current_list.
        When exhausted → return [None] * length.
        """

        remaining = len(self.current_list) - self.index

        if remaining <= 0:
            return [None] * length

        # slice
        items = self.current_list[self.index : self.index + length]

        # pad with Nones
        while len(items) < length:
            items.append(None)

        return copy.deepcopy(items)

    # Consume exactly 1 item per step
    def update_item_queue(self, orderAction):
        self.index += 1

    def generate_item(self):
        """Not used — items are fixed."""
        pass
