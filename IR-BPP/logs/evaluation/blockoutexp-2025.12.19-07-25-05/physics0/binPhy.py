import time 
import gym
import numpy as np
from torch import load
from tools import getRotationMatrix
import transforms3d
from .Interface import Interface
from .IRcreator import RandomItemCreator, LoadItemCreator, RandomInstanceCreator, RandomCateCreator, FixedListCreator
from .space import Space
from .cvTools import getConvexHullActions
import random
import threading
import copy
import trimesh, math
import torch, os

import tkinter as tk
from tkinter import messagebox
from arguments import get_args
import pybullet as p

_args = get_args()    
inference_flag = _args.inference   


def ask_user_for_index():
    """
    Show a small window asking the user for an integer in [1, 76].
    Returns the integer (0–75 index for pred[]).
    """
    root = tk.Tk()
    root.title("Diffusion-based packing")

    tk.Label(root, text="Enter order ID:").pack(padx=10, pady=5)

    entry = tk.Entry(root)
    entry.pack(padx=100, pady=5)

    user_choice = {"value": None}

    def on_start():
        try:
            v = int(entry.get())
            if 0 <= v <= 21:
                user_choice["value"] = v   # convert to 0-based index
                root.destroy()
            else:
                messagebox.showerror("Invalid", "Please enter order ID")
        except:
            messagebox.showerror("Invalid", "Please enter a valid integer.")

    start_btn = tk.Button(root, text="Start", command=on_start)
    start_btn.pack(pady=10)

    root.mainloop()
    return user_choice["value"]

if inference_flag:
    user_choice=ask_user_for_index()

def cleaned_summary(data):
    seen = set()
    cleaned_data = []

    for group in data:                     # outer list
        new_group = []
        for item_id, bins in group:        # each (id, dict)
            new_bins = {}
            for bin_key, lst in bins.items():
                new_list = []
                for x in lst:
                    if x not in seen:      # dedupe globally
                        seen.add(x)
                        new_list.append(x)
                new_bins[bin_key] = new_list
            new_group.append((item_id, new_bins))
        cleaned_data.append(new_group)  
    return cleaned_data      

def extract_packing_plan(E):
    bs, n, n, de = E.shape
    upper_cols = range(n - 2, n)

    all_packing_plans = []
    bin_item_lists = []
    summary = []

    for b in range(bs):
        bin_items = {col: {etype: [] for etype in range(1, de)} for col in upper_cols}

        for row in range(n - 2):
            for col in upper_cols:
                edge_vector = E[b, row, col]
                edge_type = np.argmax(edge_vector)
                if edge_type > 0:
                    bin_items[col][edge_type].append(row)

        # Replace empty
        for col in upper_cols:
            for et in bin_items[col]:
                if not bin_items[col][et]:
                    bin_items[col][et] = [None]

        # Store summary
        this_bin_summary = []
        for col in upper_cols:
            this_bin_summary.append((col, copy.deepcopy(bin_items[col])))

        summary.append(this_bin_summary)
        summary=cleaned_summary(summary)
        all_packing_plans.append(bin_items)

    # return flattened item lists
    packing_plan = all_packing_plans[0]
    for outer in packing_plan:
        for inner in packing_plan[outer]:
            bin_item_lists.append(packing_plan[outer][inner].copy())
    # print("summary:",summary)
    return bin_item_lists, summary
     
def clean_bin_item_lists(bin_item_lists):
    seen = set()
    cleaned = []

    # 1. Remove duplicates, keep first appearance
    for lst in bin_item_lists:
        new_lst = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                new_lst.append(x)
        cleaned.append(new_lst)

    # 2. Check if ANY element in 0–24 is missing
    all_items = set(range(25))

    missing = sorted(all_items - seen)

    # Only add missing if they truly don't appear
    if missing:
        cleaned[0].extend(missing)

    return cleaned

def non_blocking_simulation(interface, finished, id, non_blocking_result):
    succeeded, valid = interface.simulateToQuasistatic(givenId=id,
                                    linearTol=0.01,
                                    angularTol=0.01)

    finished[0] = True
    non_blocking_result[0] = [succeeded, valid]
class PackingGame(gym.Env):
    def __init__(self,
                 args
    ):
        
        self.args = args          
        args = vars(args)         
        self.resolutionAct = args['resolutionA']
        self.resolutionH   = args['resolutionH']
        self.bin_dimension = args['bin_dimension']
        self.dataset = args['dataset']
        # two medium, one small bins 
        self.bin_sequence = [
            np.round([0.34, 0.34, 0.18], decimals=6),
            np.round([0.34, 0.34, 0.18], decimals=6),
            np.round([0.24, 0.24, 0.18], decimals=6)
        ]        
        self.current_bin_index = 0
        self.bin_results = []
        self.shotInfo = args['shotInfo']

        self.scale         = args['scale']
        self.objPath       = args['objPath']
        self.meshScale     = args['meshScale']
        self.shapeDict     = args['shapeDict']
        self.infoDict      = args['infoDict']
        self.dicPath       = load(args['dicPath'])
        self.ZRotNum       = args['ZRotNum']
        self.heightMapPre  = args['heightMap']
        self.globalView    = not args['only_simulate_current']
        self.selectedAction= args['selectedAction']
        self.bufferSize    = args['bufferSize']
        self.chooseItem    = self.bufferSize > 1
        self.simulation    = args['simulation']
        self.evaluate      = args['evaluate']
        self.maxBatch      =  args['maxBatch']
        self.heightResolution   = args['resolutionZ']
        self.dataSample    = args['dataSample']
        self.dataname      = args['test_name']
        self.visual        = args['visual']
        self.non_blocking  = args['non_blocking']
        self.time_limit    = args['time_limit']
        self.inference     = args['inference'] 
        
        self.interface = None
        self.item_vec = np.zeros((1000, 9))
        self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / self.resolutionAct).astype(np.int32)
        self.space = Space(self.bin_dimension, self.resolutionAct, self.resolutionH, False,   self.ZRotNum,
                           args['shotInfo'], self.scale)
        
        if self.inference is None: 
            if self.evaluate and self.dataname is not None:
                self.item_creator = LoadItemCreator(data_name=self.dataname, infoDict=self.infoDict)
            else:
                if self.dataSample == 'category':
                    self.item_creator = RandomCateCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
                elif self.dataSample == 'instance':
                    self.item_creator = RandomInstanceCreator(np.arange(0, len(self.shapeDict.keys())), self.dicPath)
                else:
                    assert self.dataSample == 'pose'
                    self.item_creator = RandomItemCreator(np.arange(0, len(self.shapeDict.keys())))
        else:
            # Load model predictions
            pred = torch.load("/home/bakirkhon/Thesis_irregular/inference_irregular_predictions.pt")
            E=pred[user_choice]['E']            
            bin_item_lists, self.plan_summary = extract_packing_plan(E)
            # print("before:", bin_item_lists)
            if bin_item_lists[1] != [None] and len(bin_item_lists[1]) < 4:
                bin_item_lists[2] = bin_item_lists[1] + bin_item_lists[2]
                bin_item_lists[1] = [None]
            # print("after:",bin_item_lists)    
           
            # Initialize tracking of failures
            self.failed_items_per_bin = {i: [] for i in range(len(bin_item_lists))}
            self.failed_items_all = []


            if bin_item_lists[1]==[None]:
                self.bin_sequence = [
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.26, 0.26, 0.18], decimals=6)]
            elif bin_item_lists[2]==[None]:
                self.bin_sequence = [     
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.26, 0.26, 0.18], decimals=6)]
            bin_item_lists = clean_bin_item_lists([arr for arr in bin_item_lists if None not in arr])
            self.bin_item_lists_original = copy.deepcopy(bin_item_lists)      
            print("Final plan: ",bin_item_lists)      
            self.item_creator = FixedListCreator(data_name=self.dataname, infoDict=self.infoDict, lists_by_bin=bin_item_lists)        #data_name=self.dataname, infoDict=self.infoDict,

        self.total_bins = len(self.bin_sequence)
        self.next_item_vec = np.zeros((9))

        self.item_idx = 0
        self.total_items = 0

        self.transformation = []
        DownFaceList, ZRotList = getRotationMatrix(1, self.ZRotNum)
        for d in DownFaceList:
            for z in ZRotList:
                quat = transforms3d.quaternions.mat2quat(np.dot(z, d)[0:3, 0:3])
                self.transformation.append([quat[1],quat[2],quat[3],quat[0]]) # Saved in xyzw
        self.transformation = np.array(self.transformation)

        self.rotNum = self.ZRotNum
        self.act_len = self.selectedAction

        if self.chooseItem:
            self.act_len = self.bufferSize

        if not self.chooseItem:
            self.obs_len = len(self.next_item_vec.reshape(-1))

            if self.selectedAction:
                self.obs_len += self.selectedAction * 5
            else:
                self.obs_len += self.act_len
        else:
            self.obs_len = self.bufferSize

        if self.heightMapPre:
            self.obs_len += 32 * 32
            #self.obs_len += self.space.heightmapC.size

        self.observation_space = gym.spaces.Box(low=0.0, high=self.bin_dimension[2],
                                                shape=(self.obs_len,))
        self.action_space = gym.spaces.Discrete(self.act_len)

        self.tolerance = 0 # defalt 0.002

        self.episodeCounter = -1
        self.updatePeriod = 500
        self.trajs = []
        self.orderAction = 0
        self.hierachical = False

        if self.non_blocking:
            self.nullObs = np.zeros((self.obs_len))
            self.finished = [True]
            self.non_blocking_result = [None]
            self.nowTask = False
        
        self.bin_results = []
        self.item_limit = 25
        self.items_to_repack = None
        self.generated_items = []   # List of dicts: [{'seq_index': 0, 'original_id': X}, ...]
        self.repack_active = False
        self.items_per_bin = {i + 1: [] for i in range(self.total_bins)}
        self.geom_info_list = []
        self.graph_dataset=[]
        self.inference_dataset=[]



    def seed(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]

    def close(self):
        input("Press Enter to close visualization...")
        #self.interface.close()
    
    def print_bin_utilization(self):
        """
        Prints the current bin's total volume, packed volume, and utilization percentage.
        """
        # Bin volume
        bin_volume = np.prod(self.bin_dimension)

        # Calculate packed volume
        packed_volume = 0.0
        for obj_id in self.interface.objs:
            if obj_id in self.interface.meshDict:
                mesh = self.interface.meshDict[obj_id]
                packed_volume += mesh.volume/1000000

        # Utilization
        utilization = (packed_volume / bin_volume) * 100 if bin_volume > 0 else 0

        print(f"\n📦 Bin {self.current_bin_index + 1}/{self.total_bins}")
        print(f"   Size: {self.bin_dimension}")
        print(f"   Bin volume: {bin_volume:.3f}")
        print(f"   Packed volume: {packed_volume:.3f}")
        print(f"   Utilization: {utilization:.2f}%\n")

        return bin_volume, packed_volume, utilization

    def next_bin(self):
        global user_choice
        
        bin_volume, packed_volume, utilization = self.print_bin_utilization()
        """Start packing in the next bin (if available)."""
        # Save the packed results for this bin
        self.update_debug_text(utilization)

        bin_info = {
            'bin_index': self.current_bin_index + 1,
            'bin_size': self.bin_dimension.tolist(),
            'packed_items': copy.deepcopy(self.packed)
        }
        self.bin_results.append(bin_info)

        # Print the items packed in this bin
        print(f"# items packed: {len(self.packed)}")
        print("Packed items:")
        for item in self.packed:
            print(f" - {item[0]}={item[1]} at {np.round(item[2], 3)}")
                
        if self.inference:
            planned_items = self.item_creator.lists_by_bin[self.current_bin_index]
            packed_items_num = len(set([p[0] for p in self.packed]))
            # Items planned but not actually packed
            failed = planned_items[packed_items_num:]

            self.failed_items_per_bin[self.current_bin_index] = failed            
            self.failed_items_all.extend(failed)

            print(f"\n❗ Items that FAILED to pack in bin {self.current_bin_index+1}: {failed}")

        if self.inference and self.current_bin_index < self.total_bins - 1 and len(self.failed_items_all)>0:
            try:
                self.item_creator.lists_by_bin[self.current_bin_index+1]=self.failed_items_all + self.item_creator.lists_by_bin[self.current_bin_index+1]
            except:
                self.item_creator.lists_by_bin.append([])
                self.item_creator.lists_by_bin[self.current_bin_index+1]=self.failed_items_all + self.item_creator.lists_by_bin[self.current_bin_index+1]
        self.current_bin_index += 1    
        if self.inference and self.current_bin_index < self.total_bins:
            self.item_creator.reset(self.current_bin_index,user_choice)
            
        if self.current_bin_index < self.total_bins and self.total_items < self.item_limit:
            try:    
                next_bin_size = self.bin_sequence[self.current_bin_index]
                self.next_bin_size=next_bin_size
                self.interface.reset(new_bin=next_bin_size)
                self.space = Space(next_bin_size, self.resolutionAct, self.resolutionH, False, self.ZRotNum,
                                self.shotInfo, self.scale)
                self.bin_dimension = next_bin_size
                self.space.reset()
                self.rangeX_A, self.rangeY_A = np.ceil(self.bin_dimension[0:2] / self.resolutionAct).astype(np.int32)

                self.item_idx = 0
                self.packed = []
                self.packedId = []
                self.id = None
                time.sleep(2)
                self.interface.cameraForRecord()


                return self.cur_observation()
            except:
                print("No bins/items left")

        # No bins left — end episode
        print("\n✅ All bins filled for this episode.")

        if self.inference:
            new_index = ask_user_for_index()

            user_choice = new_index
            self.failed_items_all.clear()
            # Reload the predicted packing plan
            pred = torch.load("/home/bakirkhon/Thesis_irregular/inference_irregular_predictions.pt")
            E = pred[user_choice]['E']
            bin_item_lists, self.plan_summary = extract_packing_plan(E)
            if bin_item_lists[1] != [None] and len(bin_item_lists[1]) < 4:
                bin_item_lists[2] = bin_item_lists[1] + bin_item_lists[2]
                bin_item_lists[1] = [None]
            if bin_item_lists[1]==[None]:
                self.bin_sequence = [
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.26, 0.26, 0.18], decimals=6)]
            elif bin_item_lists[2]==[None]:
                self.bin_sequence = [     
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.36, 0.36, 0.18], decimals=6),
                np.round([0.26, 0.26, 0.18], decimals=6)]    
            
            # Remove empty bins
            bin_item_lists = clean_bin_item_lists([arr for arr in bin_item_lists if None not in arr])  
            print("Final plan: ",bin_item_lists)  
            self.bin_item_lists_original = copy.deepcopy(bin_item_lists)
            # Reset item creator with new fixed list
            self.item_creator = FixedListCreator(data_name=self.dataname, infoDict=self.infoDict, lists_by_bin=bin_item_lists)  #

            # Restart episode
            return self.reset()
        if self.inference is None:
            print("check: ",self.items_per_bin)
            for geom in self.geom_info_list:
                print(
                    f"ID {geom['id']:>3} | "
                    f"Class: {geom['shape_class']:<9} | "
                    f"dims(W,H,L): {geom['dims_whl'][0]:6.2f}, {geom['dims_whl'][1]:6.2f}, {geom['dims_whl'][2]:6.2f} | "
                    f"Vol: {geom['volume_estimate']:.5f} | "
                    f"Fill: {geom['outer_box_fill_factor']:.2f} | "
                    f"Aspect: {geom['aspect_ratio_long_short']:.2f} | "
                    f"Sph: {geom['sphericity']:.2f} | "
                    f"Proj(top/front/side): {geom['projected_areas']['top']:.6f}, {geom['projected_areas']['front']:.6f}, {geom['projected_areas']['side']:.6f}"
                )


        self.repack_active = False

        if self.total_items == self.item_limit and self.inference is None:
            self.build_edge_set()
            print("No items left, saving...")
            save_dir = "."
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "training_dataset_irregular.pt")
            save_path2 = os.path.join(save_dir, "inference_dataset_irregular.pt")
            torch.save(self.graph_dataset, save_path)
            torch.save(self.inference_dataset, save_path2)
        self.total_items = 0   
        return None

    def reset(self, index=None):
        """
        Reset environment at the start of a new episode.
        Keeps PyBullet connected and clears per-episode state.
        """
        # Reset bin and episode tracking
        self.episodeCounter +=  1
        self.current_bin_index = 0
        if self.inference:
            self.item_creator.reset(self.current_bin_index, user_choice)
        self.total_items = 0
        self.item_idx = 0
        self.bin_results.clear()
        self.geom_info_list.clear()
        self.generated_items.clear()
        
        self.items_per_bin = {i + 1: [] for i in range(self.total_bins)}

        # Also clear global geometry accumulator
        global Va
        Va.clear()
        
        current_bin_size = self.bin_sequence[self.current_bin_index]
        self.bin_dimension = current_bin_size
        print(f"🚀 Starting new episode | Bin {self.current_bin_index + 1}/{self.total_bins} | Size: {current_bin_size}")
        if self.inference and hasattr(self, "plan_summary"):
            print("Inference mode is active")
            print("\n📌 Predicted packing plan:")
            for (col, item_dict) in self.plan_summary[0]:
                print(f"\nBin node {col}:")
                for edge_type, items in item_dict.items():
                    print(f"  Bin {edge_type}: {items}")


        if self.interface is None:
            # First episode - create new interface
            self.interface = Interface(
                bin=self.bin_dimension,
                foldername=self.objPath,
                visual=self.visual,
                scale=self.scale,
                simulationScale=self.meshScale,
                maxBatch=self.maxBatch,
            )
        else:
            # Keep connection alive, just clear world
            self.interface.reset(new_bin=self.bin_dimension)

        # Rebuild spatial map for new bin
        self.space = Space(
            self.bin_dimension,
            self.resolutionAct,
            self.resolutionH,
            False,
            self.ZRotNum,
            self.shotInfo,
            self.scale
        )

        # Reset item generation and state tracking
        if self.inference is None: 
            self.item_creator.reset(index)
            if hasattr(self.item_creator, "item_list"):
                self.item_creator.item_list.clear()

        self.packed=[]
        self.packedId=[]
        self.next_item_vec[:] = 0
        self.item_vec[:] = 0
        self.id = None

        # Recompute grid
        self.rangeX_A, self.rangeY_A = np.ceil(
            self.bin_dimension[0:2] / self.resolutionAct
        ).astype(np.int32)
        self.update_debug_text(0.0)


        return self.cur_observation()

    
    def get_ratio(self):
        totalVolume = 0
        for idx in range(self.item_idx):
            totalVolume += self.infoDict[int(self.item_vec[idx][0])][0]['volume']
        return totalVolume / np.prod(self.bin_dimension)

    def get_item_ratio(self, next_item_ID):
        return self.infoDict[next_item_ID][0]['volume'] / np.prod(self.bin_dimension)

    def gen_next_item_ID(self):        
        return self.item_creator.preview(1)[0]

    def get_action_candidates(self, orderAction):
        self.hierachical = True
        self.next_item_ID = self.next_k_item_ID[orderAction]
        self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
        self.chooseItem = False
        locObservation  = self.cur_observation(genItem = False)
        self.chooseItem = True
        self.orderAction = orderAction
        return locObservation

    def get_all_possible_observation(self):
        self.hierachical = True
        self.chooseItem = False
        all_obs = []
        for itemID in self.next_k_item_ID:
            self.next_item_ID = itemID
            self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)
            locObservation  = self.cur_observation(genItem = False)
            all_obs.append(locObservation)
        return np.concatenate(all_obs, axis=0)


    def cur_observation(self, genItem = True, draw = False):
        if self.item_idx != 0:
            positions, orientations = self.interface.getAllPositionAndOrientation(inner=False)
            self.item_vec[0:self.item_idx, 1:4] = np.array([positions[0:self.item_idx]])
            self.item_vec[0:self.item_idx, 4:8] = np.array([orientations[0:self.item_idx]])
        if not self.chooseItem:
            if genItem:
                self.next_item_ID = self.gen_next_item_ID()
            if self.next_item_ID is None and self.current_bin_index < self.total_bins:
                return self.next_bin()
    
            self.next_item_vec[0] = self.next_item_ID
                       
            naiveMask = self.space.get_possible_position(self.next_item_ID, self.shapeDict[self.next_item_ID], self.selectedAction)


            result = self.next_item_vec.reshape(-1)

            if not self.selectedAction:
                result = np.concatenate((self.next_item_vec.reshape(-1),
                                     naiveMask.reshape(-1)))

            if self.heightMapPre:
                    fixed_hm = self._resize_heightmap(self.space.heightmapC)
                    result = np.concatenate((result, fixed_hm.reshape(-1)))

            if self.selectedAction:
                self.candidates = None
                self.candidates= getConvexHullActions(self.space.posZValid, self.space.naiveMask,
                                                             self.heightResolution)
                if self.candidates is not None:
                    if len(self.candidates) > self.selectedAction:
                        # sort with height
                        selectedIndex = np.argsort(self.candidates[:,3])[0: self.selectedAction]
                        self.candidates = self.candidates[selectedIndex]
                    elif len(self.candidates) < self.selectedAction:
                        dif = self.selectedAction - len(self.candidates)
                        self.candidates = np.concatenate((self.candidates, np.zeros((dif, 5))), axis=0)

                if self.candidates is None:
                    posz = self.space.posZValid
                    rotNum, rangeX_A, rangeY_A = posz.shape  

                    poszFlatten = posz.reshape(-1)
                    selectedIndex = np.argsort(poszFlatten)[0: self.selectedAction]

                    ROT, X, Y = np.unravel_index(selectedIndex, (rotNum, rangeX_A, rangeY_A))
                    H = poszFlatten[selectedIndex]
                    V = self.space.naiveMask.reshape(-1)[selectedIndex]

                    H[:] = self.bin_dimension[-1]

                    self.candidates = np.concatenate(
                        [ROT.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1),
                        H.reshape(-1, 1), V.reshape(-1, 1)],
                        axis=1
                    )


                result = np.concatenate((self.candidates.reshape(-1), result))
        else:
            self.next_k_item_ID = self.item_creator.preview(self.bufferSize)
            fixed_hm = self._resize_heightmap(self.space.heightmapC)
            result = np.concatenate((result, fixed_hm.reshape(-1)))

        return result

    def action_to_position(self, action):
        rotIdx, lx, ly = self.candidates[action][0:3].astype(np.int)
        return rotIdx, np.round((lx * self.resolutionAct, ly * self.resolutionAct, self.bin_dimension[2]), decimals=6), (lx,ly)

    def prejudge(self, rotIdx, translation, naiveMask):
        extents = self.shapeDict[self.next_item_ID][rotIdx].extents
        if np.round(translation[0] + extents[0] - self.bin_dimension[0], decimals=6)  > 0 \
            or np.round(translation[1] + extents[1] - self.bin_dimension[1], decimals=6) > 0:
            return False
        if np.sum(naiveMask) == 0:
            return False
        return True

    # Note the transform between Ra coord and Rh coord
    def step(self, action):
        if self.non_blocking and not self.finished[0]:
            return self.nullObs, 0.0, False, {'Valid': False}

        if self.non_blocking and self.finished[0] and self.nowTask:
            success, sim_suc = self.non_blocking_result[0]
            self.nowTask = False
            self.non_blocking_result[0] = None

        else:
            rotIdx, targetFLB, coordinate = self.action_to_position(action)
            rotation = self.transformation[int(rotIdx)]

            sim_suc = False
            success = self.prejudge(rotIdx, targetFLB, self.space.naiveMask)
            color = [random.random(), random.random(), random.random(), 1]  # random RGB + alpha
            self.id = self.interface.addObject(
                self.dicPath[self.next_item_ID][0:-4],
                targetFLB=targetFLB,
                rotation=rotation,
                linearDamping=0.5,
                angularDamping=0.5,
                color=color)
            mesh_path = './dataset/{}/shape_vhacd'.format(self.dataset)+ "/" + self.dicPath[self.next_item_ID][0:-4] + ".obj"
            if success and not self.repack_active:
                geom_params = compute_geom_params(mesh_path)
                if self.inference is None: 
                    self.geom_info_list.append({
                        "id": self.item_creator.item_set.index(self.next_item_ID),
                        **geom_params
                    })
            # record new object in sequence
            if success and self.inference is None: 
                self.record_item(self.next_item_ID)
            # self.id = self.interface.addObject(self.dicPath[self.next_item_ID][0:-4], targetFLB = targetFLB, rotation = rotation,
            #                               linearDamping = 0.5, angularDamping = 0.5)

            height = self.space.posZmap[rotIdx, coordinate[0], coordinate[1]]
            self.interface.adjustHeight(self.id , height + self.tolerance)

            if success:
                if self.simulation:
                    if self.non_blocking:
                        self.finished[0] = False
                        subProcess = threading.Thread(target=non_blocking_simulation, args=(self.interface, self.finished, self.id, self.non_blocking_result))
                        subProcess.start()
                        self.nowTask = True
                        start_time = time.time()
                        end_time   = start_time
                        while end_time - start_time < self.time_limit:
                            end_time = time.time()
                            if self.finished[0]:
                                break
                        if not self.finished[0]:
                            return self.nullObs, 0.0, False, {'Valid': False}
                    else:
                        success, sim_suc = self.interface.simulateToQuasistatic(givenId=self.id,
                                                                            linearTol = 0.01,
                                                                            angularTol = 0.01)
                else:
                    success, sim_suc = self.interface.simulateHeight(self.id)

        if not self.globalView:
            self.interface.disableObject(self.id)

        bounds = self.interface.get_wraped_AABB(self.id, inner=False)
        positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(self.id, inner=False)
        if success:
            self.packed.append([self.next_item_ID, self.dicPath[self.next_item_ID], positionT, orientationT])
            self.total_items += 1        
            self.packedId.append(self.id)

        if not success:
             
            if self.globalView and self.evaluate:
                for replayIdx, idNow in enumerate(self.packedId):
                    positionT, orientationT = self.interface.get_Wraped_Position_And_Orientation(idNow, inner=False)
                    self.packed[replayIdx][2] = positionT
                    self.packed[replayIdx][3] = orientationT
            reward = 0.0
            info = {'counter': self.item_idx,
                    'ratio': self.get_ratio(),
                    'Valid': True,
                    }
            observation = self.cur_observation()
            # Instead of ending the episode completely:            
            new_obs = self.next_bin()
            if new_obs is None:
                return observation, reward, True, info  # episode end
            else:
                return new_obs, reward, False, info


        if sim_suc:          
            if self.globalView:
                self.space.shot_whole()
            else:
                self.space.place_item_trimesh(self.shapeDict[self.next_item_ID][0], (positionT, orientationT), (bounds, self.next_item_ID))

            self.item_vec[self.item_idx, 0] = self.next_item_ID
            self.item_vec[self.item_idx, -1] = 1
            item_ratio = self.get_item_ratio(self.next_item_ID)
            
            reward = item_ratio * 10
            self.item_idx += 1
            if success: 
                self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # add a new box to the list                 
            observation = self.cur_observation()

            if observation is None:
                # Convert to a terminal zero-vector obs
                observation = np.zeros(self.obs_len, dtype=np.float32)
                return observation, 0.0, True, {'Valid': True}

            bin_volume = np.prod(self.bin_dimension)
            packed_volume = sum(
                self.interface.meshDict[obj].volume / 1_000_000
                for obj in self.interface.objs if obj in self.interface.meshDict
            )
            utilization = (packed_volume / bin_volume) * 100 
            self.update_debug_text(utilization)
            if self.total_items >= self.item_limit:
                print(f"\n Reached item limit ({self.item_limit})")
                for item in self.packed:
                    bin_volume = np.prod(self.bin_dimension)
                    packed_volume = sum(
                        self.interface.meshDict[obj].volume / 1_000_000
                        for obj in self.interface.objs if obj in self.interface.meshDict
                    )
                    utilization = (packed_volume / bin_volume) * 100 
                    self.update_debug_text(utilization)
                    if self.current_bin_index == 1 and utilization < 17.5:
                        print("... Needs repacking. Low utilization")
                        self.repack_active = True
                        self.items_per_bin[2].clear()
                        self.items_to_repack = [itm[0] for itm in self.packed]
                        self.total_items = self.total_items - len(self.items_to_repack)
                        self.item_creator.item_list = [itm[0] for itm in self.packed]
                        self.item_creator.item_list.append(0)

                        new_obs = self.next_bin()
                        if new_obs is None:
                            return observation, reward, True, {'Valid': True}  # episode end


                        # Reset environment counters
                        self.item_idx = 0
                        self.packed = []
                        self.packedId = []

                        print(f"♻️  Repacking {len(self.items_to_repack)} items into Bin 3...")
                        return new_obs, reward, False, {'Valid': True}
                    else:
                        self.next_bin()
                        return observation, reward, True, {'Valid': True}    
            return observation, reward, False, {'Valid': True}
        else:
            # Invalid call
            self.packed.pop()
            self.packedId.pop()
            delId = self.interface.objs.pop()
            self.interface.removeBody(delId)
            self.item_creator.update_item_queue(self.orderAction)
            self.item_creator.generate_item()  # Add a new box to the list
            observation = self.cur_observation()
            return observation, 0.0, False, {'Valid': False}

    def record_item(self, item_id):
        """
        Store each placed item with a sequential index, bin number, and PyBullet ID.
        Also update items_per_bin dictionary.
        """
        record = {
            "seq_index": self.item_creator.item_set.index(item_id),
            "item_id": int(item_id)
        }
        self.generated_items.append(record)
        # Add this item to the current bin’s list
        bin_num = self.current_bin_index + 1
        if bin_num not in self.items_per_bin:
            self.items_per_bin[bin_num] = []
        self.items_per_bin[bin_num].append(self.item_creator.item_set.index(item_id))

    def build_edge_set(self):
        """
        Build an edge tensor E representing item–bin relationships.
        Uses self.items_per_bin and self.bin_sequence.
        Returns:
            E: np.ndarray of shape (n, n, num_edge_types)
            where the last dimension encodes bin categories (small, medium, large).
        """
        # 1. Define bin size categories
        bin_size_map = {'small': [], 'medium': [], 'large': []}

        # classify each bin in the sequence
        for i, dims in enumerate(self.bin_sequence):
            s = sum(dims)
            if s < 0.8:  # adjust threshold (meters)
                bin_size_map['small'].append(i)
            elif 0.8 <= s <= 1.5:
                bin_size_map['medium'].append(i)
            else:
                bin_size_map['large'].append(i)

        # 2. Build node sets
        all_items = []
        for items in self.items_per_bin.values():
            all_items.extend(items)

        num_items = len(all_items)

        # 4. Map bin dims to index in Vb space
        unique_bin_dims = {tuple(map(float, b)) for b in self.bin_sequence}
        Vb = np.array(sorted(list(unique_bin_dims), reverse=True))

        # create zero array and concatenate
        Vb_extended = np.hstack((Vb, np.zeros((Vb.shape[0], 7))))
        n = num_items + Vb.shape[0]
        E = np.zeros((n, n, 3))
        E[:, :, 0] = 1  # initially "no edge"
        bin_size_to_vb_index = {tuple(dim): i for i, dim in enumerate(Vb)}

        # 5. Build edges per bin category
        for bin_category, bin_indices in bin_size_map.items():
            for local_idx, bin_global_index in enumerate(bin_indices):
                bin_dims = tuple(map(float, self.bin_sequence[bin_global_index]))
                vb_index = bin_size_to_vb_index[bin_dims]
                bin_index = num_items + vb_index

                for item_id in self.items_per_bin[bin_global_index + 1]:
                    item_index = int(item_id)

                    # Clear "no edge"
                    E[item_index, bin_index, 0] = 0
                    E[bin_index, item_index, 0] = 0

                    # One-hot encode this edge type (1 offset)
                    edge_channel = local_idx + 1
                    E[item_index, bin_index, edge_channel] = 1
                    E[bin_index, item_index, edge_channel] = 1

                
        # Construct final node matrix V
        Va_arr = np.array(Va, dtype=np.float32)  # from global
        V = np.concatenate([Va_arr, Vb_extended.astype(np.float32)], axis=0)

        # Build dictionary to save 
        graph_data = {
            "X": torch.from_numpy(V),
            "E": torch.from_numpy(E),
        }

        inference_data = {
            "X": torch.from_numpy(V),
            "traj": self.item_creator.traj
        }
        self.graph_dataset.append(graph_data)
        self.inference_dataset.append(inference_data)
        return E
    
    def update_debug_text(self, utilization=0.0):
        """
        Draws/updates bin index and utilization text in the GUI.
        Overwrites the previous text by keeping track of its ID.
        """
        text = (
            f"Ep {self.episodeCounter} | Bin {self.current_bin_index + 1}/{self.total_bins}: {utilization:.2f}%"
        )

        # Text position inside the GUI
        pos = [0, 50, self.bin_dimension[2] + 0.5]  # above the bin

        # Remove previous debug text
        if hasattr(self, "debug_text_id") and self.debug_text_id is not None:
            p.removeUserDebugItem(self.debug_text_id)

        # Draw text
        self.debug_text_id = p.addUserDebugText(
            text,
            pos,
            textColorRGB=[1, 1, 1],
            textSize=2.5,
            lifeTime=0  # 0 = persistent until removed
        )
    def _resize_heightmap(self, heightmap):
        """
        Resize / pad / crop heightmap to fixed size expected by the network.
        """
        TARGET = 32  # must match training

        h, w = heightmap.shape
        out = np.zeros((TARGET, TARGET), dtype=heightmap.dtype)

        # copy overlapping region
        min_h = min(h, TARGET)
        min_w = min(w, TARGET)
        out[:min_h, :min_w] = heightmap[:min_h, :min_w]

        return out
    


Va=[]
def compute_geom_params(mesh_path):
        """Extract simplified geometry parameters from a mesh file."""
        mesh = trimesh.load(mesh_path, force='mesh', process=True)
        if mesh.is_empty:
            return None

        # Get oriented bounding box
        try:
            transform, extents = trimesh.bounds.oriented_bounds(mesh)
            extents = np.array(extents, dtype=float)
        except Exception:
            extents = np.array(mesh.extents, dtype=float)

        # Sort dimensions W≤H≤L
        ex, ey, ez = sorted(extents.tolist())
        W, H, L = float(ex), float(ey), float(ez)

        bbox_vol = W * H * L

        # Volume estimate
        try:
            used_vol = float(mesh.volume)
        except Exception:
            print("Exception active")
            used_vol = 0.0
        # if used_vol <= 0.0:
        #     try:
        #         used_vol = float(mesh.convex_hull.volume)
        #     except Exception:
        #         used_vol = 0.0

        # Clamp extreme values
        if bbox_vol > 0 and used_vol > bbox_vol:
            used_vol = bbox_vol * 0.995

        fill = used_vol / bbox_vol if bbox_vol > 0 else 0.0
        fill = float(np.clip(fill, 0.0, 1.0))

        # Projected areas (from bbox)
        top = W * L
        front = W * H
        side = H * L
        projected_areas={"top": float(top), "front": float(front), "side": float(side)}


        # Aspect ratio and sphericity
        aspect = L / max(W, 1e-9)
        height_ratio = H / max((W + L) / 2.0, 1e-12)
        try:
            area = float(mesh.area)
        except Exception:
            area = float(mesh.convex_hull.area)
        spher = (math.pi ** (1/3)) * ((6 * used_vol) ** (2/3)) / area if (used_vol > 0 and area > 0) else 0.0

        # Shape class heuristic
        def classify(fill, aspect, height_ratio, spher):
            if spher >= 0.85:
                return "round"
            if aspect >= 3.0:
                return "long"
            if height_ratio <= 0.3 and aspect <= 3.0:
                return "flat"
            if height_ratio >= 1.5 and aspect <= 2.0:
                return "tall"
            if fill >= 0.9:
                return "boxy"
            return "irregular"

        shape_class = classify(fill, aspect, height_ratio, spher)
        Va.append(np.array([W, H, L, used_vol, fill, aspect, spher, top, front, side]))
        return {
            "dims_whl": [W, H, L],
            "volume_estimate": used_vol,
            "outer_box_fill_factor": fill,
            "aspect_ratio_long_short": aspect,
            "sphericity": spher,
            "shape_class": shape_class,
            "projected_areas": {"top": top, "front": front, "side": side}
        }
