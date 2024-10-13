import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

video_root = "/mnt/data/qizhezhang/allDataset/ProBio_valid/"
results_path = './results'


def run(file_name):
    video_dir = video_root+file_name+'/images'
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    import json
    with open(os.path.join(video_dir, '../label.json'), "rb") as f:
        label = json.load(f)
        
    objects = {}
    for frame in label:
        for item in frame['items']:
            if item['object_id'] not in objects:
                seg = item['segmentation']

                from pycocotools import mask
                import numpy as np
                rle = mask.frPyObjects(seg, 480, 640)
                binary_mask = mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = np.max(binary_mask, axis=2)
                objects[item['object_id']] = {'first_frame': int(frame['path']), 'mask': binary_mask}
                
    for i,item in enumerate(objects):
        obj = objects[item]
        ann_frame_idx = obj['first_frame']  # the frame index we interact with
        ann_obj_id = int(item)  # give a unique id to each object we interact with (it can be any integers)

        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=obj['mask']
        )
        
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
        
    with open(os.path.join(results_path, f'{file_name}.txt'), 'w') as f:
        for frame,items in video_segments.items():
            for item,out_mask in items.items():
                
                rows = np.any(out_mask[0], axis=1)
                cols = np.any(out_mask[0], axis=0)
                
                if rows.any():
                    top, bottom = np.where(rows)[0][[0, -1]]
                    left, right = np.where(cols)[0][[0, -1]]    
                    width = right - left
                    height = bottom - top
                    
                    f.write(f'{frame+1}, {item}, {left}, {top}, {width}, {height}, 1, -1, -1, -1\n')
    
# from tqdm import tqdm 
# # 使用 tqdm 包装 os.listdir 循环以显示进度条
# for i in tqdm(os.listdir(video_root), desc="Processing videos"):
#     tqdm.write(f"Processing file: {i}")
#     run(i)

def visualize(file_name):
    video_dir = video_root+file_name+'/images'
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)
    
    import json
    with open(os.path.join(video_dir, '../label.json'), "rb") as f:
        label = json.load(f)
        
    objects = {}
    for frame in label:
        for item in frame['items']:
            if item['object_id'] not in objects:
                seg = item['segmentation']

                from pycocotools import mask
                import numpy as np
                rle = mask.frPyObjects(seg, 480, 640)
                binary_mask = mask.decode(rle)
                if len(binary_mask.shape) == 3:
                    binary_mask = np.max(binary_mask, axis=2)
                objects[item['object_id']] = {'first_frame': int(frame['path']), 'mask': binary_mask}
                
    for i,item in enumerate(objects):
        obj = objects[item]
        ann_frame_idx = obj['first_frame']  # the frame index we interact with
        ann_obj_id = int(item)  # give a unique id to each object we interact with (it can be any integers)

        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=obj['mask']
        )
        
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    if not os.path.exists(f'video_results/{file_name}'):
        os.makedirs(f'video_results/{file_name}')
    
    import cv2
    for out_frame_idx in range(0, len(label)):
    # for out_frame_idx in range(4, 5):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        img = Image.open(os.path.join(video_dir, f'{out_frame_idx:03d}.jpg'))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            img = np.array(img) # (480, 640, 3)
            # out_mask.shape # (1, 480, 640)
            cmap = plt.get_cmap("tab10")
            # ValueError: shape mismatch: value array of shape (765,) could not be broadcast to indexing result of shape (364,3)
            img[out_mask[0]] = np.array(cmap(out_obj_id % 10)[:3]) * 255
            
            rows = np.any(out_mask[0], axis=1)
            cols = np.any(out_mask[0], axis=0)
            
            if rows.any():
                top, bottom = np.where(rows)[0][[0, -1]]
                left, right = np.where(cols)[0][[0, -1]]
            
            # 画边界框
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
            # 画标签
            cv2.putText(img, str(out_obj_id), (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # print(out_obj_id, ":", top, left, right, bottom)
            
        plt.imsave(f'video_results/{file_name}_b/{out_frame_idx:03d}.jpg',img)
        plt.close()
        
visualize(os.listdir(video_root)[0])

# out_mask = np.array(
#       [[[False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, True , True , False, False, False, False, False],
#         [False, False, False, True , True , True , False, False, False, False],
#         [False, False, False, False, True , True , False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, False, False, False, False, False]]])

# rows = np.any(out_mask[0], axis=1)
# cols = np.any(out_mask[0], axis=0)

# if rows.any():
#     top, bottom = np.where(rows)[0][[0, -1]]
#     left, right = np.where(cols)[0][[0, -1]]

# print(top, left, right, bottom)