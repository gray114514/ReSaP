import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image , ImageDraw
import utils.transforms as T
from utils.prompt import parse_bboxes_qwen3vl , sample_points_adaptive_grid , get_GRPO_point_prompt , parse_GRPO_point_answer , parse_GRPO_box_answer , parse_box_coords , sample_points_adaptive_random , get_qwen3_point_prompt , Threshold
from utils.visualize import visualize_and_save_pil
from utils.segment_anything import sam_model_registry, SamPredictor
import json
from peft import PeftModel
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--option",type=str,default="RRSISD")
parser.add_argument("--version",type=str)
parser.add_argument("--eval_model_path",type=str,default="models/Qwen/Qwen3-VL-8B-Instruct")
parser.add_argument("--data_root",type=str,default="")
parser.add_argument("--use_think",type=lambda x : x.lower() == 'true', default='False')
parser.add_argument("--use_lora",type=lambda x : x.lower() == 'true' , default = 'False')
parser.add_argument('--adapter_path',type=str,default='default')
parser.add_argument("--save_img",type=lambda x : x.lower() == 'true' , default='False')
parser.add_argument('--num_return_sequences',type=int,default=1)
parser.add_argument('--vlm_type',type=str,default='qwen3')
parser.add_argument('--size',type=int,default=1000)
parser.add_argument('--split',type=str,default='test')
parser.add_argument('--is_random',type=lambda x : x.lower() == 'true',default='False')
parser.add_argument("--max_step",type=int,default=100)
parser.add_argument("--sam_ckpt",type=str,default="")
parser.add_argument("--log_dir",type=str,default="")
args = parser.parse_args()
option = args.option
version = args.version
dir_key = args.dir_key
eval_model_path = args.eval_model_path
use_think = args.use_think
use_lora = args.use_lora
adapter_path = args.adapter_path
num_return_sequences = args.num_return_sequences
save_img = args.save_img
vlm_type = args.vlm_type
IMG_SIZE = args.size
split = args.split
is_random = args.is_random
max_step = args.max_step
DATA_ROOT = args.data_root
sam_ckpt = args.sam_ckpt
log_dir = args.log_dir

def load_sam_predictor(checkpoint, model_type="vit_l", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)

def to_uint8_hwc(img_tensor):
    if isinstance(img_tensor, torch.Tensor):
        if img_tensor.max() <= 1.0:
            img_tensor = img_tensor * 255.0
        img = img_tensor.byte().permute(1,2,0).cpu().numpy()
    else:
        # PIL Image
        img = np.array(img_tensor.convert("RGB"))
    return img

@torch.no_grad()
def sam_predict(
    predictor,
    img_tensor,
    points=None,             # [[x,y], ...]
    point_labels=None,      
    boxes=None,              
    multimask_output=False,
    mask_input=None,
):
    image_np = to_uint8_hwc(img_tensor)  # HxWx3 uint8 RGB
    predictor.set_image(image_np)
    H, W = image_np.shape[:2]
    device = predictor.device

    pts_t = None
    lbls_t = None
    if points is not None and len(points) > 0:
        pts_np = np.asarray(points, dtype=np.float32)     # (N,2) or (2,)
        if pts_np.ndim == 1:
            pts_np = pts_np[None, :]                      # (1,2)

        # labels
        if point_labels is None:
            point_labels = [1] * pts_np.shape[0]
        labels_np = np.asarray(point_labels, dtype=np.int64)
        if labels_np.ndim == 0:
            labels_np = labels_np[None]
        assert pts_np.shape[0] == labels_np.shape[0], "points 与 point_labels 数量不一致"

        pts_no_batch = torch.from_numpy(pts_np).float()   # (N,2), CPU tensor
        pts_no_batch = predictor.transform.apply_coords_torch(pts_no_batch, (H, W))  # (N,2)

        pts_t = pts_no_batch[None, ...].to(device)        # (1,N,2)
        lbls_t = torch.from_numpy(labels_np)[None, ...].to(device)  # (1,N)

    boxes_t = None
    if boxes is not None and len(boxes) > 0:
        boxes_np = np.asarray(boxes, dtype=np.float32)    # (M,4)
        boxes_t = torch.from_numpy(boxes_np)              # CPU
        boxes_t = predictor.transform.apply_boxes_torch(boxes_t, (H, W))  # (M,4)
        boxes_t = boxes_t.to(device)

    if (pts_t is not None) and (boxes_t is not None):
        Bb = boxes_t.shape[0]            # M
        if Bb > 1 and pts_t.shape[0] == 1:
            pts_t  = pts_t.repeat(Bb, 1, 1)   # (M,N,2)
            lbls_t = lbls_t.repeat(Bb, 1)     # (M,N)

    masks, scores, logits = predictor.predict_torch(
        point_coords=pts_t,           
        point_labels=lbls_t,         
        boxes=boxes_t,                
        multimask_output=multimask_output,
        mask_input=mask_input         
    )
    return masks, scores, logits


def compute_iou_binary(pred_mask, gt_mask):
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
    if gt_mask.ndim == 3 and gt_mask.shape[0] == 1:
        gt_mask = gt_mask[0]
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask > 0).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    print(inter,union)
    return float(inter) / float(union) if union > 0 else 0.0 , float(inter) , float(union)


def run_one_image_sam_iou(
    predictor,
    image,
    gt_mask_tensor,
    points=None,
    point_labels=None,
    boxes=None,
    multimask_output=False,
    pick="best_score"
):
    masks, scores, logits = sam_predict(
        predictor, img_tensor=image ,
        points=points, point_labels=point_labels,
        boxes=boxes, multimask_output=multimask_output
    )
    # masks: [B,C,H,W] 或 [C,H,W]  (torch.Tensor)
    # scores: [B,C] 或 [C]         (torch.Tensor)

    if masks.dim() == 4:                              # [B,C,H,W]
        B, C, H, W = masks.shape
        s = scores                                     # [B,C]
        if pick == "best_score":
            idx_each = s.argmax(dim=1)                 # [B]
        else:
            idx_each = torch.zeros(B, dtype=torch.long, device=s.device)
        picked = masks[torch.arange(B, device=masks.device), idx_each]  # [B,H,W]
        pred_t = picked.sum(dim=0)                   
    else:                                              # [C,H,W]
        s = scores.reshape(-1)                         # [C]
        idx = int(s.argmax()) if (pick == "best_score" and s.numel() > 0) else 0
        pred_t = masks[idx]                            # [H,W]

    pred_np = pred_t.detach().cpu().numpy().astype(np.uint8)

    iou , inter , union = compute_iou_binary(pred_np, gt_mask_tensor)
    return iou, pred_np , inter , union



from transformers import AutoProcessor ,  GenerationConfig 
dir_key_1 = "uid01220"
device = "cpu"
if vlm_type == 'qwen3':
    from transformers import Qwen3VLForConditionalGeneration 
    processor = AutoProcessor.from_pretrained(eval_model_path , padding_side="left")  # <- 改成你用的
    model =  Qwen3VLForConditionalGeneration.from_pretrained(eval_model_path,dtype="auto",attn_implementation="sdpa").cuda()
elif vlm_type == 'qwen2':
    from transformers import Qwen2VLForConditionalGeneration 
    processor = AutoProcessor.from_pretrained(eval_model_path , padding_side="left")  # <- 改成你用的
    model =  Qwen2VLForConditionalGeneration.from_pretrained(eval_model_path,dtype="auto",attn_implementation="sdpa").cuda()

if use_lora:
    model = PeftModel.from_pretrained(model,adapter_path)
model.eval()

def get_transform(img_size=1024):
    transforms = [
        T.Resize(img_size, img_size),
    ]
    return T.Compose(transforms)

if  option == 'RRSISD':
    from utils.RRSISD_dataset import ReferDataset , RRSISDDataCollator

    dataset = ReferDataset(refer_data_root=DATA_ROOT,split=split,eval_mode=True , image_transforms=get_transform(IMG_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=1,           
        shuffle=False,
        num_workers=0,
        collate_fn=RRSISDDataCollator(processor,use_think=use_think)
    )
elif option == 'ris_lad':
    from utils.RRSISD_dataset import ReferDataset , RRSISDDataCollator
    dataset = ReferDataset(refer_data_root = DATA_ROOT, dataset = 'ris_lad' , split=split,eval_mode=True , image_transforms=get_transform(IMG_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=1,           
        shuffle=False,
        num_workers=0,
        collate_fn=RRSISDDataCollator(processor,use_think=use_think)
    )
    
predictor = load_sam_predictor(sam_ckpt, model_type="vit_l", device="cuda")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_text = f"{log_dir}/Iou.log"
log_image = f"{log_dir}/image"
log_choice = f"{log_dir}/choice.log"
if not os.path.exists(log_image):
    os.makedirs(log_image)
with open(log_text,"w",encoding='utf-8') as f:
    pass
with open(log_choice,"w",encoding='utf-8') as f:
    pass
Ious = []
boxes_iou = []
choices = []
print(len(loader))

generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True, 
            temperature=1.0,
            top_p = 0.9 ,
            num_return_sequences=num_return_sequences
        )

from shapely.geometry import box as shapely_box
def box_iou_rewards(pred, answer):
    if pred is None or len(pred) == 0 or len(answer) == 0:
        return 0.0
    ious = []
    for p_box, gt_box in zip(pred, answer):
        p = shapely_box(*p_box)
        g = shapely_box(*gt_box)
        inter = p.intersection(g).area
        union = p.union(g).area
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

    return sum(ious) / len(ious)

for idx , batch in enumerate(loader):
    
    if idx >= max_step:
        break
    
    pil_image, tgt_batch, qwen_inputs , sentences , ref_boxes = batch
    
    boxes , points , point_labels = None , None , None
    neg_points , pos_points , count = None , None , None

    W, H = pil_image[0].size
    with torch.inference_mode():
        out_ids = model.generate(
            **qwen_inputs.to(model.device),
            generation_config=generation_config
        )
        all_text = processor.batch_decode(out_ids,skip_special_tokens=True)[0]
        out_ids = out_ids[:, qwen_inputs.input_ids.shape[1]:]
    texts = processor.batch_decode(out_ids,skip_special_tokens=True)
    choice_iou , choice_box_iou = 0.0 , 0.0
    for text in texts:
        print(text)
        box_iou = 0.0
        boxes = False 
        if vlm_type == 'qwen2':
            boxes = parse_box_coords(text)
        if not boxes:
            boxes = parse_GRPO_box_answer(text)
        if not boxes:
            boxes = parse_bboxes_qwen3vl(text)
        if not boxes:
            Ious.append((0.0,0,0,0,0))
            boxes_iou.append(box_iou)
            with open(log_text,"a",encoding="utf-8") as f:
                f.write(json.dumps({"idx" : idx , "iou" : 0.0 , "box_iou" : 0.0 , "output_text" : text , "count" : count , "all_text" : all_text}) + "\n")
            continue
        boxes = [ tuple(round( min(max(0,i),999) / 1000 * W) for i in b) for b in boxes]
        box_iou = box_iou_rewards(boxes,ref_boxes)
        boxes_iou.append(box_iou)
        choice_box_iou = max(choice_box_iou,box_iou)
        print(boxes)
        count = len(boxes)
        area_ratio = (sum(max(0, x2-x1) * max(0, y2-y1) for x1, y1, x2, y2 in boxes) / len(boxes) if boxes else 0) / (H * W)

        if is_random:
            points , points_labels_text = sample_points_adaptive_random(boxes=boxes,W=W,H=H)  
        else:
            points , points_labels_text = sample_points_adaptive_grid(boxes=boxes,W=W,H=H) 
        if use_think:
            points_labels_prompt , messages = get_GRPO_point_prompt(processor,sentences[0],points_labels_text)
        else:
            points_labels_prompt , messages = get_qwen3_point_prompt(processor,sentences[0],points_labels_text) 
        #annot_img = draw_points_numbered(pil_image[0], points, color=(0,255,255), r=4, thickness=2)
        qwen_inputs = processor(
                images=pil_image[0],
                text=points_labels_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
        ).to(model.device)
        with torch.inference_mode():
            all_ids = model.generate(
                **qwen_inputs.to(model.device),
                generation_config = generation_config
            )
            all_text = processor.batch_decode(all_ids,skip_special_tokens=True)[0]
            out_ids = all_ids[:, qwen_inputs.input_ids.shape[1]:]
        
        text = processor.batch_decode(out_ids,skip_special_tokens=True)[0]
        point_labels = parse_GRPO_point_answer(text)
        print(text)
        print(point_labels)
        

        if not point_labels or (len(point_labels) != len(points)):
            point_labels = None
            points = None
            # Ious.append((0.0,0.0,0.0))
            # with open(log_text,"a",encoding="utf-8") as f:
            #     f.write(json.dumps({"idx" : idx , "iou" : 0.0 , "box_iou" : box_iou , "output_text" : text , "count" : count , "all_text" : all_text}) + "\n")
        else:
            point_tmp , point_labels_tmp = [] , []
            is_use = Threshold(model,all_ids,qwen_inputs,len(points))
            for _ in range(len(is_use)):
                if is_use[_]:
                    point_tmp.append(points[_])
                    point_labels_tmp.append(point_labels[_])
            points = point_tmp
            point_labels = point_labels_tmp
            if len(points) == 0:
                points , point_labels = None , None
        iou, pred_mask, inter , union = run_one_image_sam_iou(
            predictor, pil_image[0], tgt_batch[0],
            points=points , point_labels=point_labels,
            boxes = boxes ,
            multimask_output=True, 
            pick="best_score"       
        )
        choice_iou = max(choice_iou,iou)
        print(f"IoU = {iou:.4f}, pred_mask shape = {pred_mask.shape}")

        print(pred_mask.shape)

        neg_points = []
        pos_points = []
        TP , FP , TN , FN = 0 , 0 , 0 , 0
        if points:
            for i in range(len(points)):
                if point_labels[i] == 1:
                    pos_points.append(points[i])
                    if tgt_batch[0][points[i][0] , points[i][1]] == 1:
                        TP += 1
                    else: FP += 1
                else:
                    neg_points.append(points[i])
                    if tgt_batch[0][points[i][0] , points[i][1]] == 0:
                        TN += 1
                    else: FN += 1

        Ious.append((iou,inter,union))
        with open(log_text,"a",encoding="utf-8") as f:
            f.write(json.dumps({"idx" : idx , "iou" : iou , "box_iou" : box_iou , "output_text" : text , "count" : count , "all_text" : all_text , "TP" : TP , "FP" : FP , "TN" : TN , "FN" : FN} ) + "\n")
        # img: torch.Tensor [3,H,W] (0~1) 或 numpy [H,W,3] (0~255)
        # tgt_mask / pred_mask: [H,W] 或 [1,H,W] 0/1
        if save_img:
            overlay_gt, overlay_pred, both , vis_points = visualize_and_save_pil(img_pil=pil_image[0],tgt_mask=tgt_batch[0],neg_points=neg_points,pos_points=pos_points,pred_mask= pred_mask, boxes=boxes,out_dir=f"{log_image}/{idx}")
    with open(log_choice,"a",encoding='utf-8') as f:
        f.write(json.dumps({"idx" : idx , "iou" : choice_iou , "box_iou" : choice_box_iou})  + "\n")
    choices.append((choice_iou,choice_box_iou))
mIou , Sinter , Sunion = 0.0 , 0.0 , 0.0
for item in Ious:
    mIou += item[0]
    Sinter += item[1]
    Sunion += item[2]
best_iou , best_box_iou = zip(*choices)
best_iou , best_box_iou = list(best_iou) , list(best_box_iou)
print(f"mIou = {mIou / len(Ious)} oIou = {Sinter / Sunion}")
print(sum(boxes_iou) / len(boxes_iou))
print(f"best_iot = {sum(best_iou) / len(best_iou)} best_box_iou = {sum(best_box_iou) / len(best_box_iou)}")