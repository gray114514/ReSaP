import numpy as np
import torch
import cv2
import os
from PIL import Image
from typing import Iterable, Tuple, Union, List

def _mask_to_uint8(mask, H=None, W=None):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("mask 必须是 [H,W] 或 [1,H,W]")
    mask = (mask > 0).astype(np.uint8)
    if H is not None and W is not None and (mask.shape[0] != H or mask.shape[1] != W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

def _pil_to_rgb_uint8(img_pil):
    if not isinstance(img_pil, Image.Image):
        raise ValueError("img 必须是 PIL.Image")
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    return np.array(img_pil, dtype=np.uint8)  # [H,W,3] RGB uint8

def _overlay_mask_on_rgb(rgb, mask, color=(0,255,0), alpha=0.5):
    H, W, _ = rgb.shape
    mask = _mask_to_uint8(mask, H, W)
    overlay = rgb.copy()
    color_arr = np.zeros_like(rgb, dtype=np.uint8)
    color_arr[:] = np.array(color, dtype=np.uint8)
    idx = mask == 1
    overlay[idx] = ((1 - alpha) * overlay[idx] + alpha * color_arr[idx]).astype(np.uint8)
    return overlay

def _to_points_array(points: Union[List[Tuple[int,int]], np.ndarray, torch.Tensor]):
    if points is None:
        return np.zeros((0,2), dtype=np.int32)
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points)
    if points.size == 0:
        return np.zeros((0,2), dtype=np.int32)
    points = points.reshape(-1, 2)
    points = np.round(points).astype(np.int32)
    return points

def _to_boxes_array(boxes: Union[List[Tuple[int,int,int,int]], np.ndarray, torch.Tensor]):
    """任意输入转为 np.ndarray[N,4]，元素为整数。"""
    if boxes is None:
        return np.zeros((0,4), dtype=np.int32)
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.zeros((0,4), dtype=np.int32)
    boxes = boxes.reshape(-1, 4)
    boxes = np.round(boxes).astype(np.int32)
    return boxes

def _normalize_boxes_xyxy(boxes: np.ndarray, W: int, H: int):
    """保证 xyxy 次序且裁剪到图像范围内；过滤掉无效或退化框。"""
    if boxes.size == 0:
        return boxes
    x1 = np.minimum(boxes[:,0], boxes[:,2])
    y1 = np.minimum(boxes[:,1], boxes[:,3])
    x2 = np.maximum(boxes[:,0], boxes[:,2])
    y2 = np.maximum(boxes[:,1], boxes[:,3])
    x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
    y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
    good = (x2 > x1) & (y2 > y1)
    return np.stack([x1[good], y1[good], x2[good], y2[good]], axis=1).astype(np.int32)

def _draw_dashed_rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 10,
):
    """Draws a dashed rectangle on an image (modifies in-place)."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Top line
    for i in range(x1, x2, dash_length * 2):
        end_x = min(i + dash_length, x2)
        cv2.line(img, (i, y1), (end_x, y1), color, thickness)
    # Bottom line
    for i in range(x1, x2, dash_length * 2):
        end_x = min(i + dash_length, x2)
        cv2.line(img, (i, y2), (end_x, y2), color, thickness)
    # Left line
    for i in range(y1, y2, dash_length * 2):
        end_y = min(i + dash_length, y2)
        cv2.line(img, (x1, i), (x1, end_y), color, thickness)
    # Right line
    for i in range(y1, y2, dash_length * 2):
        end_y = min(i + dash_length, y2)
        cv2.line(img, (x2, i), (x2, end_y), color, thickness)

def _draw_annots_on_rgb(
    rgb: np.ndarray,
    pos_points: Union[List[Tuple[int,int]], np.ndarray, torch.Tensor] = None,
    neg_points: Union[List[Tuple[int,int]], np.ndarray, torch.Tensor] = None,
    boxes: Union[List[Tuple[int,int,int,int]], np.ndarray, torch.Tensor] = None,
    pos_color: Tuple[int,int,int] = (0, 255, 0),   # 醒目的绿色
    neg_color: Tuple[int,int,int] = (0, 0, 255),   # 醒目的红色
    point_size: int = 10,
    point_thickness: int = 2,
    neg_marker_type: int = cv2.MARKER_TILTED_CROSS,
    draw_point_index: bool = True,
    point_index_color: Tuple[int,int,int] = (255,255,255),
    point_index_scale: float = 0.5,
    point_index_thickness: int = 1,
    box_color: Tuple[int,int,int] = (0, 255, 255),   # 柔和的黄色
    box_thickness: int = 2,
    box_style: str = 'solid',                       # 'solid' or 'dashed'
    box_dash_length: int = 10,
    draw_box_index: bool = True,
    box_index_color: Tuple[int,int,int] = (0,0,0),
    box_index_scale: float = 0.5,
    box_index_thickness: int = 1,
):
    """在同一张图上绘制 正/负点 和 框（任选其一或全部）。"""
    H, W, _ = rgb.shape
    canvas = rgb.copy()
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # 点
    pos = _to_points_array(pos_points)
    neg = _to_points_array(neg_points)
    def _clamp_point(p):
        x = int(np.clip(p[0], 0, W-1))
        y = int(np.clip(p[1], 0, H-1))
        return x, y

    for i, p in enumerate(pos):
        x, y = _clamp_point(p)
        cv2.circle(bgr, (x, y), point_size, pos_color, point_thickness, lineType=cv2.LINE_AA)
        if draw_point_index:
            offset = point_size
            cv2.putText(bgr, f"+{i}", (x + offset + 2, y - offset - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, point_index_scale, point_index_color,
                        point_index_thickness, cv2.LINE_AA)

    for i, p in enumerate(neg):
        x, y = _clamp_point(p)
        cv2.drawMarker(bgr, (x, y), neg_color, neg_marker_type, 16 , point_thickness)
        if draw_point_index:
            offset = point_size
            cv2.putText(bgr, f"-{i}", (x + offset + 2, y - offset - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, point_index_scale, point_index_color,
                        point_index_thickness, cv2.LINE_AA)

    # 框
    boxes_np = _to_boxes_array(boxes)
    boxes_np = _normalize_boxes_xyxy(boxes_np, W, H)
    for i, (x1,y1,x2,y2) in enumerate(boxes_np):
        if box_style == 'dashed':
            _draw_dashed_rectangle(bgr, (x1,y1), (x2,y2), box_color, box_thickness, box_dash_length)
        else:
            cv2.rectangle(bgr, (x1,y1), (x2,y2), box_color, thickness=box_thickness, lineType=cv2.LINE_AA)
        
        if draw_box_index:
            label = f"#{i}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, box_index_scale, box_index_thickness)
            cv2.rectangle(bgr, (x1, max(0, y1 - th - baseline - 2)), (x1 + tw + 4, y1), (255,255,255), thickness=-1)
            cv2.putText(bgr, label, (x1 + 2, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, box_index_scale, box_index_color, box_index_thickness, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def visualize_and_save_pil(
    img_pil,
    tgt_mask,
    pred_mask,
    out_dir: str = "./vis",
    alpha: float = 0.5,
    tgt_color=(0,255,0),
    pred_color=(255,0,0),
    pos_points: Union[List[Tuple[int,int]], np.ndarray, torch.Tensor] = None,
    neg_points: Union[List[Tuple[int,int]], np.ndarray, torch.Tensor] = None,
    point_size: int = 10,
    point_thickness: int = 2,
    draw_index: bool = True,
    boxes: Union[List[Tuple[int,int,int,int]], np.ndarray, torch.Tensor] = None,
    box_color: Tuple[int,int,int] = (0, 255, 255),
    box_thickness: int = 2,
    box_style: str = 'solid', 
    draw_box_index: bool = True,
):
    """
    保存五张图（原接口不变，只是多存了一张图到磁盘）：
      1) origin.png       （原图） -> 新增
      2) overlay_gt.png   （GT 叠加）
      3) overlay_pred.png （Pred 叠加）
      4) overlay_both.png （GT+Pred）
      5) annots.png       （在“原图”上同时画点与框，谁不为 None 就画谁）
    并返回四张 PIL.Image（保持原返回值不变）。
    """
    os.makedirs(out_dir, exist_ok=True)

    rgb = _pil_to_rgb_uint8(img_pil)
    H, W, _ = rgb.shape

    vis_gt   = _overlay_mask_on_rgb(rgb,  tgt_mask,  color=tgt_color,  alpha=alpha)
    vis_pred = _overlay_mask_on_rgb(rgb,  pred_mask, color=pred_color, alpha=alpha)
    vis_both = _overlay_mask_on_rgb(rgb,  tgt_mask,  color=tgt_color,  alpha=alpha)
    vis_both = _overlay_mask_on_rgb(vis_both, pred_mask, color=pred_color, alpha=alpha)

    vis_ann = _draw_annots_on_rgb(
        rgb,
        pos_points=pos_points,
        neg_points=neg_points,
        boxes=boxes,
        pos_color=(0, 255, 0),
        neg_color=(0, 0, 255),
        point_size=point_size,
        point_thickness=point_thickness,
        draw_point_index=draw_index,
        box_color=box_color,
        box_thickness=box_thickness,
        box_style=box_style,
        draw_box_index=draw_box_index,
    )

    # NEW: 保存原图
    cv2.imwrite(os.path.join(out_dir, "origin.png"),       cv2.cvtColor(rgb,      cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(out_dir, "overlay_gt.png"),   cv2.cvtColor(vis_gt,   cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "overlay_pred.png"), cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "overlay_both.png"), cv2.cvtColor(vis_both, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "annots.png"),       cv2.cvtColor(vis_ann,  cv2.COLOR_RGB2BGR))

    return (
        Image.fromarray(vis_gt),
        Image.fromarray(vis_pred),
        Image.fromarray(vis_both),
        Image.fromarray(vis_ann),
    )