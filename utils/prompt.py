from transformers import AutoProcessor
import json, re
from typing import List, Tuple, Dict, Any, Optional , Union
import random
from PIL import ImageDraw , ImageFont , Image
import torch
import ast
import math

def parse_box_coords(text: str) -> Optional[Tuple[int, int, int, int]]:

    if text is None:
        return None
    _number_re = re.compile(r'-?\d+(?:\.\d+)?')
    matches = _number_re.findall(text)
    if len(matches) < 4:
        return None

    coords = []
    for s in matches[:4]:
        # 保证能兼容整数或小数形式，最后转为 int（四舍五入）
        if '.' in s:
            val = float(s)
            coords.append(int(round(val)))
        else:
            coords.append(int(s))

    return [ tuple(coords) ]  # [(x1, y1, x2, y2)]

def parse_GRPO_box_answer(text: str, first_only: bool = True):

    pattern = re.compile(r"<answer\b[^>]*>\s*(\[.*?\])\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return None

    parsed_groups = []
    for m in matches:
        try:
            data = ast.literal_eval(m)
        except Exception:
            return None

        if isinstance(data, list):
            if all(isinstance(x, (int, float)) for x in data) and len(data) == 4:
                boxes = [data]
            elif all(
                isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box)
                for box in data
            ):
                boxes = [list(box) for box in data]
            else:
                return None
        else:
            return None

        parsed_groups.append(boxes)

    if first_only:
        return parsed_groups[0]
    return parsed_groups

def parse_GRPO_point_answer(text: str, first_only: bool = True):
    pattern = re.compile(r"<answer\b[^>]*>\s*(\[.*?\])\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if not matches:
        return None

    parsed_lists = []
    for m in matches:
        try:
            data = ast.literal_eval(m)
        except Exception:
            return None

        if not isinstance(data, list):
            return None

        normalized = []
        for el in data:
            if isinstance(el, bool):
                normalized.append(int(el))
            elif isinstance(el, (int,)):
                if el in (0, 1):
                    normalized.append(el)
                else:
                    return None
            elif isinstance(el, str):
                el_stripped = el.strip()
                if el_stripped in ("0", "1"):
                    normalized.append(int(el_stripped))
                else:
                    return None
            else:
                return None

        parsed_lists.append(normalized)

    if first_only:
        return parsed_lists[0]
    return parsed_lists

def get_GRPO_box_prompt(processor, target_name,answer=None):
    system_text = (
        "You are a vision annotator working on visual labeling tasks.\n"
        "You should reason step-by-step and clearly explain your reasoning process.\n"
        "After your reasoning, output your final result strictly inside <answer>...</answer> tags."
    )

    system_dict = {
        "role": "system",
        "content": [{"type": "text", "text": system_text}],
    }
  
    user_dict = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"Locate <|object_ref_start|> {target_name} <|object_ref_end|> in the image.\n"
                    "Output its bounding box as [xmin, ymin, xmax, ymax].\n"
                    "If not visible, output [].\n"
                    "Then give the final result in <answer>...</answer>."
                ),
            },
        ],
    }

    messages = [system_dict, user_dict]

    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

    return (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(answer is None)
        ),
        messages,
    )

def get_GRPO_point_prompt(processor, target_name, points_json,answer=None):
    system_text = (
        "You are a vision annotator working on visual labeling tasks.\n"
        "You should reason step-by-step and clearly explain your reasoning process.\n"
        "After your reasoning, output your final result strictly inside <answer>...</answer> tags."
    )

    system_dict = {
        "role": "system",
        "content": [{"type": "text", "text": system_text}],
    }
  
    user_dict = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"Target class: '{target_name}'.\n"
                    f"Candidate points [x,y]: {points_json}.\n"
                    "For each point, output 1 if it lies on the target object, otherwise 0.\n"
                    "First, show your step-by-step reasoning.\n"
                    "Then, provide only the final list in the format <answer>[...]</answer>."
                ),
            },
        ],
    }
    

    messages = [system_dict, user_dict]

    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )

    return (
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=(answer is None)
        ),
        messages,
    )

def get_qwen3_point_prompt(processor, target_name, points_json, answer=None):

    system_text = (
        "You are a vision annotator. "
        "Provide only the final answer in the requested format."
    )

    system_dict = {
        "role": "system",
        "content": [{"type": "text", "text": system_text}],
    }
 
    user_dict = {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"Target class: '{target_name}'.\n"
                    f"Candidate points [x,y]: {points_json}.\n"
                    "For each point, output 1 if it lies on the target object, otherwise 0.\n"
    
                    "Provide only the final list in the format <answer>[...]</answer>."
                ),
            },
        ],
    }
    
    messages = [system_dict, user_dict]


    if answer is not None:

        answer_string = str(answer).replace(" ", "")
        
        formatted_answer_text = f"<answer>{answer_string}</answer>"
        
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": formatted_answer_text}]}
        )

    return (
        processor.apply_chat_template(
  
            messages, tokenize=False, add_generation_prompt=(answer is None)
        ),
        messages,
    )
    
def get_qwen3_box_prompt(processor, text, answer=None):


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Output the bounding box of all the following objects in the image. "
                        f"<|object_ref_start|> {text} <|object_ref_end|>"
                    ),
                },
            ],
        },
    ]

    if answer is not None:
        if len(answer) == 0 or isinstance(answer[0], (int, float)):
            answer = [answer]

        json_items = [
            f'\t{{"bbox_2d": {json.dumps(box)}, "label": "{text} "}}'
            for box in answer
        ]
        json_body = "[\n" + ",\n".join(json_items) + "\n]"
        json_text = f"```json\n{json_body}\n```"

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": json_text}],
        })

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None),
    )

    return prompt, messages

def parse_bboxes_qwen3vl(
    input_data: Union[str, List[Dict]],
    key: str = "bbox_2d"
) -> Optional[List[Tuple[int, int, int, int]]]:
   
    def _strip_code_fence(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s.strip()
    data = input_data
    if isinstance(input_data, str):
        s = _strip_code_fence(input_data)
        try:
            data = json.loads(s)
        except Exception:
            try:
                data = ast.literal_eval(s)
            except Exception as e:
                print(f"❌ 无法解析输入为 JSON 或 Python 字面量: {e}")
                return None

    if not isinstance(data, list):
        print("❌ 输入数据不是列表。")
        return None

    boxes: List[Tuple[int, int, int, int]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get(key)
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = map(int, bbox)
                boxes.append((x1, y1, x2, y2))
            except ValueError:
                continue

    return boxes if boxes else None

import json, re
from typing import List, Tuple, Dict, Any, Optional


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)  # 去开头 ```json 等
        s = re.sub(r"\s*```$", "", s)               # 去结尾 ```
    return s.strip()

def _extract_first_balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:

    start = s.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def _safe_json_loads(s: str) -> Optional[Any]:

    if not isinstance(s, str):
        return None

    s0 = s.strip()

    try:
        return json.loads(s0)
    except Exception:
        pass

    s1 = _strip_code_fences(s0)
    try:
        return json.loads(s1)
    except Exception:
        pass

    obj = _extract_first_balanced(s1, "{", "}")
    if obj is not None:
        try:
            return json.loads(obj)
        except Exception:
            pass

    arr = _extract_first_balanced(s1, "[", "]")
    if arr is not None:
        try:
            return json.loads(arr)
        except Exception:
            pass

    return None


def parse_count(text: str) -> Optional[int]:
    """解析 {"count": N}；失败返回 None。"""
    data = _safe_json_loads(text)
    if not isinstance(data, dict):
        return None
    n = data.get("count", None)
    try:
        n = int(n)
    except Exception:
        return None
    if n < 0:
        return None
    return n

def parse_boxes(text: str, W: int, H: int, max_boxes: int = 20) -> Optional[List[Tuple[int,int,int,int]]]:

    data = _safe_json_loads(text)
    if data is None:
        return None

    if isinstance(data, list):
        return []

    if not isinstance(data, dict):
        return None

    boxes = data.get("boxes", [])
    if not isinstance(boxes, list):
        return None

    out: List[Tuple[int,int,int,int]] = []
    for b in boxes[:max_boxes]:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                x1, y1, x2, y2 = [int(round(float(v))) for v in b]
            except Exception:
                continue
            # if 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H:
            out.append((x1, y1, x2, y2))
    return out

def parse_points(text: str, W: int, H: int, key: str = "points", max_points: int = 200) -> Optional[List[Tuple[int,int]]]:
    data = _safe_json_loads(text)
    if not isinstance(data, dict):
        return None
    pts = data.get(key, None)
    if not isinstance(pts, list):
        return None
    out: List[Tuple[int,int]] = []
    for p in pts[:max_points]:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            try:
                x, y = int(round(float(p[0]))), int(round(float(p[1])))
            except Exception:
                continue
            if 0 <= x < W and 0 <= y < H:
                out.append((x, y))
    return out

def parse_point_labels(text: str, expected_len: Optional[int] = None) -> Optional[List[int]]:
    data = _safe_json_loads(text)
    if not isinstance(data, dict):
        return None
    labels = data.get("labels", None)
    if not isinstance(labels, list):
        return None
    out: List[int] = []
    for v in labels:
        try:
            vv = int(round(float(v)))
        except Exception:
            return None
        if vv not in (0, 1):
            return None
        out.append(vv)
    if expected_len is not None and len(out) != expected_len:
        return None
    return out


def pack_sam_point_prompts(points: Optional[List[Tuple[int,int]]], label: int) -> Optional[Tuple[List[List[int]], List[int]]]:

    if points is None:
        return None
    try:
        coords = [[int(x), int(y)] for x, y in points]
        labs = [int(label)] * len(coords)
        return coords, labs
    except Exception:
        return None

def pack_sam_mixed_points(positive: Optional[List[Tuple[int,int]]],
                          negative: Optional[List[Tuple[int,int]]]) -> Optional[Tuple[List[List[int]], List[int]]]:
    if positive is None or negative is None:
        return None
    try:
        coords = [[int(x), int(y)] for x, y in positive] + [[int(x), int(y)] for x, y in negative]
        labels = [1]*len(positive) + [0]*len(negative)
        return coords, labels
    except Exception:
        return None


def sample_points_adaptive_grid(
    boxes: List[Tuple[int, int, int, int]],
    W: int,  # 图像总宽度 (必须提供)
    H: int,  # 图像总高度 (必须提供)
    alpha: float = 16.0,      # 缩放因子 α (超参数)
    K_max: int = 6,          # K_hmax 和 K_wmax 的最大值 (超参数)
    return_answer = False ,
    mask = None ,
) -> Tuple[List[List[int]], str, List[int]]:
    """
    在每个 box 内自适应采样点。采样密度 (Kh, Kw) 由 box 相对于图像的尺寸决定。
    
    Kh = min(K_max, floor(α * H_box / H))
    Kw = min(K_max, floor(α * W_box / W))
    
    返回:
      - points_list: [[x, y], ...]
      - points_json: '{"points": [[x, y], ...]}'
      - answer_list: [0, 1, 1, 0, ...] (根据 mask[y, x] 的值确定)
    """

    points_list = []
    answer_list = [] 

    for idx , box in enumerate(boxes):
        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        W_box = x2 - x1
        H_box = y2 - y1


        ratio_h = H_box / H
        Kh_base = math.floor(alpha * ratio_h)
        Kh = min(K_max, Kh_base)

        ratio_w = W_box / W
        Kw_base = math.floor(alpha * ratio_w)
        Kw = min(K_max, Kw_base)
        
        Kh = max(1, Kh)
        Kw = max(1, Kw)
        
        step_x = W_box / Kw  
        step_y = H_box / Kh  

        for i in range(Kw):
            for j in range(Kh):
                cx = x1 + (i + 0.5) * step_x
                cy = y1 + (j + 0.5) * step_y

                px = int(round(cx))
                py = int(round(cy))

                px = min(max(px, 0), W - 1)
                py = min(max(py, 0), H - 1)

                points_list.append([px, py])
                
                if return_answer:
                    try:
                        label = int(mask[idx][py, px])
                    except IndexError:
                        label = 0
                        
                    answer_list.append(label)

    points_json = json.dumps({"points": points_list}, ensure_ascii=False)
    if return_answer:
        return points_list, points_json, answer_list
    else:  return points_list , points_json

def sample_points_adaptive_random(
    boxes: List[Tuple[int, int, int, int]],
    W: int,  
    H: int,  
    alpha: float = 8.0,     
    K_max: int = 6,     
    return_answer: bool = False, 
    mask: Optional[Any] = None, 
) -> Union[Tuple[List[List[int]], str], Tuple[List[List[int]], str, List[int]]]:


    points_list = []
    answer_list = [] 

    if return_answer and mask is None:
        raise ValueError("当 return_answer=True 时，mask 必须被提供。")
    if W <= 0 or H <= 0:
        raise ValueError("W 和 H 必须是正整数。")

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        if x2 <= x1 or y2 <= y1:
            continue

        W_box = x2 - x1
        H_box = y2 - y1

        ratio_h = H_box / H
        Kh_base = math.floor(alpha * ratio_h)
        Kh = min(K_max, Kh_base)

        ratio_w = W_box / W
        Kw_base = math.floor(alpha * ratio_w)
        Kw = min(K_max, Kw_base)
        
        Kh = max(1, Kh)
        Kw = max(1, Kw)
        
        num_points_to_sample = Kh * Kw

        for _ in range(num_points_to_sample):
            cx = random.uniform(x1, x2)
            cy = random.uniform(y1, y2)

       
            px = int(round(cx))
            py = int(round(cy))

            # 裁剪在图像范围内 (使用总 W, H)
            px = min(max(px, 0), W - 1)
            py = min(max(py, 0), H - 1)

            points_list.append([px, py])
            
            if return_answer:
              
                try:
    
                    label = int(mask[py, px])
                except IndexError:
                    label = 0 
                    
                answer_list.append(label)

    points_json = json.dumps({"points": points_list}, ensure_ascii=False)
    
    if return_answer:
        return points_list, points_json, answer_list
    else: 
        return points_list, points_json
    
def random_sample_points_in_boxes(
    boxes: List[Tuple[int, int, int, int]],
    num_per_box: int = 5,
    seed: Optional[int] = None,
    W: Optional[int] = None,
    H: Optional[int] = None,
) -> Tuple[List[List[int]], str]:

    rng = random.Random(seed)
    points: List[List[int]] = []

    for b in boxes:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1, y1, x2, y2 = map(int, b)
        if x2 <= x1 or y2 <= y1:
            continue

    
        for _ in range(max(0, int(num_per_box))):
            x = rng.randint(x1, x2 - 1)
            y = rng.randint(y1, y2 - 1)

            # 可选：裁剪到整图范围（W/H 给出时）
            if W is not None:
                x = max(0, min(int(W) - 1, x))
            if H is not None:
                y = max(0, min(int(H) - 1, y))

            points.append([int(x), int(y)])

    points_json = json.dumps({"points": points}, separators=(",", ":"))
    return points, points_json

def draw_points_numbered(img_pil, points, color=(0,255,255), r=4, thickness=2):
    """
    在图上绘制带编号的十字小标记：
      - points: [[x,y], ...]  像素坐标（整数）
      - 编号从 1 开始，标注在点右上方，尽量不遮挡中心像素
      - 返回新图像对象（不会修改原图）
    """
    img = img_pil.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (x, y) in enumerate(points, start=1):
        x, y = int(x), int(y)
        # 十字标记
        draw.line([(x - r, y), (x + r, y)], fill=color, width=thickness)
        draw.line([(x, y - r), (x, y + r)], fill=color, width=thickness)
        # 编号（轻微偏移，避免挡住中心）
        draw.text((x + r + 3, y - r - 10), str(i), fill=color, font=font)
    return img



def _choice_rows(idx_tensor, k: int, with_replacement: bool):
    """从 N×2 的索引里取 k 行；可选有放回。"""
    n = idx_tensor.size(0)
    if k <= 0 or n == 0:
        return idx_tensor.new_zeros((0, 2))
    if with_replacement:
        sel = torch.randint(0, n, (k,))
        return idx_tensor[sel]
    if k <= n:
        sel = torch.randperm(n)[:k]
        return idx_tensor[sel]
    # 无放回但不够：全取 + 随机补（等价部分有放回）
    extra_sel = torch.randint(0, n, (k - n,))
    return torch.cat([idx_tensor, idx_tensor[extra_sel]], dim=0)

@torch.no_grad()
def sample_points_and_labels_json(
    masks_t: torch.Tensor,        # [B,H,W] 或 [B,1,H,W]，1=前景
    mean_points: int = 10,        # 期望总点数 λ
    min_points: int = 1,          # 每张图最少点数
    max_points: int = 30,         # 每张图最多点数
    pos_prob: float = 0.7,        # 正样本期望比例
    shuffle: bool = True,
):
    """
    返回：
      points_json_list:  每张图一个 JSON：'{"points":[[x,y],...]}'
      answer_json_list:  每张图一个 JSON：'{"labels":[0/1,...]}'
    说明：
      - 每张图的总点数 T ~ Poisson(λ=mean_points)，并裁剪到 [min_points, max_points]
      - 正点数 K ~ Binomial(T, pos_prob)（通过伯努利累加实现）
      - x=列, y=行；labels 与 points 一一对应
    """
    # 形状适配
    if masks_t.ndim == 4 and masks_t.size(1) == 1:
        masks = masks_t.squeeze(1)
    elif masks_t.ndim == 3:
        masks = masks_t
    else:
        raise ValueError("masks_t 需为 [B,H,W] 或 [B,1,H,W]")

    B, H, W = masks.shape
    masks = masks.to(torch.bool).cpu()

    points_json_list, answer_json_list = [], []

    for b in range(B):
        m = masks[b]                             # [H,W] bool
        pos_idx = torch.nonzero(m,  as_tuple=False)    # (Npos,2) (y,x)
        neg_idx = torch.nonzero(~m, as_tuple=False)    # (Nneg,2) (y,x)

        # 1) 采样总点数：T ~ Poisson(λ) → 裁剪到 [min_points, max_points]
        #    torch.poisson 需要 rate 张量，这里构造一个标量张量
        lam = torch.tensor(float(mean_points))
        total_points = int(torch.poisson(lam).item())
        total_points = max(min_points, min(max_points, total_points))

        if total_points <= 0:
            # 理论上不会触发（有 min_points），但兜底
            points_json_list.append('{"points":[]}')
            answer_json_list.append('{"labels":[]}')
            continue

        # 2) 正负数量：K ~ Binomial(T, pos_prob)（用伯努利累加）
        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            # 整幅图无像素（极端异常），全空
            points_json_list.append('{"points":[]}')
            answer_json_list.append('{"labels":[]}')
            continue

        k_pos = int(torch.bernoulli(torch.full((total_points,), float(pos_prob))).sum().item())
        k_neg = total_points - k_pos

        # 3) 若某类不足 → 名额让给另一类；仍不足 → 有放回补齐
        n_pos, n_neg = pos_idx.size(0), neg_idx.size(0)
        k_pos_eff = min(k_pos, n_pos)
        k_neg_eff = min(k_neg, n_neg)

        leftover = total_points - (k_pos_eff + k_neg_eff)
        if leftover > 0:
            cap_pos = n_pos - k_pos_eff
            take_pos = min(cap_pos, leftover)
            k_pos_eff += take_pos
            leftover -= take_pos

        if leftover > 0:
            cap_neg = n_neg - k_neg_eff
            take_neg = min(cap_neg, leftover)
            k_neg_eff += take_neg
            leftover -= take_neg

        need_replace_pos = (k_pos_eff < k_pos) or (leftover > 0 and n_pos > 0)
        need_replace_neg = (k_neg_eff < k_neg) or (leftover > 0 and n_neg > 0)

        # 4) 采样
        pos_yx = _choice_rows(pos_idx, k_pos_eff + (leftover if leftover>0 and n_pos>0 else 0),
                              with_replacement=need_replace_pos and (n_pos>0))
        neg_yx = _choice_rows(neg_idx, k_neg_eff + (leftover if leftover>0 and n_neg>0 else 0),
                              with_replacement=need_replace_neg and (n_neg>0))

        triplets = []
        for y, x in pos_yx.tolist():
            triplets.append((int(x), int(y), 1))
        for y, x in neg_yx.tolist():
            triplets.append((int(x), int(y), 0))

        if shuffle:
            random.shuffle(triplets)

        # 截断/补齐到 total_points
        if len(triplets) > total_points:
            triplets = triplets[:total_points]
        elif len(triplets) < total_points and len(triplets) > 0:
            while len(triplets) < total_points:
                triplets.append(random.choice(triplets))

        pts    = [[x, y] for (x, y, _) in triplets]
        labels = [lb for (_, _, lb) in triplets]

        points_json = json.dumps({"points": pts},   separators=(",", ":"))
        answer_json = json.dumps({"labels": labels}, separators=(",", ":"))
        points_json_list.append(points_json)
        answer_json_list.append(answer_json)

    return points_json_list, answer_json_list



def Threshold(model, out_ids, qwen_inputs, point_nums, one_ids=16, zero_ids=15, theata=0.9):
    """
    根据置信度阈值过滤模型生成的点分类标签。

    Args:
        model: 模型对象
        out_ids: 生成后的 token IDs 序列 (Tensor)
        qwen_inputs: 原始输入 (在此函数中未直接使用，但在完整流程中可能作为上下文)
        point_nums: 预期需要找到的点/标签数量
        one_ids: 代表 "1" (前景/Positive) 的 token ID
        zero_ids: 代表 "0" (背景/Negative) 的 token ID
        theata: 置信度阈值 (0.0 ~ 1.0)

    Returns:
        validity_labels: 一个包含 0 和 1 的列表。
                         1 表示该点置信度高 (Score >= theta)，保留。
                         0 表示该点置信度低 (Score < theta)，建议过滤。
                         列表顺序与生成的文本顺序一致（从前到后）。
    """
    
    # 1. 维度处理：确保 out_ids 是 (Batch_Size, Seq_Len)
    if out_ids.dim() == 1:
        out_ids = out_ids.unsqueeze(0)
    
    device = model.device
    out_ids = out_ids.to(device)
    qwen_inputs1 = {"input_ids" : out_ids , "pixel_values" : qwen_inputs.pixel_values.to(model.device) , "image_grid_thw" : qwen_inputs.image_grid_thw.to(model.device)}

    # 2. 重新 Forward 获取整个序列的 Logits
    #    我们需要知道模型在生成每个 token 时的“自信程度”
    with torch.no_grad():
        outputs = model(**qwen_inputs1)
        all_logits = outputs.logits # Shape: (Batch, Seq_Len, Vocab)

    # 假设 batch_size 为 1，取第一条数据
    seq_logits = all_logits[0] # (Seq_Len, Vocab)
    seq_ids = out_ids[0]       # (Seq_Len,)
    
    validity_labels = []
    found_count = 0
    seq_len = seq_ids.size(0)
    
    # 3. 从后往前遍历序列 (Reverse Traversal)
    #    从最后一个 token 开始，直到第1个 token
    #    (索引 0 的 token 没有对应的预测 logit，因为它前面没有 token)
    for i in range(seq_len - 1, 0, -1):
        # 如果已经找到了预期数量的点，提前结束
        if found_count >= point_nums:
            break
            
        token_id = seq_ids[i].item()
        
        # 4. 判断是否是我们要找的分类标签 (0 或 1)
        if token_id == one_ids or token_id == zero_ids:
            # 关键：位置 i 的 token 是由位置 i-1 的 hidden state 预测出来的
            # 所以我们要取 i-1 的 logits
            predicting_logit = seq_logits[i-1]
            
            # 计算概率分布 (Softmax)
            probs = torch.softmax(predicting_logit, dim=-1)
            
            # 获取模型实际生成的那个 token (0 或 1) 的概率分数
            score = probs[token_id].item()
            print(score)
            # 5. 根据阈值打标
            # 如果分数小于 theta，标记为 0 (过滤)
            # 否则标记为 1 (保留)
            if score < theata:
                label = 0
            else:
                label = 1
                
            validity_labels.append(label)
            found_count += 1
            
            # 调试信息 (可选)
            # print(f"Found token {token_id} at pos {i}, Score: {score:.4f} -> Label: {label}")

    # 6. 恢复顺序
    #    因为我们是从后往前找的，得到的列表是倒序的 (Point N, Point N-1 ...)
    #    需要反转回正序 (Point 1, Point 2 ...)
    validity_labels.reverse()
    
    # 7. (可选) 如果没找够点数，可能生成被截断了，可以补 0 或不做处理
    # 这里直接返回找到的部分
    
    return validity_labels