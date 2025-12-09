import os

import math
import random
import numpy as np
import json
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig , get_peft_model , TaskType
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
import utils.transforms as T
from utils.RRSISD_dataset import ReferDataset, SFT_DataCollator

# --------------------
# 配置
# --------------------
SEED               = 42
IMG_SIZE           = 1000
BATCH_SIZE         = 1
NUM_EPOCHS         = 2
GRAD_ACC_STEPS     = 8
LEARNING_RATE      = 5e-5
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.03
MAX_GRAD_NORM      = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(SEED)

# ======== 模型加载 ========
model_path = r""
processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
).cuda()

def get_transform(img_size=480):
    return T.Compose([T.Resize(img_size, img_size)])

dataset = ReferDataset(image_transforms=get_transform(IMG_SIZE))
collator = SFT_DataCollator_v1(processor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collator)

for p in model.parameters():
    p.requires_grad = False

# 2) 只训练 projector（这里按你结构选择 visual.merger）
for _, p in model.model.visual.merger.named_parameters():
    p.requires_grad = True

# 3) 精确枚举“语言模型里的”LoRA 目标层
suffixes = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

exact_lora_targets = []
for name, module in model.named_modules():
    if name.startswith("model.language_model.layers.") and name.endswith(suffixes):
        exact_lora_targets.append(name)

peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                # 显存够可上 32/64
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=exact_lora_targets,
    modules_to_save=["model.visual.merger"],   
    bias="none",
)

model = get_peft_model(model, peft_cfg)

# 5) 优化器参数就是：LoRA 参数 + projector（merger）参数
optim_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(optim_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_update_steps_per_epoch = math.ceil(len(loader) / GRAD_ACC_STEPS)
max_train_steps = NUM_EPOCHS * num_update_steps_per_epoch
num_warmup_steps = int(WARMUP_RATIO * max_train_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)

# --------------------
# 训练循环（bf16 autocast，无日志/无保存）
# --------------------
global_step = 0
sample_num = 100
print(len(loader))
batch_log = {"loss" : []}
version = "sftv2"
dir_key = "uid01220"
log_file = f"/desay120T/ct/dev/{dir_key}/UAV-LISA/log/train/{version}.log"
save_dir = f"/desay120T/ct/dev/{dir_key}/UAV-LISA/model/adapter/{version}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(log_file,"w",encoding='utf-8') as f:
    pass
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        image , masks , qwen_inputs , labels = batch

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            outputs = model(
                **qwen_inputs.to(model.device),
                labels=labels,
                use_cache=False
            )
            loss = outputs.loss / GRAD_ACC_STEPS
            batch_log['loss'].append(loss.item())
        loss.backward()
        if (step + 1) % GRAD_ACC_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            log_data = {
                "step" : step , 
                "global_step" : global_step ,
                "loss" : sum(batch_log['loss'][-GRAD_ACC_STEPS:]) , 
                f"last_{sample_num}_loss" : sum(batch_log['loss'][-GRAD_ACC_STEPS*sample_num:]) / sample_num
            }
            with open(log_file,"a",encoding='utf-8') as f:
                f.write(json.dumps(log_data) + "\n")
        if (global_step+1) % 100 == 0:
            print(f"save step {global_step}")
            model.save_pretrained(save_dir)
            
print("done.")
