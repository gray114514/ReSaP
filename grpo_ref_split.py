from transformers import AutoProcessor , Qwen3VLForConditionalGeneration , GenerationConfig
import json, os, shutil, re, random, requests, io, sys, time
import torch
import torch.distributed as dist
import argparse


#训练参数配置
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
start_time = time.time() 

def init_dist():
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        print(f"进程{rank}绑定GPU {local_rank}（物理卡{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[local_rank]}）")

init_dist()

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0) 
parser.add_argument("--option",type=str,default="RRSISD")
parser.add_argument("--model_path", type=str,default="")
parser.add_argument("--data_root",type=str,default="")
parser.add_argument("--log_dir",type=str,default="")
parser.add_argument("--version_name",type=str,default="")
parser.add_argument("--log_dir",type=str,default="")
args = parser.parse_args()
option = args.option
dir_key = args.dir_key
model_path = args.model_path
DATA_ROOT = args.data_root
version_name = args.version_name 
log_dir = args.log_dir 

beta = 0.04
epsilon = 0.1
num_pre_Q = 8
Q_batch_size = 1 
train_batch_size = 1
all_steps = 1000
max_new_tokens = 768   
save_steps = 20

ds_config = {
    "train_micro_batch_size_per_gpu": Q_batch_size*train_batch_size,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

# 模型，数据集，分布式引擎 初始化
import deepspeed
from utils.prompt import parse_GRPO_point_answer , parse_GRPO_box_answer
import utils.transforms as T
from torch.utils.data import DataLoader , DistributedSampler

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
    dtype=torch.bfloat16, _attn_implementation="sdpa")
gen_model = model 
pad_token_id = processor.tokenizer.pad_token_id
generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.9, 
            num_return_sequences=num_pre_Q,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                               model_parameters=model.parameters())
gen_model = engine

def get_transform(img_size=480):
    transforms = [
        T.Resize(img_size, img_size),
    ]
    return T.Compose(transforms)
from utils.RRSISD_dataset import GRPO_DataCollator , ReferDataset
W = 1000
dataset = ReferDataset(image_transforms=get_transform(W))
collator = GRPO_DataCollator(
    processor = processor
)
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
sampler = DistributedSampler(dataset=dataset,shuffle=True,seed=42,num_replicas=world_size,rank=rank)
loader = DataLoader(
    dataset,
    batch_size=1,             # 随便设个小 batch
    num_workers=4,
    collate_fn=collator,
    sampler=sampler
)

#与ref_server 交互部分 传输data 获取batch
ref_server = "http://localhost:59875"
from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['input_ids'] = bytes_to_tensor(dd[1])
    data['attention_mask'] = bytes_to_tensor(dd[2])
    data['pixel_values'] = bytes_to_tensor(dd[3])
    data['image_grid_thw'] = bytes_to_tensor(dd[4])
    data['rewards'] = bytes_to_tensor(dd[5])
    data['loss_mask'] = bytes_to_tensor(dd[6])
    data['refs'] = bytes_to_tensor(dd[7])
    return data

def point_acc_rewards(pred,answer):
    pred = parse_GRPO_point_answer(pred)
    if pred is None:
        return 0.0
    return sum([i == j for i , j in zip(pred,answer)]) / len(answer)
from shapely.geometry import box as shapely_box
def box_iou_rewards(pred, answer):
    pred = parse_GRPO_box_answer(pred)
    if pred is None or len(pred) == 0 or len(answer) == 0:
        return 0.0
    pred = [ tuple(round( min(max(0,i),999) / 1000 * W) for i in b) for b in pred]
    ious = []
    for p_box, gt_box in zip(pred, answer):
        # 构造 shapely 的矩形几何对象
        p = shapely_box(*p_box)
        g = shapely_box(*gt_box)
        inter = p.intersection(g).area
        union = p.union(g).area
        iou = inter / union if union > 0 else 0.0
        ious.append(iou)

    return sum(ious) / len(ious)
    
def get_input_stream(loader):
    for idx , batch in enumerate(loader):
        yield batch
stream = get_input_stream(loader)

def generate_mode(num=1,rank=0):
    if rank == 0: print('enter generate mode')
    tic = time.time()
    for idx in range(num):
        pil_image, masks , qwen_inputs , answer , task_type = next(stream)
        if task_type:
            if isinstance(answer[0],list):
                answer_list = answer
            else:
                answer_list = [json.loads(answer[i])['labels'] for i in range(len(answer))]
        else:
            answer_list = [answer]

        with torch.inference_mode():
            all_ids = model.generate(**qwen_inputs.to(model.device),
                                     generation_config=generation_config)
            output_ids = all_ids[:,qwen_inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(output_ids,skip_special_tokens=True)
        if rank == 0: print(processor.batch_decode(all_ids,skip_special_tokens=True))
        rewards = []
        for i in range(len(output_text)):
            j = i // num_pre_Q 
            if task_type:
                rewards.append(point_acc_rewards(output_text[i],answer_list[j]))
            else:
                rewards.append(box_iou_rewards(output_text[i],answer_list[j]))
        batch_rewards = torch.tensor(rewards,device=model.device,dtype=torch.float32)
        if (batch_rewards.max() - batch_rewards.min()).item() <= 0.01:
            if rank == 0:
                print("***** All right or All error ++ *****")
            continue
        input_ids_list , attention_mask_list , loss_mask_list = [] , [] , []
        for i in range(len(output_text)):
            j = i // num_pre_Q
            out_ids = output_ids[i]
            all_length = out_ids.shape[0]
            out_ids = out_ids[out_ids != pad_token_id]
            pad_length = all_length - out_ids.shape[0]
            prompt_ids = qwen_inputs.input_ids[j]
            attention_mask = qwen_inputs.attention_mask[j]
            input_ids = torch.cat([torch.full((pad_length,),pad_token_id,device=prompt_ids.device,dtype=prompt_ids.dtype),prompt_ids,out_ids],dim=0)
            attention_mask = torch.cat([torch.zeros((pad_length),device=attention_mask.device,dtype=attention_mask.dtype),attention_mask,
                                        torch.ones((out_ids.shape[0]),device=attention_mask.device,dtype=attention_mask.dtype)],dim=0)
            loss_mask = torch.cat([torch.zeros((pad_length+prompt_ids.shape[0]),device=attention_mask.device,dtype=attention_mask.dtype),
                                torch.ones((out_ids.shape[0]),device=attention_mask.device,dtype=attention_mask.dtype)],dim=0)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            loss_mask_list.append(loss_mask)
        batch_input_ids = torch.stack(input_ids_list,dim=0)
        batch_attention_mask = torch.stack(attention_mask_list,dim=0)
        batch_loss_mask = torch.stack(loss_mask_list,dim=0)
        xdata = make_bytes_list([json.dumps({"plen":0}).encode(),
        tensor_to_bytes(batch_input_ids),tensor_to_bytes(batch_attention_mask),
        tensor_to_bytes(qwen_inputs.pixel_values),tensor_to_bytes(qwen_inputs.image_grid_thw),
        tensor_to_bytes(batch_rewards),tensor_to_bytes(batch_loss_mask)])
        requests.post(f"{ref_server}/upload",data=xdata)
    if rank == 0: print("exit generate mode")
    print(f"{rank}: {time.time()-tic:.3f}s")    
    
def GRPO_step(batch):
    batch_input_ids , batch_attention_mask = batch['input_ids'].to(engine.device) , batch['attention_mask'].to(engine.device)
    batch_pixel_values,batch_image_grid_thw = batch['pixel_values'].to(engine.device).repeat(num_pre_Q,1) , batch['image_grid_thw'].to(engine.device).repeat(num_pre_Q,1)
    batch_rewards , batch_loss_mask , batch_ref_logp = batch['rewards'].to(engine.device).float() , batch['loss_mask'].to(engine.device).float() , batch['refs'].to(engine.device).float()
    batch_rewards = (batch_rewards - torch.mean(batch_rewards)) /  torch.std(batch_rewards)
    pixel_value_chunck = batch_pixel_values.shape[0] // (num_pre_Q // train_batch_size)
    for idx in range(num_pre_Q // train_batch_size):
        input_ids , attention_mask = batch_input_ids[idx:idx+train_batch_size,...] , batch_attention_mask[idx:idx+train_batch_size,...]
        pixel_values,image_grid_thw = batch_pixel_values[idx*pixel_value_chunck:(idx+1)*pixel_value_chunck,...] ,batch_image_grid_thw[idx:idx+train_batch_size,...]
        rewards , loss_mask , ref_logp = batch_rewards[idx:idx+train_batch_size,...], batch_loss_mask[idx:idx+train_batch_size,...] , batch_ref_logp[idx:idx+train_batch_size,...]
        policy_logits = engine(input_ids=input_ids,attention_mask=attention_mask,
                    pixel_values=pixel_values,image_grid_thw=image_grid_thw).logits.float()
        policy_logits = policy_logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        loss_mask = loss_mask[:, 1:]
        policy_logp = torch.gather(policy_logits.log_softmax(dim=-1),dim=-1,index=input_ids.unsqueeze(-1)).squeeze(-1)
        policy_loss1 = torch.exp(policy_logp - policy_logp.detach())
        policy_loss2 = torch.clamp(policy_loss1,1-epsilon,1+epsilon)
        policy_loss = torch.min(policy_loss1,policy_loss2) * rewards.unsqueeze(-1)
        
        res1 = ref_logp - policy_logp
        kl_loss = torch.exp(res1) - res1 - 1
        
        policy_loss = (policy_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        kl_loss = (kl_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
        loss = -(policy_loss - kl_loss * beta).mean() / num_pre_Q
        engine.backward(loss)
    return loss , policy_loss.mean().item() , kl_loss.mean().item() , batch['rewards']

log_file = f"{log_dir}/{version_name}.log"
if torch.distributed.get_rank() == 0:
    with open(log_file,"w",encoding='utf-8') as f:
        pass

local_step = len(loader)
for step in range(local_step):
    batch = get_batch()
    while batch is None:
        generate_mode(rank = torch.distributed.get_rank())
        batch = get_batch()
    
    loss , policy_loss , kl_loss , rewards = GRPO_step(batch)
    engine.step()
    
    if torch.distributed.get_rank() == 0:
        with open(log_file,"a",encoding='utf-8') as f:
            f.write(json.dumps({"step":step, "used_time" : f"{time.time()-start_time:.3f}s" ,"loss" : loss.item() , "policy_loss" : policy_loss , "kl_loss" : kl_loss , "rewards" : batch['rewards'].tolist()}) + "\n") 
    
    if (step+1) % save_steps == 0:
        dist.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"***save_model {step}***")
            save_name = version_name
            state_dict = engine.module.state_dict()
            state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
            engine.module.save_pretrained(r"/desay120T/ct/dev/uid01220/UAV-LISA/model/fsdp_grpo_1", state_dict=state_dict)
        dist.barrier()