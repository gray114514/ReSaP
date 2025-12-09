import json, os , io
import torch 

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t.to('cpu'),buffer)
    return buffer.getvalue()

def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b))

def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

if __name__ == '__main__':
    from transformers import AutoProcessor , Qwen3VLForConditionalGeneration
    from bottle import request
    import bottle , threading , queue

    model_path = r""
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
        dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    raw_queue = queue.Queue()
    result_queue = queue.Queue() 
    
    app = bottle.Bottle()
    
    @app.route('/upload',method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        data = {'base' : json.loads(dd[0])}
        data['input_ids'] = bytes_to_tensor(dd[1])
        data['attention_mask'] = bytes_to_tensor(dd[2])
        data['pixel_values'] = bytes_to_tensor(dd[3])
        data['image_grid_thw'] = bytes_to_tensor(dd[4])
        data['rewards'] = bytes_to_tensor(dd[5])
        data['loss_mask'] = bytes_to_tensor(dd[6])
        raw_queue.put(data)
        print(f"receive {data['input_ids'].shape} {data['attention_mask'].shape} {data['pixel_values'].shape} {data['image_grid_thw'].shape} {data['rewards'].shape}")
        
    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()
    
    def run_server():
        bottle.run(app,host='0.0.0.0', port=59875, server='tornado')
    threading.Thread(target=run_server,daemon=False).start()
    
    while True:
        d = raw_queue.get()
        prompt_length = d['base']['plen']
        with torch.no_grad():
            device = ref_model.device
            input_ids , attention_mask = d['input_ids'].to(device) , d['attention_mask'].to(device)
            num_per_Q = input_ids.shape[0]
            pixel_values = d['pixel_values'].to(device).repeat(num_per_Q,1)
            image_grid_thw =  d['image_grid_thw'].to(device).repeat(num_per_Q,1)
            logits = ref_model(input_ids=input_ids,attention_mask=attention_mask,
            pixel_values=pixel_values,image_grid_thw=image_grid_thw).logits
            logits = logits[:,:-1,:]
            input_ids = input_ids[:, 1:]
            log_probs = logits.log_softmax(dim=-1)
            per_token_logps = torch.gather(log_probs,dim=-1,index=input_ids.unsqueeze(-1)).squeeze(-1)
            xdata = make_bytes_list([json.dumps(d['base']).encode(),
                                    tensor_to_bytes(d['input_ids']),
                                    tensor_to_bytes(d['attention_mask']),
                                    tensor_to_bytes(d['pixel_values']),
                                    tensor_to_bytes(d['image_grid_thw']),
                                    tensor_to_bytes(d['rewards']),
                                    tensor_to_bytes(d['loss_mask']),
                                    tensor_to_bytes(per_token_logps)])
            result_queue.put(xdata)
            