"""
Sample from a trained Shakespeare character-level model
"""
import os
import pickle
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT

# Color codes for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

# -----------------------------------------------------------------------------
init_from = 'resume' # 'resume' from an out_dir
out_dir = 'out-shakespeare-char' # directory where model was saved
start = "\n" # starting prompt
num_samples = 10 # number of samples to draw
max_new_tokens = 200 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200 # retain only the top_k most likely tokens
seed = 1337
device = 'cuda' # single GPU only
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    raise ValueError(f"init_from must be 'resume', got {init_from}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in the Shakespeare character dataset folder
meta_path = os.path.join('data', 'shakespeare_char', 'meta.pkl')
if os.path.exists(meta_path):
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    raise FileNotFoundError(f"Meta file not found at {meta_path}. Please run data/shakespeare_char/prepare.py first.")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
for k in range(num_samples):
    # Generate initial sequence
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Extract z0 and make it optimizable
    z0 = model.latest_z.clone().detach().requires_grad_(True)
    
    # Initialize AdamW optimizer for z0
    optimizer = torch.optim.AdamW([z0], lr=0.01)
    
    # Iterative refinement using gradient-based optimization
    max_iterations = 5
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Run z0 through q-head processing path
        z_q = torch.zeros_like(z0)
        for recursion_step in range(model.config.recursion):
            # Add recursion embedding
            rec_emb = model.transformer.wre_q(torch.tensor(recursion_step, device=z0.device))
            rec_emb = rec_emb.unsqueeze(0).unsqueeze(0).expand(z0.size(0), z0.size(1), -1)
            z_q = z_q + rec_emb
            
            # Process through transformer blocks with non-causal attention
            for block in model.transformer.h:
                block.attn.causal = False
                z_q = block(z_q)
            z_q = model.transformer.ln_f(z_q)
        
        # Get q_head output
        q_head_output = model.q_head(z_q)  # (b, t, 1)
        q_logits = q_head_output.squeeze(-1)  # (b, t)
        
        # Target: all 1s (expecting high confidence)
        target = torch.ones_like(q_logits)
        
        # Binary cross entropy loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(q_logits, target)
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Decode and print optimized z0
        with torch.no_grad():
            # Get logits from optimized z0
            logits = model.lm_head(z0)
            # Sample from logits
            probs = torch.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1), num_samples=1).view(z0.size(0), z0.size(1))
            decoded_text = decode(sampled_indices[0].tolist())
            print(f"Iteration {iteration + 1}: {decoded_text}")
            print(f"Loss: {loss.item():.4f}")
    
    # Final generation from optimized z0
    with torch.no_grad():
        logits = model.lm_head(z0)
        probs = torch.softmax(logits, dim=-1)
        sampled_indices = torch.multinomial(probs.view(-1), num_samples=1).view(z0.size(0), z0.size(1))
        final_text = decode(sampled_indices[0].tolist())
        
        print(f"Final optimized result: {final_text}")
        print('---------------')
