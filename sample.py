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
max_new_tokens = 500 # number of tokens generated in each sample
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
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # Iterative refinement using q_probs
            current_x = x.clone()  # Start with original conditioning sequence
            max_iterations = 5
            
            for iteration in range(max_iterations):
                # Generate sequence from current conditioning
                y, q_loss = model.generate(current_x, max_new_tokens, temperature=temperature, top_k=top_k)
                
                # Get the q_head output from the model's latest forward pass
                q_head_output = model.latest_q_head_output  # (1, t, 1)
                
                # q_head outputs logits for binary classification (correctness probability)
                q_logits = q_head_output.squeeze(0).squeeze(-1)  # (t,)
                
                # Convert logits to probabilities [0, 1] using sigmoid
                q_probs = torch.sigmoid(q_logits)  # (t,)
                
                # Find first low confidence token
                first_low_conf_idx = None
                for i, prob in enumerate(q_probs):
                    if prob < 0.5:
                        first_low_conf_idx = i
                        break
                
                # If no low confidence tokens found, we're done
                if first_low_conf_idx is None:
                    break
                
                # Truncate at first low confidence token and use as new conditioning
                if first_low_conf_idx > 0:
                    current_x = y[:, :first_low_conf_idx]  # Keep only good tokens
                else:
                    # If first token is low confidence, break to avoid infinite loop
                    break
            
            # Final generation result
            y, q_loss = model.generate(current_x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            # Get final q_head output for coloring
            q_head_output = model.latest_q_head_output  # (1, t, 1)
            q_logits = q_head_output.squeeze(0).squeeze(-1)  # (t,)
            q_probs = torch.sigmoid(q_logits)  # (t,)
            
            # Decode the final generated text
            decoded_text = decode(y[0].tolist())
            
            # Color each character based on its q_head probability
            # Higher probabilities = more confident the prediction is correct
            colored_text = ""
            for i, char in enumerate(decoded_text):
                if i < len(q_probs):
                    if q_probs[i] < 0.5:  # Low probability = likely incorrect
                        colored_text += f"{Colors.RED}{char}{Colors.RESET}"
                    else:  # High probability = likely correct
                        colored_text += f"{Colors.GREEN}{char}{Colors.RESET}"
                else:
                    colored_text += char
            
            print(colored_text)
            print(f"Overall q_loss: {q_loss.item():.4f}")
            print('---------------')
