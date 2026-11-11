import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'explorer'))
from runtime import Runtime
from transformers import Qwen3_5ForCausalLM,Qwen3_5TextConfig
if not torch.cuda.is_available():
    print('SKIP: CUDA device unavailable'); raise SystemExit(0)
torch.manual_seed(19)
config=Qwen3_5TextConfig(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32,linear_key_head_dim=64,linear_value_head_dim=64,linear_num_key_heads=1,linear_num_value_heads=2,layer_types=['linear_attention','full_attention'],pad_token_id=0,rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.0,'mrope_section':[4,6,6]})
model=Qwen3_5ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16).eval()
rt=object.__new__(Runtime);rt.model=model;rt.dev='cuda'
ids=torch.tensor([[1,2,3],[1,2,4],[0,1,2]],device='cuda');mask=(ids!=0).long();pos=(mask.cumsum(-1)-1).clamp(min=0)
print('Tiny hybrid model ready',flush=True)
with torch.inference_mode():
    eager=rt._decode_eager(ids,mask,pos); graph=rt._decode_graph(ids,mask,pos,8)
    for i in range(6):
        token=None if i==0 else torch.tensor([4+i,5+i,6+i],device='cuda')
        a=eager(token);b=graph(token)
        error=(a.float()-b.float()).abs().max().item()
        print('step',i,'max_abs_error',error,'argmax_equal',torch.equal(a.argmax(-1),b.argmax(-1)),flush=True)
        torch.testing.assert_close(a,b,atol=.01,rtol=.02)
states = [cell.cell_contents for cell in graph.__closure__]
assert any(isinstance(state, dict) and isinstance(state.get('graph'), torch.cuda.CUDAGraph) for state in states), 'Graph capture fell back to eager execution'
print('PASS tiny hybrid CUDA/eager parity with captured graph replay',flush=True)

