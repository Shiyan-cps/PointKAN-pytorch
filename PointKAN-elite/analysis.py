import torch
import fvcore.nn
import fvcore.common
from fvcore.nn import FlopCountAnalysis
from classification_ModelNet40.models import pointKANElite

model = pointKANElite()
model.eval()
# model = deit_tiny_patch16_224()

inputs = (torch.randn((1,3,2048)))
if torch.cuda.is_available():
    # 将模型移至 GPU
    model = model.cuda()
    # 将输入数据移至 GPU
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cuda()
    elif isinstance(inputs, (list, tuple)):
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
k = 1024.0
flops = FlopCountAnalysis(model, inputs).total()
print(f"Flops : {flops}")
flops = flops/(k**3)
print(f"Flops : {flops:.1f}G")
params = fvcore.nn.parameter_count(model)[""]
print(f"Params : {params}")
params = params/(k**2)
print(f"Params : {params:.1f}M")
