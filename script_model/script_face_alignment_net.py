import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

face_alignment_net = torch.jit.load("./2DFAN4-cd938726ad.zip")

input = torch.rand(1, 3, 256, 256)
output = face_alignment_net(input)
print(f"{len(output)=}, {output[0].shape}")

# traced_script_module = torch.jit.script(face_alignment_net, input)
# # # traced_script_module = torch.jit.trace(face_alignment, input)
# # # traced_script_module = torch.jit.trace(face_alignment, input, strict=False)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model._save_for_lite_interpreter("./face_alignment_net_scripted.pt")

# run torchscript
torchscript = torch.jit.load("./face_alignment_net_scripted.pt")
# # print(torchscript.code)
output = torchscript(input)
print(f"{len(output)=}, {output[0].shape}")
# print(len(output), [o.shape for o in output])
# print(output)