import torch
import torchvision.models as models
import custom_profiler

# Set model to eval mode and CPU (disable fused ops)
model = models.resnet18(pretrained=False).eval().cpu()

x = torch.randn(1, 3, 224, 224)

# Start profiler
custom_profiler.start_profiler()

with torch.no_grad():
    y = model(x)

# Stop profiler
custom_profiler.stop_profiler("trace.json")
