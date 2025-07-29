import torch
import torch.nn as nn
import torchvision.models as models
import time
import psutil
import os
from collections import OrderedDict
from tabulate import tabulate
import matplotlib.pyplot as plt
import textwrap

# Load model
model = models.resnet18(pretrained=False)
model.eval()

# Input tensor
input_tensor = torch.randn(1, 3, 224, 224)

# Profiling storage
profiling_data = OrderedDict()
process = psutil.Process(os.getpid())

def get_memory():
    return process.memory_info().rss / 1024 / 1024  # in MB

# Register hooks
def register_hooks(model):
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            profiling_data[name] = {'cpu_start': 0, 'cpu_time': 0, 'mem': 0}
            def pre_hook(module, input, name=name):
                profiling_data[name]['cpu_start'] = time.perf_counter()
                profiling_data[name]['mem'] = get_memory()
            def post_hook(module, input, output, name=name):
                end_time = time.perf_counter()
                profiling_data[name]['cpu_time'] = (end_time - profiling_data[name]['cpu_start']) * 1000  # ms
                profiling_data[name]['mem'] = get_memory() - profiling_data[name]['mem']
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)

register_hooks(model)

# Run forward pass
with torch.no_grad():
    _ = model(input_tensor)

# Display tabular output
print("\n==== Custom Profiler Output (Model Order) ====")
print(tabulate(
    [(k, f"{v['cpu_time']:.3f}", f"{v['mem']:.3f}") for k, v in profiling_data.items()],
    headers=["Layer", "CPU Time (ms)", "Memory Usage (MB)"],
    tablefmt="fancy_grid"
))

# Extract names and metrics for plotting
layer_names = list(profiling_data.keys())
cpu_times = [v['cpu_time'] for v in profiling_data.values()]
mem_usages = [v['mem'] for v in profiling_data.values()]

# Shorten layer names for plot
wrapped_layer_names = ['\n'.join(textwrap.wrap(name, 20)) for name in layer_names]

# Plotting CPU Time
plt.figure(figsize=(14, 8))
plt.barh(wrapped_layer_names, cpu_times, color='skyblue')
plt.xlabel("CPU Time (ms)")
plt.title("Per-Layer CPU Time Usage (Ordered by Model Execution)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Plotting Memory Usage
plt.figure(figsize=(14, 8))
plt.barh(wrapped_layer_names, mem_usages, color='lightcoral')
plt.xlabel("Memory Usage (MB)")
plt.title("Per-Layer Memory Usage (Ordered by Model Execution)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
