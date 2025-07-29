import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os
from datetime import datetime

#1. Load the model
model = models.densenet121(pretrained=True).eval()

#2. Input tensor 
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Warm-up 
for _ in range(3):
    _ = model(dummy_input)


LOGDIR = os.path.join("./CNN_llm_profiler_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOGDIR, exist_ok=True)
print(f"Log directory: {LOGDIR}")

# 4. Run profiler
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
    on_trace_ready=tensorboard_trace_handler(LOGDIR),
) as prof:
    for _ in range(5):
        _ = model(dummy_input)
        prof.step()

# 5. Print profiling summary
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(f"tensorboard --logdir={LOGDIR}\nThen open: http://localhost:6006 ")