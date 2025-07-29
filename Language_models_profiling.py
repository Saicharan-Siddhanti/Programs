import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import os
from datetime import datetime

# Create a unique log directory with timestamp
LOGDIR = os.path.join("./llm_profiler_logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(LOGDIR, exist_ok=True)
print(f"Log directory: {LOGDIR}")

# 1. Load model on CPU
model_name = "bert-base-uncased"
model = AutoModelForCausalLM.from_pretrained(model_name).eval()  # loading model in inference mode not in training mode
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare input
text = "Hello from Sai Charan!"
inputs = tokenizer(text, return_tensors="pt")

# 3. Warm-up runs (not profiled)
for _ in range(3):
    with torch.no_grad():
        _ = model(**inputs)

# 4. Start profiling
# Dynamically set activities (avoid passing None)
activities = [ProfilerActivity.CPU]
with profile(
    activities=[
        ProfilerActivity.CPU,
        # Do not include CUDA if you only run on CPU
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(LOGDIR),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for step in range(6):  # total steps = wait(1) + warmup(1) + active(3) * repeat(1) = 5
        with torch.no_grad():
            outputs = model(**inputs)
        prof.step() #tells profile to move to the next step 

# 5. View results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(f"📈 To view in TensorBoard, run:\n")
print(f"tensorboard --logdir={LOGDIR}\nThen open: http://localhost:6006")
