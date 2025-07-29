from ptflops import get_model_complexity_info
import torchvision.models as models

model = models.resnet50()
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                         print_per_layer_stat=True)
print(f"MACs: {macs}, Params: {params}")
