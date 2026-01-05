import torch
import onnx
import io
from snac import SNAC

# 1. Load and Wrap
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model.eval()

class SimpleSNAC(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model
    def forward(self, codes):
        l1 = codes[:, 0:1]
        l2 = codes[:, 1:3]
        l3 = codes[:, 3:7]
        return self.decoder.decode([l1, l2, l3])

wrapper = SimpleSNAC(model)
dummy_input = torch.zeros((1, 7), dtype=torch.long)

# 2. Export to a Buffer first
f = io.BytesIO()
torch.onnx.export(
    wrapper,
    dummy_input,
    f,
    input_names=['codes'],
    output_names=['audio'],
    opset_version=17,
    export_params=True
)

# 3. Load from buffer and save using the 'onnx' library 
# This ensures everything is bundled into one protobuf
onnx_model = onnx.load_model_from_string(f.getvalue())

# CRITICAL: This saves the model as a single file, period.
onnx.save(onnx_model, "snac_decoder_24khz.onnx")

print("✅ Force Export complete. You should see ONLY snac_decoder_24khz.onnx (around 50-70MB).")