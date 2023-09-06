from bitsandbytes.nn.spqr.load_quantized import get_model

model_path = '/mnt/D/data/falcon7b/'
model = get_model(model_path, model_path)