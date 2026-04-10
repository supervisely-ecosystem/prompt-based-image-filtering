"""Pre-download model weights to the exact cache paths expected by clip_api.py."""

import os
import urllib.request

MODELS = [
    {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "path": "/root/.cache/clip/ViT-L-14.pt",
    },
    {
        "url": "https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/main/open_clip_pytorch_model.bin",
        "path": "/root/.cache/huggingface/hub/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/blobs/4572137af44b2e26f01f638337a59688ec289e9363e15c08dde16640afb86988",
    },
]

for model in MODELS:
    dst = model["path"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Downloading {model['url']} -> {dst}")
    urllib.request.urlretrieve(model["url"], dst)
    print(f"Saved: {dst}")
