# Download 2 models: coca_ViT-L-14, convnext_large_d_320
import open_clip
cfg1 = open_clip.get_pretrained_cfg("coca_ViT-L-14", "mscoco_finetuned_laion2B-s13B-b90k")
open_clip.download_pretrained(cfg1)
cfg2 = open_clip.get_pretrained_cfg("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup")
open_clip.download_pretrained(cfg2)
