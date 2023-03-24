import os
import io
import numpy as np
from tqdm import tqdm
import supervisely as sly
from dotenv import load_dotenv

from src.clip_api import load_image

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

batch_size = 50
fill_color = [255,255,255]
src_dataset_id = 61386  # 542
# src_dataset_id = 61175  # 100
output_project_id = 18698  # 542

src_project_id = api.dataset.get_info_by_id(src_dataset_id).project_id

project_meta = sly.ProjectMeta.from_json(api.project.get_meta(src_project_id))

img_infos = api.image.get_list(src_dataset_id)
img_ids = [img_info.id for img_info in img_infos]
print(len(img_ids))


dataset_name = f"masks"
output_dataset_id = api.dataset.create(output_project_id, name=dataset_name, change_name_if_conflict=True).id

for img_infos_batch in tqdm(sly.batched(img_infos, batch_size)):
    names, metas, result_imgs = [], [], []
    img_ids_batch = [info.id for info in img_infos_batch]
    imgs_bytes = api.image.download_bytes(src_dataset_id, img_ids_batch)
    anns = api.annotation.download_json_batch(src_dataset_id, img_ids_batch)
    for ann, img_bytes, img_info in zip(anns, imgs_bytes, img_infos_batch):
        img_np = np.array(load_image(io.BytesIO(img_bytes)))
        ann = sly.Annotation.from_json(ann, project_meta)
        for i, label in enumerate(ann.labels):
            if isinstance(label.geometry, sly.Bitmap):
                mask = label.geometry.data
                bbox = label.geometry.to_bbox().to_json()['points']['exterior']
                (x1,y1),(x2,y2) = bbox
                img_patch = img_np[y1:y2+1, x1:x2+1]  # check if this correct
                img_patch[~mask] = fill_color
                result_imgs.append(img_patch)
                img_name, ext = os.path.splitext(img_info.name)
                names.append(f"{img_name}_mask_{i}{ext}")
                metas.append(img_info.meta)
    api.image.upload_nps(output_dataset_id, names, result_imgs, metas=metas)
