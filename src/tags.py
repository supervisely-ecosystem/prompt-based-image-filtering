import supervisely as sly

api = sly.Api()

def create_tag_meta(project_id, tag_meta: sly.TagMeta):
    project_meta_json = api.project.get_meta(id=project_id)
    project_meta = sly.ProjectMeta.from_json(data=project_meta_json)
    project_meta = project_meta.add_tag_meta(new_tag_meta=tag_meta)
    api.project.update_meta(id=project_id, meta=project_meta)
    tag_meta = get_tag_meta(project_id, name=tag_meta.name)  # we need to re-assign tag_meta
    return project_meta, tag_meta

def get_tag_meta(project_id, name) -> sly.TagMeta:
    project_meta = api.project.get_meta(project_id)
    project_meta = sly.ProjectMeta.from_json(project_meta)
    return project_meta.get_tag_meta(name)

def add_img_tag(image_id, tag_meta, value=None):
    return api.image.add_tag(image_id=image_id, tag_id=tag_meta.sly_id, value=value)
