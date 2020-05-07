from tqdm import tqdm


def find_anns(coco, img_info):
    anns = [
        ann for ann in coco['annotations'] if ann['image_id'] == img_info['id']
    ]
    return anns


def find_all_img_anns(coco):
    img_id_list = []
    anns_list = []
    for img_info in coco['images']:
        img_id_list.append(img_info['id'])
        anns_list.append([])
    for ann in tqdm(coco['annotations']):
        index = img_id_list.index(ann['image_id'])
        anns_list[index].append(ann)
    return coco['images'], anns_list


def sort_coco(coco):
    new_coco = create_coco(coco)
    for i in range(len(coco['images'])):
        img_info = coco['images'][i]
        anns = find_anns(coco, img_info)
        insert_img_anns(new_coco, img_info, anns)
    for ai in range(len(new_coco['annotations'])):
        new_coco['annotations'][ai]['id'] = ai
    return new_coco


def insert_img_anns(coco, img_info, anns):
    for ai in range(len(anns)):
        anns[ai]['image_id'] = len(coco['images'])
        anns[ai]['id'] = ai + len(coco['annotations'])
    img_info['id'] = len(coco['images'])
    coco['images'].append(img_info)
    coco['annotations'] += anns
    return coco


def create_coco(categories_coco=None):
    coco = {}
    if categories_coco is not None:
        coco['categories'] = categories_coco['categories']
    else:
        coco['categories'] = []
        for i in range(81):
            coco['categories'].append({
                'name': None,
                'supercategory': str(i),
                'id': i
            })
    coco['annotations'] = []
    coco['images'] = []
    return coco
