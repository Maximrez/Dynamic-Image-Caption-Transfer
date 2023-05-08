import json
import os
import shutil
from tqdm import tqdm
import pandas

pure_data_dir_path = 'pure_data'
target_data_dir_path = 'data'


def parse_flickr_data():
    with open(os.path.join(pure_data_dir_path, 'archive', 'captions.txt')) as f:
        lines = f.readlines()

    image_transfer = dict()
    images = list()
    captions = list()

    img_id = 0
    for line in tqdm(lines[1:]):
        file_name = line.split(',')[0]
        caption = ','.join(line.strip().split(',')[1:])
        caption = caption.strip('"').strip()
        # print(file_name, caption)

        if file_name not in image_transfer:
            img_id += 1

            prev_file_path = os.path.join(pure_data_dir_path, 'archive', 'Images', file_name)
            _, extension = os.path.splitext(file_name)
            new_file_name = f"flickr_{img_id}{extension}"
            image_transfer[file_name] = new_file_name

            new_file_path = os.path.join(target_data_dir_path, 'images', new_file_name)

            shutil.copyfile(prev_file_path, new_file_path)

        images.append(image_transfer[file_name])
        captions.append(caption)

    return images, captions, image_transfer


def parse_coco_data(data_name):
    with open(os.path.join(pure_data_dir_path, 'annotations', f'captions_{data_name}.json')) as f:
        data = json.load(f)

    image_transfer = dict()
    images = list()
    captions = list()
    id_to_name = dict()

    img_id = 0
    for image in tqdm(data['images']):
        file_name = image['file_name']
        coco_image_id = image['id']

        if file_name not in image_transfer:
            img_id += 1

            prev_file_path = os.path.join(pure_data_dir_path, data_name, file_name)
            _, extension = os.path.splitext(file_name)
            new_file_name = f"coco_{data_name}_{img_id}{extension}"
            image_transfer[file_name] = new_file_name
            id_to_name[coco_image_id] = new_file_name

            new_file_path = os.path.join(target_data_dir_path, 'images', new_file_name)

            shutil.copyfile(prev_file_path, new_file_path)

    for annot in data['annotations']:
        coco_image_id = annot['image_id']
        caption = annot['caption']

        images.append(id_to_name[coco_image_id])
        captions.append(caption.strip())

    return images, captions, image_transfer


if __name__ == '__main__':
    flickr_images, flickr_captions, flickr_image_transfer = parse_flickr_data()
    train2014_images, train2014_captions, train2014_image_transfer = parse_coco_data('train2014')
    val2014_images, val2014_captions, val2014_image_transfer = parse_coco_data('val2014')

    print('flickr', len(flickr_images), len(flickr_image_transfer))
    print('train2014', len(train2014_images), len(train2014_image_transfer))
    print('val2014', len(val2014_images), len(val2014_image_transfer))

    images = flickr_images + train2014_images + val2014_images
    captions = flickr_captions + train2014_captions + val2014_captions

    df = pandas.DataFrame(data={"image": images, "caption": captions})
    df.to_csv(os.path.join(target_data_dir_path, "captions.csv"), sep=',', index=False)

    with open('flickr_image_transfer.json', 'w') as f:
        json.dump(flickr_image_transfer, f)
    with open('train2014_image_transfer.json', 'w') as f:
        json.dump(train2014_image_transfer, f)
    with open('val2014_image_transfer.json', 'w') as f:
        json.dump(val2014_image_transfer, f)
