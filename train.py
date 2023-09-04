import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, BertTokenizerFast, ViTImageProcessor, AutoTokenizer
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import random

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

VISIONENCODERDECODERMODEL_PATH = "./visionencoderdecodermodel"
TOKENIZER_PATH = "./tokenizer"
IMAGEPROCESSOR_PATH = "./imageprocessor"

TRAIN_VISIONENCODERDECODERMODEL_PATH = "./train/visionencoderdecodermodel"
TRAIN_TOKENIZER_PATH = "./train/tokenizer"
TRAIN_IMAGEPROCESSOR_PATH = "./train/imageprocessor"


def init_model(device):
    # encoder_model = "google/vit-base-patch16-224"
    encoder_model = "google/vit-base-patch16-224-in21k"
    # encoder_model = "vit-rugpt2-image-captioning"
    # encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"

    # decoder_model = "bert-base-multilingual-cased"
    # decoder_model = "Helsinki-NLP/opus-mt-ru-en"
    # decoder_model = "vit-rugpt2-image-captioning"
    decoder_model = "sberbank-ai/rugpt3large_based_on_gpt2"

    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    # tokenizer = BertTokenizerFast.from_pretrained(decoder_model)

    image_processor = ViTImageProcessor.from_pretrained(encoder_model)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model).to(device)
    # model = VisionEncoderDecoderModel.from_pretrained("vit-rugpt2-image-captioning").to(device)

    tokenizer.pad_token = tokenizer.eos_token
    # model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.eos_token_id = tokenizer.sep_token_id
    model.config.max_new_tokens = 64  # максимальное количество слов в датасете - 64
    model.decoder.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, image_processor


def init_model_offline(device, train_path: bool = False):
    if train_path:
        model = VisionEncoderDecoderModel.from_pretrained(TRAIN_VISIONENCODERDECODERMODEL_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(TRAIN_TOKENIZER_PATH)
        image_processor = ViTImageProcessor.from_pretrained(TRAIN_IMAGEPROCESSOR_PATH)
    else:
        model = VisionEncoderDecoderModel.from_pretrained(VISIONENCODERDECODERMODEL_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        image_processor = ViTImageProcessor.from_pretrained(IMAGEPROCESSOR_PATH)

    return model, tokenizer, image_processor


def train(device, model, tokenizer, image_processor, data_loader, num_epochs=20, learning_rate=0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, amsgrad=True, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)

    loss_history = list()

    for epoch in tqdm(range(num_epochs)):
        model.train()
        sum_losses = 0
        for images, captions in data_loader:
            pixel_values = image_processor(images=images, return_tensors="pt").pixel_values.to(device)
            for image in images:
                image.close()

            labels = tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(device)

            losses = model(pixel_values=pixel_values, labels=labels).loss

            pixel_values = pixel_values.to('cpu')
            labels = labels.to('cpu')

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            sum_losses += float(losses.to('cpu'))

        avg_loss = sum_losses / len(data_loader)
        loss_history.append(avg_loss)
        # print(f'Epoch: {epoch + 1}/{num_epochs}, avg loss: {avg_loss}')

        scheduler.step()
        # torch.cuda.empty_cache()

    model.eval()
    model.save_pretrained(TRAIN_VISIONENCODERDECODERMODEL_PATH)
    tokenizer.save_pretrained(TRAIN_TOKENIZER_PATH)
    image_processor.save_pretrained(TRAIN_IMAGEPROCESSOR_PATH)

    return loss_history, model, tokenizer, image_processor


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, names, captions):
        self.data_dir = data_dir

        if len(names) != len(captions):
            raise ValueError

        self.names = names
        self.captions = captions
        self.size = (224, 224)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.names[index])
        caption = self.captions[index]

        img = Image.open(path).convert('RGB').resize(self.size)
        return img, caption

    def __len__(self):
        return len(self.names)


def init_dataloader(batch_size=5):
    def collate_fn(batch):
        return tuple(zip(*batch))

    captions_df = pd.read_csv(os.path.join('data', 'captions_ru.csv'), index_col='index')
    images = list(captions_df['image'])[:1000]
    captions = list(captions_df['caption_ru'])[:1000]

    my_dataset = MyDataset(data_dir='/home/msreznik/PythonProjects/DICT/data/images',
                           names=images,
                           captions=captions)
    # my_dataset = MyDataset(data_dir='data/images',
    #                        names=images,
    #                        captions=captions)

    return torch.utils.data.DataLoader(my_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn)


def save_offline_models():
    model, tokenizer, image_processor = init_model('cpu')

    tokenizer.save_pretrained(TOKENIZER_PATH)
    image_processor.save_pretrained(IMAGEPROCESSOR_PATH)
    model.save_pretrained(VISIONENCODERDECODERMODEL_PATH)


def test_picture(device, model, tokenizer, image_processor, file_path):
    model.eval()

    img = Image.open(file_path)
    pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)
    img.close()

    generated_ids = model.generate(pixel_values)

    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


if __name__ == '__main__':
    # save_offline_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model, tokenizer, image_processor = init_model_offline(device, train_path=False)
    # model, tokenizer, image_processor = init_model(device)
    loss_history, model, tokenizer, image_processor = train(device, model, tokenizer, image_processor, init_dataloader())
    print(loss_history)
    with open('loss_history.txt', 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(list(map(str, loss_history))))

    test_file = r'/home/msreznik/PythonProjects/DICT/data/test.jpg'
    # test_file = r'data/test.jpg'
    generated_text = test_picture(device, model, tokenizer, image_processor, test_file)
    print(f'Test image caption: {[generated_text]}')
