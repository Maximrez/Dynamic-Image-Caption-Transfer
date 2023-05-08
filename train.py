import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, BertTokenizerFast, ViTImageProcessor
from tqdm import tqdm
import os
import pandas as pd


def init_model(device):
    encoder_model = "google/vit-base-patch16-224"
    # encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"

    decoder_model = "bert-base-uncased"

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model).to(device)

    # tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    tokenizer = BertTokenizerFast.from_pretrained(decoder_model)

    image_processor = ViTImageProcessor.from_pretrained(encoder_model)

    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, image_processor


def train(model, tokenizer, image_processor, data_loader, num_epochs=100, learning_rate=0.001):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, amsgrad=True, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    loss_history = list()

    for epoch in tqdm(range(num_epochs)):
        if epoch != 0:
            scheduler.step()
        model.train()
        sum_losses = 0
        for images, captions in data_loader:
            pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)

            labels = tokenizer(captions, return_tensors="pt").input_ids
            labels.to(device)

            losses = sum(model(pixel_values=pixel_values, labels=labels).loss)
            # print(losses)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            sum_losses += losses

        avg_loss = sum_losses / len(data_loader)
        loss_history.append(avg_loss)
        print(f'Epoch: {epoch + 1}/{num_epochs}, avg loss: {avg_loss}')

    return loss_history


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, names, captions, transforms=None):
        self.data_dir = data_dir

        if len(names) != len(captions):
            raise ValueError

        self.names = names
        self.captions = captions
        self.transforms = transforms

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.names[index])
        caption = self.captions[index]

        img = Image.open(path)
        return img, caption

    def __len__(self):
        return len(self.names)


def init_dataloader(batch_size=10):
    def collate_fn(batch):
        return tuple(zip(*batch))

    captions_df = pd.read_csv(os.path.join('data', 'captions.csv'))
    my_dataset = MyDataset(data_dir='data\\images',
                           names=list(captions_df['image']),
                           captions=list(captions_df['caption']))

    return torch.utils.data.DataLoader(my_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=collate_fn)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train(*init_model(device), init_dataloader())
