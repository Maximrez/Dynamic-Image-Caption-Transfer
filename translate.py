import os

import pandas
from deep_translator import GoogleTranslator

data_path = 'data'

captions_df = pandas.read_csv(os.path.join(data_path, 'captions.csv'))
# print(list(captions_df['caption']))

images = list(captions_df['caption'])

translated = GoogleTranslator(source='en', target='ru').translate_batch(list(captions_df['caption']))
print(translated)

captions_df['caption_ru'] = translated
captions_df.to_csv(os.path.join(data_path, "captions_ru.csv"), sep=',', index=False)
