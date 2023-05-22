import os

import pandas
from tqdm import tqdm
from deep_translator import GoogleTranslator

data_path = 'data'


def translate(start_index=0, end_index=None, file_name='captions.csv'):
    captions_df = pandas.read_csv(os.path.join(data_path, file_name), index_col='index')
    if end_index is None:
        end_index = len(captions_df)
    else:
        end_index = min(len(captions_df), end_index)

    translator = GoogleTranslator(source='en', target='ru')

    for index in tqdm(range(start_index, end_index)):
        # print(f"{index - start_index}/{end_index - start_index}")
        try:
            caption_ru = translator.translate(captions_df.loc[index, 'caption'])
            captions_df.at[index, 'caption_ru'] = caption_ru
        except Exception as e:
            print(f'Error {e} with index {index}')

    captions_df.to_csv(os.path.join(data_path, f"captions_ru_{start_index}_{end_index}.csv"), sep=',')


def concat_captions(file_names, indexes, source_file_name='captions.csv', target_file_name='captions_ru.csv'):
    if len(file_names) - 1 != len(indexes):
        raise ValueError

    source_df = pandas.read_csv(os.path.join('data', source_file_name), index_col='index')
    indexes = [0] + indexes + [len(source_df)]

    for i in range(len(file_names)):
        df = pandas.read_csv(os.path.join('data', file_names[i]), index_col='index')
        for j in range(indexes[i], indexes[i + 1]):
            caption_ru = df.loc[j, 'caption_ru']
            if caption_ru is None or pandas.isna(caption_ru) or type(caption_ru) != str or len(caption_ru) == 0:
                print(j, source_df.loc[j, 'caption'])
            else:
                source_df.at[j, 'caption_ru'] = caption_ru

    source_df.to_csv(os.path.join(data_path, target_file_name), sep=',')


if __name__ == '__main__':
    # translate(400000, 800000, file_name='captions_ru_300000_400000.csv')
    concat_captions(['captions_ru_0_10000.csv',
                     'captions_ru_10000_100000.csv',
                     'captions_ru_100000_200000.csv',
                     'captions_ru_200000_230000.csv',
                     'captions_ru_230000_300000.csv',
                     'captions_ru_300000_775681.csv'], [10000, 100000, 200000, 230000, 300000])
