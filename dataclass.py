from torch.utils.data import Dataset
from datasets import load_dataset

class atomic_text(Dataset):
    def __init__(self, interval):
        self.texts = load_dataset("TREC-AToMiC/AToMiC-Texts-v0.2.1", split='train', )
        self.features_fields = ['page_title', 'section_title', 'context_page_description',
                                'context_section_description', ]
        self.interval = interval

    def __len__(self):
        return self.interval[1]-self.interval[0]

    def __getitem__(self, idx):
        idx = self.interval[0] + idx
        concat_text = ' '
        concat_text = concat_text.join([self.texts[idx][feature] for feature in self.features_fields])
        concat_text = concat_text + ' '.join([str(self.texts[idx]['category'])])
        concat_text = concat_text + ' '.join([str(self.texts[idx]['media'])])
        concat_text = concat_text + ' '.join([str(self.texts[idx]['hierachy'])])
        return concat_text