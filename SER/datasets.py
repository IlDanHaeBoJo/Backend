from torch.utils.data import Dataset
from config import Config
from typing import List, Dict
from data_utils import *
import json
import sys
import torch

# Character Vocabulary 로드
try:
    with open('char_to_id.json', 'r', encoding='utf-8') as f:
        CHAR2ID = json.load(f)
    CHAR_VOCAB = list(CHAR2ID.keys())
    ID2CHAR = {i: char for char, i in CHAR2ID.items()}
    print(f"✅ Character Vocabulary 로드 완료 ({len(CHAR_VOCAB)}개)")
except FileNotFoundError:
    print("❌ 'char_to_id.json'을 찾을 수 없습니다. 먼저 build_vocab.py를 실행하세요.")
    sys.exit(1)


class EmotionDataset(Dataset):
    def __init__(self, audio_paths: List[str], labels: List[str], processor: Wav2Vec2Processor, is_training: bool = True, config=Config):
        self.data_dir = "/data/ghdrnjs/SER/small/"
        self.audio_paths = audio_paths
        self.labels = labels
        self.processor = processor
        self.config = config
        self.encoded_labels = [self.config.LABEL2ID[label] for label in labels]
        self.is_training = is_training

        with open("script.json", "r", encoding="utf-8") as f:
            self.text_json = json.load(f)

        self.spk2id = build_speaker_mapping(audio_paths, self.data_dir)        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        emotion_label = self.encoded_labels[idx]
        file_number = extract_number_from_filename(audio_path, type="content")
        
        content_text = ""
        if file_number is not None and str(file_number) in self.text_json:
            content_text = self.text_json[str(file_number)]
        
        input_values = preprocess_audio(audio_path, self.processor, self.is_training)
        if input_values is None:
            input_values = torch.zeros(int(SAMPLE_RATE * MAX_DURATION))
        
        spk_idx_tensor = None
        if self.spk2id is not None:
            spk_str = extract_speaker_id(audio_path, self.data_dir)
            if spk_str in self.spk2id:
                spk_idx = self.spk2id[spk_str]
                spk_idx_tensor = torch.tensor(spk_idx, dtype=torch.long)

        return {
            'input_values': input_values,
            'emotion_labels': torch.tensor(emotion_label, dtype=torch.long),
            'content_text': content_text,
            'speaker_id': spk_idx_tensor
        }




def collate_fn(batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
    input_values = [item['input_values'] for item in batch]
    emotion_labels = [item['emotion_labels'] for item in batch]
    content_texts = [item['content_text'] for item in batch]
    spk_list = [item.get('speaker_id', None) for item in batch]
    
    padded_input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    
    tokenized_contents = []
    content_lengths = []
    for text in content_texts:
        ids = [CHAR2ID.get(char, CHAR2ID['<unk>']) for char in text]
        tokenized_contents.append(torch.tensor(ids, dtype=torch.long))
        content_lengths.append(len(ids))

    padded_content_labels = torch.nn.utils.rnn.pad_sequence(
        tokenized_contents, 
        batch_first=True, 
        padding_value=CHAR2ID['<pad>']
    )

    # attention_mask for audio
    attention_mask = torch.ones_like(padded_input_values, dtype=torch.long)
    for i, seq in enumerate(input_values):
        attention_mask[i, len(seq):] = 0

    if all((s is not None) and isinstance(s, torch.Tensor) for s in spk_list):
        # 각 요소가 0-dim long tensor라면 stack -> (B,)
        speaker_ids = torch.stack(spk_list)            # shape: (B,)
        speaker_ids = speaker_ids.view(-1).long()      # 보정
    else:
        speaker_ids = None

    return {
        'input_values': padded_input_values,
        'attention_mask': attention_mask,
        'labels': torch.stack(emotion_labels),
        'content_labels': padded_content_labels,
        'content_labels_lengths': torch.tensor(content_lengths, dtype=torch.long),
        'speaker_ids': speaker_ids,
    }        




