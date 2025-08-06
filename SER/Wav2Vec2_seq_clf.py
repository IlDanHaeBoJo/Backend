import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
from adversary import ContentAdversary 

class custom_Wav2Vec2ForEmotionClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.adversary = ContentAdversary(
            input_size=config.hidden_size, 
            num_chars=config.char_vocab_size 
        )

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        content_labels=None, 
        content_labels_lengths=None,
        adv_lambda=1.0, 
    ):
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        # 1. 감정 분류
        pooled_output = torch.mean(hidden_states, dim=1)
        if hasattr(self, 'projector'):
            pooled_output = self.projector(pooled_output)
        emotion_logits = self.classifier(pooled_output)

        # 2. 내용 분류 (적대자)
        adversary_logits = self.adversary(hidden_states, adv_lambda)

        loss = None
        if labels is not None and content_labels is not None:
            # 3. 손실 계산
            loss_emotion_fct = nn.CrossEntropyLoss()
            loss_emotion = loss_emotion_fct(emotion_logits.view(-1, self.config.num_labels), labels.view(-1))
            
            # CTCLoss 사용
            loss_adversary_fct = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            
            log_probs = nn.functional.log_softmax(adversary_logits, dim=-1).transpose(0, 1)
            
            # Wav2Vec2 모델의 내장 함수를 사용하여 정확한 다운샘플링 후 길이 계산
            if attention_mask is not None:
                # 원본 오디오 길이에 대한 attention mask를 기반으로 실제 오디오 길이를 계산
                raw_input_lengths = torch.sum(attention_mask, dim=1).to(torch.long)
                # 모델의 내부 함수를 호출하여 다운샘플링된 후의 길이를 얻음
                input_lengths = self.wav2vec2._get_feat_extract_output_lengths(raw_input_lengths)
            else:
                # attention_mask가 없는 경우, 모든 시퀀스가 최대 길이라고 가정
                input_lengths = torch.full(
                    size=(hidden_states.shape[0],), fill_value=hidden_states.shape[1], device=hidden_states.device
                )
            
            loss_adversary = loss_adversary_fct(
                log_probs, 
                content_labels, 
                input_lengths, 
                content_labels_lengths
            )
            
            loss = loss_emotion + loss_adversary

        return {
            "loss": loss,
            "emotion_logits": emotion_logits,
            "adversary_logits": adversary_logits,
        }

