from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaForSequenceClassification

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 특성 변환을 위한 Dense 레이어
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 정규화를 위한 Dropout 레이어
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(0.1)
        # 분류를 위한 출력 프로젝션 레이어
        self.out_proj = nn.Linear(config.hidden_size, 30)

    def forward(self, features, add_features, **kwargs):
        # 첫 번째 토큰의 특성 추출
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x + add_features.squeeze())
        return x

class output_class():
    def __init__(self, output=None, ner_outputs=None):
        # 모델 출력 logits 및 NER logits을 저장하기 위한 클래스
        self.logits = output
        self.ner_logits = ner_outputs

class CustomTrainer(RobertaForSequenceClassification):
    def __init__(self, MODEL_NAME="klue/roberta-large", model_config=None):
        super().__init__(model_config)
        # 만약 모델 구성이 제공되지 않으면, pretrained 모델에서 불러와 사용
        if model_config is None:
            model_config = AutoConfig.from_pretrained(MODEL_NAME)
        # 분류를 위한 라벨 수 설정
        model_config.num_labels = 30

        # pretrained RoBERTa 모델 및 토크나이저 불러오기
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # 어텐션 메커니즘을 위한 선형 레이어들
        self.word_token_Linear_key = nn.Linear(1024, 1024)
        self.word_token_Linear_value = nn.Linear(1024, 1024)
        self.cls_token_Linear_query = nn.Linear(1024, 1024)

        # 엔터티 인덱스를 위한 임베딩 레이어
        self.entity_embeddings = nn.Embedding(2, 1024)
        self.entity_embeddings.weight = nn.Parameter(torch.zeros(2, 1024))

        # 어텐션 메커니즘을 위한 드롭아웃 레이어들
        self.query_dropout = nn.Dropout(0.1)
        self.key_dropout = nn.Dropout(0.1)
        self.value_dropout = nn.Dropout(0.1)

        # classifier 및 NER classification head
        self.classifier = RobertaClassificationHead(model_config)
        self.ner_classifier = RobertaClassificationHead(model_config)
        # NER 분류기를 위한 dropout 레이어
        self.dropout_ner = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, index_ids=None, inputs_embeds=None, ner_list=None):
        # 만약 inputs_embeds가 None이라면, input_ids에 대한 RoBERTa 임베딩을 사용
        if inputs_embeds is None:
            inputs_embeds = self.roberta.embeddings.word_embeddings(input_ids)

        # entity index에 대한 임베딩 레이어
        add_embeds = self.entity_embeddings(index_ids)
        inputs_embeds = inputs_embeds + add_embeds

        # RoBERTa의 forward pass
        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # CLS 토큰 및 모든 단어 토큰의 특성 추출
        cls_token = outputs[0][:, 0, :]
        word_token = outputs[0]

        # 선형 변환 및 드롭아웃을 포함한 어텐션 메커니즘
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        word_token_idx = torch.zeros((word_token.size()[0], word_token.size()[1], 1024))
        word_token_idx[index_ids == 1] = 1
        word_token_tensor = word_token * word_token_idx.to(device)

        cls_token_query = self.cls_token_Linear_query(cls_token)
        word_token_key = self.word_token_Linear_key(word_token_tensor)
        word_token_value = self.word_token_Linear_value(word_token_tensor)

        cls_token_query = self.query_dropout(cls_token_query)
        word_token_key = self.key_dropout(word_token_key)
        word_token_value = self.value_dropout(word_token_value)

        query = cls_token_query
        key = word_token_key
        attn_scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2))
        attn_dist = torch.nn.functional.softmax(attn_scores, dim=-1)
        value = word_token_value
        weighted_avg = torch.matmul(attn_dist, value)

        # NER classifier & dropout
        logits_ner = self.dropout_ner(word_token + value)
        logits_ner = self.ner_classifier(logits_ner)

        # 주 classifier와 어텐션 가중치 평균 특성
        logits = self.classifier(cls_token.unsqueeze(1), weighted_avg.to(device))

        # 출력을 위해 logits 재구성
        outputs = (logits.view(-1, 30))
        outputs_ner = (logits_ner.view(-1, 13))
        return output_class(outputs, outputs_ner)
