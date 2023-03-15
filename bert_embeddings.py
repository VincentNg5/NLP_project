from torch.utils.data import DataLoader
import torch

import numpy as np


def tp_bert(
    text_list,
    lm_bert_model,
    lm_bert_tokenizer,
    device,
    max_length=128,
    batch_size=50,
):
    """
    Input:
        :param text_list: Pandas.Series of strings;
        :param lm_bert_model: BERT model (e.g. model returned by:
                AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens"))
        :param lm_bert_tokenizer: Transformers tokenizer (e.g. tokenizer returned by:
                AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        :param device: torch.device object
        :param max_length: The maximum input sequence length for the model.
    Returns: 2D Numpy.Array of shape=(no. samples, m));
             BERT embedding representation of the input sequence.
    Reference: https://github.com/zhouhanxie/react-detection/blob/main/lineardetect-bert.py, or
        https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    """
    # prepare data
    data = text_list.tolist()
    dataloader = DataLoader(data, batch_size=batch_size)

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Iterate through data
    outputs = []
    for batch_text in dataloader:
        # Tokenize sentences in this batch and send to device
        encoded_input = lm_bert_tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded_input = encoded_input.to(device)

        # Compute token embeddings for this batch
        with torch.no_grad():
            model_output = lm_bert_model(**encoded_input)

        # Get sentence embeddings for this batch
        sentence_embeddings = (
            mean_pooling(model_output, encoded_input["attention_mask"]).cpu().numpy()
        )

        outputs.append(sentence_embeddings)

    # Aggregate outputs into single numpy array
    outputs = np.vstack(outputs)

    assert outputs.shape[1] == 768

    return outputs
