from torch.utils.data import Dataset

import torch

PAD_TOKEN = "[PAD]"
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"


def get_encoder_mask(encoder_inputs, token_to_mask):
    # Used to fill mask for (QK^T) which is 1 * seq_len * seq_len
    # int converts true and false to 1 and 0
    return (encoder_inputs != token_to_mask).unsqueeze(0).unsqueeze(0).int()


def get_causal_mask(decoder_inputs, token_to_mask, size, put_mask_to_device=None):
    # (1, seq_len)
    mask_special_tokens = (decoder_inputs != token_to_mask).unsqueeze(0).unsqueeze(0).int()
    # 1101 & 1000
    # 1101   1100
    # 1101   1110
    # 1101   1111
        
    mask = torch.tril(torch.ones((1, size, size)), diagonal=1).type_as(decoder_inputs)

    if put_mask_to_device:
        mask.to(put_mask_to_device)
    # upper 1's
    return mask_special_tokens & mask


class TranslationDataLoader(Dataset):
    def __init__(self, ds_raw, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, desired_seq_len) -> None:
        super().__init__()
        self.ds_raw = ds_raw
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.desired_seq_len = desired_seq_len

        self.src_pad_token_id = self.src_tokenizer.token_to_id(PAD_TOKEN)
        self.src_sos_token = torch.tensor([self.src_tokenizer.token_to_id(SOS_TOKEN)], dtype=torch.int64)
        self.src_eos_token = torch.tensor([self.src_tokenizer.token_to_id(EOS_TOKEN)], dtype=torch.int64)

        self.tgt_pad_token_id = self.tgt_tokenizer.token_to_id(PAD_TOKEN)
        self.tgt_sos_token = torch.tensor([self.tgt_tokenizer.token_to_id(SOS_TOKEN)], dtype=torch.int64)
        self.tgt_eos_token = torch.tensor([self.tgt_tokenizer.token_to_id(EOS_TOKEN)], dtype=torch.int64)

    '''
    ds_raw
    index, translation
    0      {source:, target:,}
    '''
    def __len__(self):
        return len(self.ds_raw)

    def __getitem__(self, index):
        # Within this fn, just ignore batch dimension
        # So, to construct mask, this size is enough
        # 1,                 seq_len,       seq_len
        # ^ size of headers
        # In batching phase and in multi-head phase
        # it will accumulate
        # Fetch one in dataset, and then return tokens
        one_record = self.ds_raw[index]
        src_text = one_record["translation"][self.src_lang]
        tgt_text = one_record["translation"][self.tgt_lang]

        src_tokens = self.src_tokenizer.encode(src_text).ids
        tgt_tokens = self.tgt_tokenizer.encode(tgt_text).ids

        # Move space for SOS and EOS
        src_padding_target = self.desired_seq_len - len(src_tokens) - 2
        # Move space for SOS only
        tgt_padding_target = self.desired_seq_len - len(tgt_tokens) - 1
        # Quick validation, if src/tgt tokens too many
        if src_padding_target < 0 or tgt_padding_target < 0:
            raise ValueError(f"Too big! {len(src_tokens)}x{len(tgt_tokens)}, {self.desired_seq_len}")

        encoder_inputs = torch.cat([
            self.src_sos_token,
            torch.tensor(src_tokens, dtype=torch.int64),
            self.src_eos_token,
            torch.tensor([self.src_pad_token_id] * src_padding_target, dtype=torch.int64)
        ])

        decoder_inputs = torch.cat([
            self.tgt_sos_token,
            torch.tensor(tgt_tokens, dtype=torch.int64),
            torch.tensor([self.tgt_pad_token_id] * tgt_padding_target, dtype=torch.int64)
        ])

        label = torch.cat([
            torch.tensor(tgt_tokens, dtype=torch.int64),
            self.tgt_eos_token,
            torch.tensor([self.tgt_pad_token_id] * tgt_padding_target, dtype=torch.int64)
        ])

        return {
            "encoder_input": encoder_inputs,
            "decoder_input": decoder_inputs,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
            "encoder_mask": get_encoder_mask(encoder_inputs, self.src_pad_token_id),
            "decoder_mask": get_causal_mask(decoder_inputs, self.tgt_pad_token_id, decoder_inputs.size(0))
        }
