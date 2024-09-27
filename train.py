import os
import argparse

from local_dataloader import TranslationDataLoader, get_causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

from model import build_transformer

import torch
import torch.profiler
from torch import cuda

from torch.utils.data import DataLoader, random_split

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from pathlib import Path
import torchmetrics


# Huggingface datasets and tokenizers
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


from torch.utils.tensorboard import SummaryWriter


def run_inference(model, one_batch, inference_device, output_tokenizer, seq_len):

    encoder_input = one_batch["encoder_input"].to(inference_device)
    encoder_mask = one_batch["encoder_mask"].to(inference_device)
    # Since its going to be used all the time

    encoder_output = model.encode(encoder_input, encoder_mask)
    decoder_input = torch.empty(1,1).fill_(output_tokenizer.token_to_id("[SOS]")).type_as(encoder_input).to(inference_device)

    next_token = None

    eos_tgt_tokenizer_id = output_tokenizer.token_to_id("[EOS]")

    print(f"EOS ID: {eos_tgt_tokenizer_id}")

    while (not next_token) or (next_token.item() != eos_tgt_tokenizer_id):
        if decoder_input.size(1) > seq_len:
            print(f"Shape: {decoder_input.shape}, Value: {value}, Next Token: {next_token.item()}")
            break

        decoder_mask = get_causal_mask(
            decoder_input, output_tokenizer.token_to_id("[PAD]"), decoder_input.size(1),
            put_mask_to_device=inference_device
        )

        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        # B * seq_len * vocab
        logits = model.project(decoder_output[:, -1])
        # Greedy decoding
        value, next_token = torch.max(logits, dim=1)

        # Re-calculate decoder input
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1,1).fill_(next_token.item()).type_as(encoder_input).to(inference_device)
            # Important to cat on dim=1 because dim=0 is batch size
        ], dim=1)

    # decoder input = (B, vocab)
    # Typically, hugging face tokenizer strips special tokens
    return output_tokenizer.decode(decoder_input.squeeze(0).detach().cpu().numpy())

def validate_training_results(model, val_ds, output_tokenizer, inference_device, seq_len, global_step, writer, use_parallel):
    # !!!!!
    model.eval()
    counter = 0
    outputs = []
    expected_texts = []

    for one_batch in val_ds:
        # !!!!
        print(f"Inference {counter}")
        with torch.no_grad():
            outputs.append(run_inference(
                model.module if use_parallel else model,
                one_batch, inference_device, output_tokenizer, seq_len))
        expected_texts.append(one_batch["tgt_text"][0])

        # Just do 5 infernece
        if counter > 5:
            break

        counter += 1

    score_results(outputs, expected_texts, global_step, writer)


def score_results(predicted, expected, global_step, writer):
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.text.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('validation cer', cer, global_step)

    # Compute the word error rate
    metric = torchmetrics.text.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('validation wer', wer, global_step)

    # Compute the BLEU metric
    metric = torchmetrics.text.BLEUScore()
    bleu = metric(predicted, expected)
    writer.add_scalar('validation BLEU', bleu, global_step)

    writer.flush()


def check_envrionments():
    ## TODO: If bp16, check further
    if cuda.is_available():
        print(f"Cuda capactiy: {cuda.get_device_capability(torch.device('cuda'))}")
        print(f"Cuda Props: {cuda.get_device_properties(torch.device('cuda'))}")
        print("Choose cuda")
        return torch.device("cuda")
    else:
        # Warning
        print("Choose CPU")
        return torch.device("cpu")

    
def setup_parallism(config):
    # Add local rank and global rank to the config
    config["local_rank"] = int(os.environ['LOCAL_RANK'])
    config["global_rank"] = int(os.environ['RANK'])

    assert config["local_rank"] != -1, "LOCAL_RANK environment variable not set"
    assert config["global_rank"] != -1, "RANK environment variable not set"

    # Print configuration (only once per server)
    if config["local_rank"] == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"{key:>20}: {value}")

    # Setup distributed training
    init_process_group(backend='nccl')
    torch.cuda.set_device(config['local_rank'])


def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_train_tokenizer(dataset, config, lang):
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    if tokenizer_path.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    return tokenizer


def get_datasets(config):
    local_rank = int(os.environ['LOCAL_RANK'])
    if Path("dsraw.hf").exists():
        print("Load dataset from local file")
        ds_raw = load_from_disk(f"dsraw.hf")
    else:
        print(f"current working dir: {os.getcwd()}")
        print("Load dataset from huggingface")
        ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')
        ds_raw.save_to_disk(f"dsraw.hf")

    # cherry pick first 300 to test
    ds_raw = ds_raw.select(range(300))

    input_tokenizer = get_or_train_tokenizer(ds_raw, config, config["lang_src"])
    output_tokenizer = get_or_train_tokenizer(ds_raw, config, config["lang_tgt"])
    print("Got input and output tokenizers")

    # Split to 9:1 ratio
    training_set, validation_set = random_split(ds_raw, [0.9, 0.1])

    training_ds = TranslationDataLoader(training_set, input_tokenizer, output_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])
    validation_ds = TranslationDataLoader(validation_set, input_tokenizer, output_tokenizer, config['lang_src'], config['lang_tgt'], config['seq_len'])

    if not config["use_parallel"]:
        dist_sampler = None
        train_loader = DataLoader(training_ds, batch_size=config['batch_size'], shuffle=True)
    else:
        dist_sampler = DistributedSampler(training_ds, shuffle=True)
        train_loader = DataLoader(training_ds, batch_size=config['batch_size'], shuffle=False, sampler=dist_sampler)

    validation_loader = DataLoader(validation_ds, batch_size=1, shuffle=True)

    print("Datasets loaded")

    return train_loader, validation_loader, dist_sampler, input_tokenizer, output_tokenizer


def get_model(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, device):
    model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len)
    return model.to(device)


def train_model(config):
    writer = SummaryWriter("original_transformer")
    # Check environment, decide the training device
    device = check_envrionments()

    # Load dataset, tokenizers
    train_ds, val_ds, dist_sampler, in_tokenizer, out_tokenizer = get_datasets(config)

    # Load model if exists
    # !!! Why use len() instead of vocab_size? because len considered special tokens, vocab size dont
    # During embedding, we want special tokens to have their own embeddings
    model = get_model(
        in_tokenizer.get_vocab_size(with_added_tokens=True),
        out_tokenizer.get_vocab_size(with_added_tokens=True),
        config["seq_len"], config["seq_len"], device)

    if config["use_parallel"]:
        model = DistributedDataParallel(model, device_ids=[config['local_rank']])

    # Setup optimizer / loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    # !!!! to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=out_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/plain/"),
        profile_memory=True
    )

    # Setup training loop
    ### If any preload
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    model_filename = model_filename if not config["use_parallel"] else (model_filename + "_ddp")

    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    for each_epoch in range(initial_epoch, config["num_epochs"]):
        # !!!!
        torch.cuda.empty_cache()
        # Set model to train
        # !!!!
        model.train()
        # Set parallel sampler if exits
        if config["use_parallel"] and dist_sampler:
            dist_sampler.set_epoch(each_epoch)

        prof.start()
        batch_iter = tqdm(train_ds, desc=f"processing {each_epoch}", disable=(int(os.environ["LOCAL_RANK"]) != 0))
        for one_batch in batch_iter:
            # extract 4 inputs, to(device)
            encoder_input = one_batch["encoder_input"].to(device)
            decoder_input = one_batch["decoder_input"].to(device)
            encoder_mask = one_batch["encoder_mask"].to(device)
            decoder_mask = one_batch["decoder_mask"].to(device)
            # B * seq_len
            label = one_batch["label"].to(device)
            # output = model()
            # B * seq_len * len_vocab
            model_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)
            # loss = criterion()
            loss = criterion(model_output.view(-1, out_tokenizer.get_vocab_size(with_added_tokens=True)), label.view(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        prof.stop()
        # After each epoch
        # Run validation once and print out text
        validate_training_results(
            model, val_ds, out_tokenizer, device, config["seq_len"], global_step, writer, config["use_parallel"])

        # save models
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{each_epoch:02d}")
        torch.save({
            'epoch': each_epoch,
            'model_state_dict': model.module.state_dict() if config["use_parallel"] else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename if not config["use_parallel"] else model_filename + "_ddp")


def arg_parser():
    parser = argparse.ArgumentParser()
    # To update
    for key, value in get_config().items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", default=value, type=type(value), \
            help=f"Choose {key} for config"
        )

    parser.add_argument("--use-parallel", action="store_true")
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()

    updated_config = get_config()
    updated_config.update({
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "datasource": args.datasource,
        "lang_src": args.lang_src, 
        "lang_tgt": args.lang_tgt,
        "model_folder": args.model_folder,
        "model_basename": args.model_basename,
        "preload": args.preload,
        "tokenizer_file": args.tokenizer_file,
        "experiment_name": args.experiment_name,
        "use_parallel": args.use_parallel
    })

    if not Path(args.model_folder).exists():
        Path(args.model_folder).mkdir(exist_ok=True)

    if updated_config["use_parallel"]:
        setup_parallism(updated_config)

    train_model(updated_config)

    if updated_config["use_parallel"]:
        destroy_process_group()