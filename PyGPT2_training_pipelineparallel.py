# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node 2 PyGPT2_training_pipelineparallel.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from hf_utils import get_number_of_params
from torch.profiler import profile, ProfilerActivity

def generate_inputs(model, batch_size, device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Load and tokenize dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset['train'].map(
        lambda examples: tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128),
        batched=True,
        remove_columns=["text"]
    )
    train_dataset.set_format(type='torch', columns=['input_ids'])

    # Subset adjustment
    end_idx = batch_size or len(train_dataset)
    subset = Subset(train_dataset, list(range(0, end_idx)))
    train_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    # Fetching the first batch to return
    batchdata = next(iter(train_dataloader))
    batchdata = {k: v.to(device) for k, v in batchdata.items()}

    return batchdata

def run(args):
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    gpt2.to(args.device)
    gpt2.train()
    if args.rank == 0:
        print(f"GPT-2 total number of params = {get_number_of_params(gpt2) // 10 ** 6}M")

    batch_size = args.batch_size // args.chunks
    mb_inputs = generate_inputs(model=gpt2, batch_size=batch_size, device=args.device)

    # Pipeline split spec
    decoders_per_rank = (gpt2.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline representation
    pipe = pipeline(
        gpt2,
        mb_args=(),
        mb_kwargs=mb_inputs,
        split_spec=split_spec,
    )

    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # Create schedule runtime
    stage = pipe.build_stage(
        args.rank,
        device=args.device,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    # Attach to a schedule
    schedule = ScheduleGPipe(stage, args.chunks,loss_fn=loss_fn)
    inputs = generate_inputs(model=gpt2, batch_size=args.batch_size, device=args.device)
    targets = inputs['input_ids']
    # Run
    # Run the pipeline with Input data (whole batch)
    # Input data (whole batch) will be divided into microbatches automatically
    num_iters = 11
    num_epochs = 1
    print("Starting training...")
    for epoch in range(num_epochs):
        for i in range(num_iters):
            if i==10:
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True) as prof:
                    if args.rank == 0:
                        losses = []
                        out = schedule.step(**inputs, target=targets, losses=losses)
                    else:
                        losses = []
                        out = schedule.step(target=targets, losses=losses)
                prof.export_chrome_trace("./trace/traceprofilergpt2-"+str(args.rank)+".json")
                print("Save Execution Trace")
            else:
                if args.rank == 0:
                    losses = []
                    out = schedule.step(**inputs, target=targets, losses=losses)
                else:
                    losses = []
                    out = schedule.step(target=targets, losses=losses)
                            
    dist.destroy_process_group()
    print(f"Rank {args.rank} completes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    # Note: this specific example requires: 1) a batch size that is divisible by
    # the number of chunks; 2) the division result (i.e. chunk size) must be 1,
    # otherwise padding token must be provided too (see GPT-2's forward function)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
    else:
        args.device = torch.device("cpu")

    # Init process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)