# Copyright (c) Meta Platforms, Inc. and affiliates

# Minimum effort to run this example:
# $ torchrun --nproc-per-node=4 pippy_gpt2.py

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2TokenizerFast

from hf_utils import generate_inputs_for_model, get_number_of_params


def train_and_save_model(args, gpt2, schedule):
    """Train the model with a simple loss function and save it after training."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=3e-5)

    print(f"[Rank {args.rank}] Starting Training...")
    for epoch in range(3):  # Train for 3 epochs
        # Generate full batch inputs (for simplicity, using the hf_utils helper)
        inputs = generate_inputs_for_model(
            GPT2ForSequenceClassification, gpt2, "GPT2ForSequenceClassification", args.batch_size, args.device
        )
        # Create dummy labels (assume 2 classes)
        labels = torch.randint(0, 2, (args.batch_size,), device=args.device)

        optimizer.zero_grad()
        if args.rank == 0:
            output = schedule.step(**inputs)
            loss = criterion(output.logits, labels)
            loss.backward()
            optimizer.step()
            print(f"[Rank 0] Epoch {epoch+1}, Loss: {loss.item():.4f}")
        else:
            schedule.step()

        dist.barrier()

    # Save the trained model (only Rank 0)
    if args.rank == 0:
        save_path = "trained_gpt2_pipeline.pth"
        torch.save(gpt2.state_dict(), save_path)
        print(f"[Rank 0] Model saved at {save_path}")

    dist.barrier()  # Ensure all ranks synchronize before moving to testing


def load_model_and_test(args, gpt2, tokenizer, schedule):
    """Load the trained model and run inference."""
    if args.rank == 0:
        print(f"[Rank {args.rank}] Loading Trained Model...")
        gpt2.load_state_dict(torch.load("trained_gpt2_pipeline.pth"))
        print(f"[Rank {args.rank}] Model Loaded Successfully.")

    dist.barrier()  # Synchronize all ranks before testing

    print(f"[Rank {args.rank}] Starting Inference...")
    
    # Example test prompts
    test_inputs = ["This movie was fantastic!", "I did not enjoy this film at all."]
    encoded_inputs = tokenizer(test_inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    encoded_inputs = {k: v.to(args.device) for k, v in encoded_inputs.items()}

    # Only rank 0 will provide the input; other ranks call schedule.step() with no arguments.
    output = schedule.step(**encoded_inputs) if args.rank == 0 else schedule.step()

    if args.rank == 0:
        predictions = torch.argmax(output.logits, dim=-1)  # Get predicted labels
        print(f"[Rank 0] Predictions: {predictions.cpu().numpy()} (0=Negative, 1=Positive)")

    dist.barrier()


def run(args):
    """Train GPT-2 using pipeline parallelism, then save and test the model."""
    # Configure GPT-2 model settings
    config = GPT2Config()
    config.n_embd = args.n_embd or config.n_embd
    config.n_layer = args.n_layer or config.n_layer
    config.n_head = args.n_head or config.n_head

    print(f"[Rank {args.rank}] Using device: {args.device}")

    # Create GPT-2 model and tokenizer for sequence classification
    model = GPT2ForSequenceClassification(config)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.to(args.device)

    if args.rank == 0:
        print(model.config)
        print(f"GPT-2 total number of params = {get_number_of_params(model) // 10 ** 6}M")

    # Define pipeline split specification
    decoders_per_rank = (model.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }

    # Create pipeline representation (using microbatch inputs from hf_utils helper)
    mb_inputs = generate_inputs_for_model(
        GPT2ForSequenceClassification, model, "GPT2ForSequenceClassification", args.batch_size // args.chunks, args.device
    )
    pipe = pipeline(model, mb_args=(), mb_kwargs=mb_inputs, split_spec=split_spec)
    assert pipe.num_stages == args.world_size, f"nstages = {pipe.num_stages} nranks = {args.world_size}"
    smod = pipe.get_stage_module(args.rank)
    print(f"Pipeline stage {args.rank} {get_number_of_params(smod) // 10 ** 6}M params")

    # Build pipeline stage and attach to a schedule runtime
    stage = pipe.build_stage(args.rank, device=args.device)
    schedule = ScheduleGPipe(stage, args.chunks)

    # First: Train the model and save it
    train_and_save_model(args, model, schedule)

    # Next: Load the trained model and run inference (testing)
    load_model_and_test(args, model, tokenizer, schedule)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {args.rank}] Pipeline Execution Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '29500'))
    parser.add_argument('--schedule', type=str, default="FillDrain")
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument("--chunks", type=int, default=4)
    # Note: batch_size must be divisible by chunks; here we use a small number for demonstration
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=None)
    parser.add_argument('--n_layer', type=int, default=None)
    parser.add_argument('--n_head', type=int, default=None)

    args = parser.parse_args()

    if args.cuda:
        dev_id = args.rank % torch.cuda.device_count()
        args.device = torch.device(f"cuda:{dev_id}")
        torch.cuda.set_device(dev_id)
    else:
        args.device = torch.device("cpu")

    # Initialize the distributed process group
    backend = "nccl" if args.cuda else "gloo"
    dist.init_process_group(
        backend=backend,
        rank=args.rank,
        world_size=args.world_size,
    )

    run(args)
