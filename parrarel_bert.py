import os
import torch
import torch.distributed as dist
from transformers import BertForSequenceClassification, BertTokenizerFast
from datasets import load_dataset
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe


def setup_distributed():
    """Initialize the distributed process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  # Use a different port if needed
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank = rank, world_size = world_size)
    # dist.init_process_group(backend="nccl" )

    return rank, world_size


def load_imdb_dataset(tokenizer):
    """Load and tokenize IMDb dataset"""
    dataset = load_dataset("imdb")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    dataset = dataset.map(tokenize_function, batched=True)
    dataset = dataset.remove_columns(["text"]).rename_column("label", "labels")
    dataset.set_format("torch")

    return dataset["train"], dataset["test"]


def parallel_train_bert(rank, world_size, micro_batch_size=2, full_batch_size=8):
    """Parallel BERT training with Pipeline Parallelism"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device).eval

    # Load tokenizer and dataset
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    train_dataset, _ = load_imdb_dataset(tokenizer)

    # Sample batch of IMDb reviews as input
    full_batch_prompts = [train_dataset[i]["input_ids"] for i in range(full_batch_size)]
    mb_prompts = full_batch_prompts[:micro_batch_size]  # Micro-batch subset

    # Define pipeline split points
    layers_per_rank = model.config.num_hidden_layers // world_size
    split_spec = {
        f"bert.encoder.layer.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    # Create pipeline representation
    mb_inputs = torch.stack(mb_prompts).to(device)  # Convert microbatch to tensor
    dist.barrier()
    pipe = pipeline(model, mb_args=(mb_inputs,), split_spec = split_spec,)

    # Create pipeline stage for each rank
    stage = pipe.build_stage(rank, device=device)

    # Tokenize full batch inputs
    full_inputs = torch.stack(full_batch_prompts).to(device)

    # Attach pipeline to schedule
    num_mbs = full_batch_size // micro_batch_size  # 8 / 2 = 4 micro-batches
    schedule = ScheduleGPipe(stage, num_mbs)

    # Run Training
    if rank == 0:
        args = full_inputs
    else:
        args = None

    output = schedule.step(args)

    # Decode predictions
    if output is not None:
        predictions = torch.argmax(output[0], dim=-1)  # Get predicted labels
        print(f"Rank {rank} predictions:", predictions.cpu().numpy())

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    rank, world_size = setup_distributed()
    parallel_train_bert(rank, world_size)
