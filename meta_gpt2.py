import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint
from torch.optim import AdamW
import argparse
import os


# 1. Define a custom dataset (same as before)
class RandomTensorDataset(Dataset):
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids}



# 3. Define the worker function to be launched on each GPU
def run(args):

    print(f"[Rank {args.rank}] Using device: {args.device}")
    # Load tokenizer and base model (from HuggingFace)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.to(args.device)

    if args.rank == 0:
        print(model.config)


    decoders_per_rank = (model.config.n_layer + args.world_size - 1) // args.world_size
    print(f"decoders_per_rank = {decoders_per_rank}")
    split_spec = {
        f'transformer.h.{i * decoders_per_rank}': SplitPoint.BEGINNING
        for i in range(1, args.world_size)
    }


    # # Partition the model using Pipe.
    # # Here we set the devices list to span all available GPUs.
    # devices = [torch.device("cuda", i) for i in range(world_size)]
    # pipe_model = Pipe(model_seq, devices=devices, chunks=2)
    
    # # Create dataset and DataLoader (each process loads the same data)
    num_samples = 100
    seq_length = 64
    vocab_size = tokenizer.vocab_size
    x = RandomTensorDataset(num_samples, seq_length, vocab_size)
    print(x)
    chunks = 4

    pipe = pipeline(
        model,
        mb_args=example_input_microbatch,
        mb_kwargs=(),
        split_spec=split_spec,
    )
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # batch_size = 2
    # train_dataloader = DataLoader(random_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    # # Set up optimizer over the parameters of the pipeline-wrapped model
    # optimizer = AdamW(pipe_model.parameters(), lr=5e-5)
    # num_epochs = 3
    # save_steps = 10
    # global_step = 0

    # pipe_model.train()
    # for epoch in range(num_epochs):
    #     if rank == 0:
    #         print(f"Epoch {epoch+1}/{num_epochs}")
    #     for batch in train_dataloader:
    #         # For Pipe, inputs must reside on the first stage's device.
    #         input_ids = batch["input_ids"].to(devices[0])
    #         labels = batch["labels"].to(devices[0])
    #         optimizer.zero_grad()
    #         # Forward pass through the pipeline model
    #         outputs = pipe_model(input_ids, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         global_step += 1
    #         if global_step % save_steps == 0 and rank == 0:
    #             print(f"Step {global_step}: Loss = {loss.item()}")
    #             # Note: Saving a pipeline-parallel model may require extracting the underlying module.
    #             # For simplicity, we show a call to save_pretrained on the base model.
    #             base_model.save_pretrained(f"./results/step-{global_step}")

    # if rank == 0:
    #     print("Training complete!")
    # dist.destroy_process_group()

    model.to(args.device)
    x = x.to(args.device)
    y = y.to(args.device)

    model.train()
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model.config.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    schedule = ScheduleGPipe(stage, n_microbatches= 4, loss_fn=tokenwise_loss_fn)

    if rank == 0:
        schedule.step(x)
    elif rank == 1:
        losses = []
        output = schedule.step(target=y, losses=losses)
        print(f"losses: {losses}")
    dist.destroy_process_group()

# 4. Launch the training across multiple processes/GPUs.
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
