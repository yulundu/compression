import random
import torch
import wandb
from mingpt.model import GPT
from torch.utils.data import Dataset, TensorDataset, DataLoader
from loguru import logger

class EightDigitAdditionDataset(Dataset):
    def __init__(self, train_size=100_000_000_000, valid_size=10000):
        self.train_size = train_size
        self.valid_pairs = [self.get_pair() for i in range(valid_size)]
        self.valid_set = set([str(pair) for pair in self.valid_pairs])
        self.idx2vocab = [str(each) for each in range(10)] + ["s", "e", "+", "="]
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}

    def __len__(self):
        return self.train_size

    def get_pair(self):
        return (random.randrange(10_000_000, 100_000_000), \
                random.randrange(10_000_000, 100_000_000))

    def get_tokens(self, pair):
        a, b, c = str(pair[0]), str(pair[1]), str(pair[0] + pair[1])
        if len(c) < 9: 
            c = '0' + c
        sequence = "s" + a + "+" + b + "=" + c + "e"
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence):
        return "".join([self.idx2vocab[item] for item in sequence])

    def get_valid(self):
        x = [self.get_tokens(pair)[:-1] for pair in self.valid_pairs]
        y = [self.get_tokens(pair)[1:]  for pair in self.valid_pairs]
        return TensorDataset(torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long))

    def __getitem__(self, index):
        pair = self.get_pair()
        while str(pair) in self.valid_set:
            pair = self.get_pair()
        tokens = self.get_tokens(pair)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

if __name__ == '__main__':
    wandb.init(project='compression_demo_dylan')
    logger.add("compression_demo.log")
    random.seed(0)
    torch.manual_seed(0)

    # dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cuda:6"
    train_data = EightDigitAdditionDataset()
    valid_data = train_data.get_valid()
    train_dataloader = DataLoader(train_data, num_workers=2, batch_size=512)
    valid_dataloader = DataLoader(valid_data, num_workers=2, batch_size=512)
    
    # init model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = 14 # vocab size: "0"-"9", "s", "e", "+", "="
    model_config.block_size = 28 # seqlen 
    model = GPT(model_config)
    model.to(device)
    model.train()

    # set optimizers
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    warmup = 10
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda s: min(s / warmup, 1))

    # training loop
    eval_steps = 5
    step = 0
    for x, y in train_dataloader:
        logits, loss = model(x.to(device), y.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schedule.step()

        if (step + 1) % eval_steps == 0:
            model.eval()
            valid_ret = []
            with torch.no_grad():
                for x_val, y_val in valid_dataloader:
                    logits_val, loss_val = model(x_val.to(device), y_val.to(device))
                    valid_ret.extend(((logits_val[:, -10:, :].argmax(dim=-1) ==  \
                                       y_val.to(device)[:, -10:]).float().mean(1) == 1).tolist())
            train_acc = ((logits[:, -10:, :].argmax(dim=-1) == y.to(device)[:, -10:]).float().mean(1) == 1).sum()/512
            output_log = {'train_loss': loss.tolist(), 'valid_loss': loss_val.tolist(), \
                       'train_acc': train_acc.tolist(), 'valid_acc': sum(valid_ret)/len(valid_data), \
                       'step': (step+1), 'lr': float(lr_schedule.get_last_lr()[0])}
            wandb.log(output_log)
            logger.info(output_log)
            model.train()
        step += 1