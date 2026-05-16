## 2024-05-14 - PyTorch DataLoader Empty Iterations
Learning: PyTorch's `DataLoader` with `drop_last=True` will quietly yield 0 batches if the dataset size is smaller than `batch_size`. This leads to `len(dataloader) == 0` and causes a divide-by-zero crash in the training loop when calculating average loss.
Action: Whenever using `DataLoader(..., drop_last=True)`, always add a protective check `if len(dataloader) == 0: raise ValueError(...)` to fail early and descriptively before entering the training loop.
