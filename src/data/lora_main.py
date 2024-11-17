import random
from functools import partial

from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torchtune.data._collate import padded_collate_sft
from torchtune.utils import logging

from .ebay_dataset import EbayDataset

logger = logging.get_logger("DEBUG")

# class EbayDataset(Dataset):
#     def __init__(self, cfg_dataset):
#         self.data = Data(cfg_dataset)

#     def __len__(self):
#         return 5000000

#     def __getitem__(self, idx):
#         return self.data.get(idx)


class RecordIDSampler(Sampler):
    def __init__(self, min_record_id, max_record_id, shuffle=False):
        self.min_record_id = min_record_id
        self.max_record_id = max_record_id
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(iter(range(self.min_record_id, self.max_record_id + 1)))
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.max_record_id - self.min_record_id + 1

    def set_epoch(self, epoch):
        random.seed(epoch)


class RecordIDSamplerDistributed(DistributedSampler):
    def __init__(
        self, min_record_id, max_record_id, shuffle=False, num_replicas=None, rank=None
    ):
        assert (
            num_replicas is not None and rank is not None
        ), "num_replicas and rank must be provided"
        self.min_record_id = min_record_id
        self.max_record_id = max_record_id
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.total_size = self.max_record_id - self.min_record_id + 1
        self.num_samples = int(self.total_size // self.num_replicas)

    def __iter__(self):
        indices = list(iter(range(self.min_record_id, self.max_record_id + 1)))
        indices = indices[self.rank : self.total_size : self.num_replicas]

        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        random.seed(epoch)


def get_dataloader_and_sampler(
    cfg_dataset,
    shuffle=False,
    batch_size=1,
    distributed=False,
    num_replicas=None,
    rank=None,
):
    dataset = EbayDataset(cfg_dataset)
    if distributed:
        assert (
            num_replicas is not None and rank is not None
        ), "num_replicas and rank must be provided"
        sampler = RecordIDSamplerDistributed(
            min_record_id=0,
            max_record_id=3999,
            shuffle=shuffle,
            num_replicas=num_replicas,
            rank=rank,
        )
    else:
        sampler = RecordIDSampler(min_record_id=0, max_record_id=3999, shuffle=shuffle)
    return sampler, DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=partial(padded_collate_sft),
    )


def get_dataset(cfg_dataset):
    return EbayDataset(cfg_dataset)


class RecordIDProxySampler:
    def __init__(self, client):
        self.client = client

    def get(self):
        while True:
            self.client.post_done()
            status_code, record_id_dict = self.client.post_task()
            record_id = record_id_dict.get("record_id")
            logger.info(f"Received record_id: {record_id}")
            # ipdb.set_trace()
            if status_code == 410:
                return None
            return record_id


class EbayDataloader:
    def __init__(self, dataset, batch_size, sampler, collate_fn):
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.sampler = sampler
        self.dataset = dataset
        self.is_ended = False

    def __iter__(self):
        while not self.is_ended:
            record_ids = []
            texts = []
            for _ in range(self.batch_size):
                record_id = self.sampler.get()
                if record_id is None:
                    self.is_ended = True
                    break
                record_ids.append(record_id)
                record_index, text = self.dataset[record_id]
                assert (
                    record_index == record_id
                ), "Record index must match record_id, something is wrong"
                texts.append(text)
            if len(record_ids) > 0:
                yield (
                    record_ids,
                    self.collate_fn(texts)["tokens"] if self.collate_fn else texts,
                )
            else:
                break


class FakeRecordIDSampler:
    def __init__(self, client):
        self.client = client

    def get(self):
        return 4006


def get_dataloader(cfg_dataset, client, batch_size=2, is_inference=False):
    return EbayDataloader(
        dataset=EbayDataset(
            cfg_dataset, return_record_id=True, is_inference=is_inference
        ),
        batch_size=batch_size,
        sampler=RecordIDProxySampler(client),
        collate_fn=partial(padded_collate_sft) if not is_inference else None,
    )
