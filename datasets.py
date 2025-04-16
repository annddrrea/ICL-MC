import torch
import numpy as np
from utils import stationary_distribution, approx_stationary_distribution
from torch.utils.data import Dataset
from random import choices, seed, Random, sample, random
import pickle
from hashlib import sha3_256
import random
import os

        

def split(tensor):
  ret = int.from_bytes(sha3_256(pickle.dumps(tensor.flatten().tolist())).digest())
  return ret

class ngrams(Dataset):

    """
    Dataset for the in context markov learning. Each example is a series of outputs from a markov chain
    """

    def __init__(self, split:str, n:int, length = 101, num_symbols = 2, size = 1000, last_token_only = False, device = 'cpu', offline=False, cache_dir='train_transition', sequential=False):
        self.length = length
        self.num_symbols = num_symbols
        self.split = split
        self.offline = offline
        assert split in ['train', 'test'], f"split must be 'train' or 'test' not {split}"
        self.size = size
        self.last_token_only = last_token_only
        self.device = device
        self.n = n - 1
        self.sequential = sequential
        self.transition_matrix_gen = torch.distributions.dirichlet.Dirichlet(torch.ones((num_symbols**(n-1),num_symbols), device = device)).sample # creates a zero tensor temp with shape (..., num_contexts, num_symbols)
        
        # Compute powers of num_symbols
        self.powers = self.num_symbols ** torch.arange(self.n - 1, -1, -1, device=device, dtype=torch.long)  # Shape: (n,)

        self.conv = torch.tensor([num_symbols ** k for k in range(self.n)])


        self.cache_dir = f'{split}_transition_size={size}'
        self.cache_file = f"{self.cache_dir}/transitions_n{n}_s{num_symbols}.pt"
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load or generate transition matrices
        if os.path.exists(self.cache_file):
            print(f"Loading cached transition matrices from {self.cache_file}")
            self.transition_matrices = torch.load(self.cache_file)
        else:
            print(f"Generating and caching transition matrices to {self.cache_file}")
            self.transition_matrices = self._generate_transition_matrices(size)
            torch.save(self.transition_matrices, self.cache_file)

    def _generate_transition_matrices(self, size):
        """Generate all transition matrices upfront"""
        print("Generating transition matrices...")
        matrices = []
        batch_size = 1000  # Process in batches to manage memory
        
        for i in range(0, size, batch_size):
            current_batch = min(batch_size, size - i)
            matrices.append(self.transition_matrix_gen([current_batch]))
        
        return torch.cat(matrices, dim=0)

    def __len__(self):
        return self.size

    def get_vocab_size(self):
        return self.num_symbols

    def get_block_size(self):
        # the length of the sequence that will feed into transformer
        return self.length
    
    def stationary_distribution(self, transition_matrices):
        if self.n > 1:
            temp = torch.zeros(transition_matrices[...,0,0].size()+torch.Size((self.num_symbols**(self.n),self.num_symbols**(self.n))), device = self.device) # zero tensor with shape = (*batch_dims, num_symbols**n, num_symbols**n)
            for i in range(self.num_symbols ** self.n):
                for j in range(self.num_symbols):
                    converted = int((i %self.num_symbols**(self.n - 1)) * self.num_symbols + j)
                    temp[..., i, converted] = transition_matrices[:, i,j]
            return stationary_distribution(temp)
        else:
            return stationary_distribution(transition_matrices)

    def __getitem__(self, idx):
        if self.sequential:
            transition_matrix = self.transition_matrix_gen([1])
        else:
            transition_matrix = self.transition_matrices[idx]
        # stationary = self.stationary_distribution(transition_matrix)
        output = torch.zeros(self.length, dtype=torch.long, device=self.device)

        # Sample one initial state
        # initial_idx = torch.multinomial(stationary.squeeze(0), 1).item()
        # output[:self.n] = self.single_symbol_convert(torch.tensor([initial_idx]))[0]
        output[:self.n] = torch.randint(0, self.num_symbols, (self.n,), device=self.device)


        # No batch offset needed
        flat_transition = transition_matrix.flatten(0, 1)

        for ind in range(self.n, self.length):
            if self.n == 1:
                context = output[ind - 1].item()
            else:
                context = self.multi_symbol_convert(output[ind - self.n:ind].unsqueeze(0)).item()

            probs = flat_transition[context]
            if probs.ndim == 0:
                probs = probs.unsqueeze(0)
            output[ind] = torch.multinomial(probs, 1).item()

        x = output[:-1]


        if self.split == 'train':
            y = output[1:].clone()
            if (self.last_token_only):
                y[:-1] = -1
        elif self.split == 'test':
            y = transition_matrix, torch.tensor(output[1:], dtype=torch.long)
        else:
            raise ValueError("Invalid split, this should not be possible unless split was changed after initialization")
        return x, y


    def __getitems__(self, indices):
        if self.sequential:
            transition_matrices = self.transition_matrix_gen([len(indices)])
        else:
            transition_matrices = self.transition_matrices[indices]
        # stationary_distributions = self.stationary_distribution(transition_matrices)

        #generate sequence
        output = torch.zeros((len(transition_matrices), self.length), dtype=torch.long, device = self.device)
        # output[:, :self.n] =self.single_symbol_convert(torch.multinomial(stationary_distributions, 1).squeeze())
        output[:, :self.n] = torch.randint(0, self.num_symbols, (len(transition_matrices), self.n), device=self.device)
        cons = torch.arange(len(transition_matrices), device = self.device) * self.num_symbols ** self.n
        for ind in range(self.n, self.length):
            if self.n == 1:
                temp = transition_matrices.flatten(end_dim=1)[cons + output[:,ind-1]]
            else:
                temp = transition_matrices.flatten(end_dim=1)[cons + self.multi_symbol_convert(output[:,ind-self.n:ind])]
            
            output[:,ind] = torch.multinomial(temp,1).squeeze()

        x = output[:, :-1]

        if self.split == 'train':
            y = output[:, 1:]
            if (self.last_token_only):
                y = torch.ones_like(y) * -1
                y[:, -1] = output[:, -1]
        elif self.split == 'test':
            y = zip(transition_matrices, output[:, 1:])
        else:
            raise ValueError("Invalid split, this should not be possible unless split was changed after initialization")
        return tuple(zip(x,y))
    
    def multi_symbol_convert(self, l):
        # assert len(l) == self.n-1
        # print((l, self.conv))
        # exit()
        # print()
        return (l * self.conv).sum(axis=1)

    # def single_symbol_convert(self, m):
    #     if self.n == 1:
    #         return m.unsqueeze(-1)
    #     out = torch.zeros((m.size(0), self.n), dtype=torch.long)
    
    #     for i in range(self.n - 1, -1, -1):
    #         out[:, i] = m % self.num_symbols
    #         m = m // self.num_symbols
    #     return out

    def single_symbol_convert(self, m):
        if self.n == 1:
            return m.unsqueeze(-1)
        # Expand m to match the shape for broadcasting
        m_expanded = m.unsqueeze(1)  # Shape: (batch_size, 1)
        # Compute the digits in base num_symbols
        out = (m_expanded // self.powers) % self.num_symbols  # Shape: (batch_size, n)
        return out

import warnings
class doubly_stochastic(ngrams):
    def __init__(self, split, length = 101, num_symbols = 2, size = 1000, last_token_only = False, device = 'cpu'):
        super().__init__(split, 2, length, size=size, last_token_only=last_token_only, device=device)
        if num_symbols > 5:
            warnings.warn("uniform doubly stochastic with more than 5 symbols is impractically slow. will use non-uniform sampling")
        
        self.dirichlet_markov_ensemble = torch.distributions.dirichlet.Dirichlet(torch.ones((num_symbols, num_symbols), device = device)).sample

    def stationary_distribution(self, transition_matrices):
        return torch.ones((transition_matrices.size(0), self.num_symbols), device = self.device) / self.num_symbols

    # num is number of output transition matrices
    # n (only used when num_symbols < 6) is number of attempts to make per loop (if num_symbols is 5 or num is large, larger n is faster but uses more memory)
    def transition_matrix_gen(self, sample_shape = torch.Size(), n = 100):
        k = self.num_symbols
        return_matrix = sample_shape == torch.Size()
        if len(sample_shape) > 1:
            warnings.warn(f"sample_shape should have length zero or one, not {len(sample_shape)}, ignoring all but the first element")
        num = sample_shape[0] if not return_matrix else 1

        out = torch.zeros(size=[num, k, k], device=self.device)
        num_created = 0
        if k < 6:
            m = torch.zeros((n, k, k), device=self.device)
            while num_created < num:
                m[:, :k-1, :k-1] = torch.rand(size=(n, k-1, k-1), device=m.device)
                m = m[(m.sum(axis=1)<1).all(dim=1)]
                m = m[(m.sum(axis=2)<1).all(dim=1)]
                m[:, k-1, :k-1] = 1 - m[:, :k-1, :k-1].sum(axis=1)
                m[:, :k, k-1] = 1 - m[:, :, :k-1].sum(axis=2)
                m = m[m[:,-1,-1]>0]
                out[num_created:min(num_created+n, len(m))] = m[:min(num_created+n, len(m))]
                num_created += len(m)
        else:
            next_out = self.dirichlet_markov_ensemble([num])
            out = torch.ones_like(next_out)
    
            # Iteratively adjust rows and columns
            while not torch.allclose(out, next_out):
                out = next_out
                # Normalize columns
                next_out = out / out.sum(axis=0, keepdims=True)
                # Normalize rows
                next_out = next_out / next_out.sum(axis=1, keepdims=True)
            out = next_out

        if return_matrix:
            return out[0]
        else:
            return out

class unigram(ngrams):
    def __init__(self, split, length = 101, num_symbols = 2, size = 1000, last_token_only = False, device = 'cpu'):
        # calls the constructor of the parent class ngrams
        super().__init__(split, 2, length, num_symbols, size=size, last_token_only=last_token_only, device=device)
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones((num_symbols), device = device)).sample

    def stationary_distribution(self, transition_matrices):
        return transition_matrices[..., 0, :]

    def transition_matrix_gen(self, sample_shape = torch.Size()):
        return_matrix = sample_shape == torch.Size()
        if len(sample_shape) > 1:
            warnings.warn(f"sample_shape should have length zero or one, not {len(sample_shape)}, ignoring all but the first element")
        num = sample_shape[0] if not return_matrix else 1

        # generate random transition probabilities  
        probs = self.dirichlet(sample_shape)

        return probs.unsqueeze(-2).expand(sample_shape+torch.Size([self.num_symbols, self.num_symbols]))
        

class mixture_sequential(Dataset):
    def __init__(self, dblist : list, plist : list):
        self.dblist = dblist
        self.plist = plist

    def __len__(self):
        return self.dblist[0].__len__()
    
    def __getitem__(self, idx):
        # randomly select a db according to plist
        db = choices(self.dblist, self.plist)[0]
        self.n = db.n
        return db.__getitem__(idx)

class mixture_parallel(Dataset):
    def __init__(self, dblist: list, plist: list, batch_size: int = 1024, num_workers: int = 8):
        """
        Args:
            dblist: List of ngram datasets
            plist: List of probabilities for each dataset
            batch_size: Size of batches for parallel generation
        """
        self.dblist = dblist
        self.plist = plist
        self.batch_size = batch_size
        
        # Calculate how many sequences to generate from each dataset
        self.total_size = min(db.__len__() for db in dblist)
        self.sequence_counts = [int(self.total_size * p) for p in plist]
        
        # Adjust counts to ensure they sum to total_size
        diff = self.total_size - sum(self.sequence_counts)
        if diff > 0:
            self.sequence_counts[0] += diff  # Add remaining to first dataset
        
        # Pre-generate indices for each dataset
        self.indices = []
        for i, count in enumerate(self.sequence_counts):
            self.indices.extend([(i, j) for j in range(count)])
        
        # Shuffle the indices
        random.shuffle(self.indices)
        
        # Pre-generate sequences in batches
        self.sequences = []
        for i in range(0, len(self.indices), batch_size):
            batch_indices = self.indices[i:i+batch_size]
            batch_sequences = self._generate_batch(batch_indices)
            self.sequences.extend(batch_sequences)
        
        # Shuffle the sequences
        random.shuffle(self.sequences)

        # Create a pool of workers for parallel processing
        self.sequences = []
        for i in range(0, len(self.indices), batch_size):
            batch_indices = self.indices[i:i+batch_size]
            batch_sequences = self._generate_batch(batch_indices)
            self.sequences.extend(batch_sequences)

    def _generate_batch(self, batch_indices):
        """Generate a batch of sequences in parallel"""
        # Group indices by dataset
        dataset_indices = {}
        for idx, (db_idx, seq_idx) in enumerate(batch_indices):
            if db_idx not in dataset_indices:
                dataset_indices[db_idx] = []
            dataset_indices[db_idx].append(seq_idx)
        
        # Generate sequences for each dataset in parallel
        all_sequences = []
        for db_idx, indices in dataset_indices.items():
            db = self.dblist[db_idx]
            # Use __getitems__ for parallel generation
            sequences = db.__getitems__(indices)
            all_sequences.extend(sequences)
        
        return all_sequences

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __getitems__(self, indices):
        return [self.sequences[i] for i in indices]