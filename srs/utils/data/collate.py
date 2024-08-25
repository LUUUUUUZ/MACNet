import random
import numpy as np
import pandas as pd
import torch
import dgl
import random
import copy


# Fix random seed for reproducibility
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.2):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence

class Substitute1(object):
    """Randomly substitute k items given a sequence"""

    def __init__(self,knowledge_graph, gamma=0.2):
        self.gamma = gamma
        self.knowledge_graph = knowledge_graph
        self.saved_substitute = self.find_substitute()

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        if len(sequence)==1:
            copied_sequence[0] = self.saved_substitute[sequence[0]]
            return copied_sequence 
        mask_nums = int(self.gamma * len(copied_sequence))
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx in mask_idx:
            copied_sequence[idx] = self.saved_substitute[sequence[idx]]
        return copied_sequence
    
    def find_substitute(self):
        # iteratively find the substitute
        return_dict = {}
        for each_id in self.knowledge_graph.nodes('item'):
            # find items ids:
            original_id = each_id.item()
            max_cnt = -1
            new_id = original_id
            for item in self.knowledge_graph.successors(original_id,etype='transitsto').tolist():
                edge_id = self.knowledge_graph.edge_ids(original_id,item,etype='transitsto')
                cnt = self.knowledge_graph.edges['transitsto'].data['cnt'][edge_id]
                if cnt > max_cnt:
                    new_id = item
                    max_cnt = cnt
            return_dict[original_id]=new_id 
        return return_dict

class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index : start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length :]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq

    
class Insert(object):
    """Insert similar items every time call"""
    def __init__(self, item_similarity_model, insert_rate=0.2):

        self.item_similarity_model = item_similarity_model
        self.insert_rate = insert_rate
        self.max_len = 50
    
    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
        insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                inserted_sequence += [self.item_similarity_model.most_similar(item)]
            inserted_sequence += [item]

        if len(inserted_sequence) > self.max_len:
            return inserted_sequence[-(self.max_len):]
        else:
            return inserted_sequence

                    
class Substitute(object):
    """Substitute with similar items"""
    def __init__(self, item_similarity_model, substitute_rate=0.2):
        self.item_similarity_model = item_similarity_model
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        substitute_nums = max(int(self.substitute_rate*len(copied_sequence)), 1)
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k = substitute_nums)
        inserted_sequence = []
        for index in substitute_idx:
            copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])
        return copied_sequence


def sample_blocks(g, uniq_uids, uniq_iids, fanouts, steps):
    seeds = {'user': torch.LongTensor(uniq_uids), 'item': torch.LongTensor(uniq_iids)}
    blocks = []
    for fanout in fanouts:
        if fanout <= 0:
            frontier = dgl.in_subgraph(g, seeds)
        else:
            frontier = dgl.sampling.sample_neighbors(
                g, seeds, fanout, copy_ndata=False, copy_edata=True
            )
        block = dgl.to_block(frontier, seeds)
        seeds = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
        blocks.insert(0, block)
    return blocks, seeds

class CollateFnContrativeLearning:
    """
    Collation function for Contrastive Learning.
    """
    def __init__(self, knowledge_graph, num_layers, num_neighbors, seq_to_graph_fns, noise_ratio, num_items, similarity_model, augment_type, **kwargs):
        self.knowledge_graph = knowledge_graph
        self.num_layers = num_layers
        self.seq_to_graph_fns = seq_to_graph_fns
        self.fanouts = num_neighbors * num_layers if len(num_neighbors) != num_layers else num_neighbors
        self.max_len = 50
        self.noise_ratio = noise_ratio
        self.item_size = num_items
        self.similarity_model = similarity_model
        self.augmentations = self._initialize_augmentations(similarity_model)
        self.base_transform = self.augmentations[augment_type]
        print(f"Creating Contrastive Learning Dataset using '{augment_type}' data augmentation")
    
    def _initialize_augmentations(self, similarity_model):
        """
        Initialize data augmentation strategies.
        """
        return {
            'crop': Crop(),
            'mask': Mask(),
            'reorder': Reorder(),
            'substitute': Substitute(similarity_model),
            'insert': Insert(similarity_model),
        }

    def _one_pair_data_augmentation(self, input_ids):
        """
        Provides two positive samples given one sequence.
        """
        if len(input_ids) < 2:
            return input_ids, input_ids

        augmented_seqs = [self.base_transform(input_ids) for _ in range(2)]
        return augmented_seqs[0], augmented_seqs[1]

    def _process_sequences(self, seqs):
        """
        Processes sequences for unique ID mapping.
        """
        iids = np.concatenate(seqs)
        new_iids, uniq_iids = pd.factorize(iids, sort=True)
        new_seqs = self._factorize_sequences(seqs, new_iids)
        return new_iids, uniq_iids, new_seqs

    def _factorize_sequences(self, seqs, new_iids):
        """
        Factorizes sequences into new ID space.
        """
        cur_idx = 0
        new_seqs = []
        for seq in seqs:
            new_seq = new_iids[cur_idx:cur_idx + len(seq)]
            cur_idx += len(seq)
            new_seqs.append(new_seq)
        return new_seqs

    def _pad_sequences(self, sequences):
        """
        Pads sequences to a fixed length.
        """
        padded_sequences = []
        for seq in sequences:
            padded_seq = [-1] * self.max_len 
            padded_seq[-len(seq): ] = seq
            padded_sequences.append(padded_seq)
        return torch.LongTensor(padded_sequences)

    def _create_input_tensors(self, new_uids, padded_seqs):
        """
        Creates input tensors for the model.
        """
        pos = torch.arange(self.max_len)
        inputs = [torch.LongTensor(new_uids), padded_seqs, pos]
        for seq_to_graph in self.seq_to_graph_fns:
            graphs = [seq_to_graph(seq) for seq in padded_seqs]
            inputs.append(dgl.batch(graphs))
        return inputs

    def _generate_augmented_views(self, original_seqs, new_uids, uniq_uids,fanouts):
        """
        Generates augmented views of the sequences.
        """
        augmented_seqs = [self._one_pair_data_augmentation(seq) for seq in original_seqs]
        first_view, second_view = zip(*augmented_seqs)
        combined_seqs = list(first_view) + list(second_view)
        augmented_iids, augmented_new_seqs = self._process_augmented_sequences(combined_seqs)
        augmented_padded_seqs = self._pad_sequences(augmented_new_seqs)

        augmented_inputs = self._create_input_tensors(torch.LongTensor(new_uids).repeat(2), augmented_padded_seqs)
        augmented_extra_inputs = sample_blocks(self.knowledge_graph, uniq_uids, augmented_iids, fanouts, self.num_layers)
        
        return augmented_inputs, augmented_extra_inputs

    def _process_augmented_sequences(self, augmented_seqs):
        """
        Processes augmented sequences for unique ID mapping.
        """
        augmented_iids = np.concatenate(augmented_seqs)
        augmented_new_iids, augmented_uniq_iids = pd.factorize(augmented_iids, sort=True)
        augmented_new_seqs = self._factorize_sequences(augmented_seqs, augmented_new_iids)
        return augmented_uniq_iids, augmented_new_seqs

    def collate_train(self, samples):
        """
        Collates samples for training.
        """
        return self._collate_fn(samples, self.fanouts)

    def collate_test(self, samples):
        """
        Collates samples for testing, optionally adding noise.
        """
        uids, seqs, labels = zip(*samples)
        padded_seqs = self._prepare_test_sequences(seqs)
        uids = torch.LongTensor(uids)
        labels = torch.LongTensor(labels)

        inputs = self._create_input_tensors(uids, padded_seqs)
        return (inputs, ), labels

    def _prepare_test_sequences(self, seqs):
        """
        Prepares test sequences, adding noise if specified.
        """
        if self.noise_ratio != 0:
            noisy_seqs = [self._add_noise_interactions(seq) for seq in seqs]
            padded_seqs = self._pad_sequences([seq[-self.max_len:] if len(seq) > self.max_len else seq for seq in noisy_seqs])
        else:
            padded_seqs = self._pad_sequences(seqs)
        return padded_seqs

    def collate_test_otf(self, samples):
        """
        Collates samples for on-the-fly testing.
        """
        return self._collate_fn(samples, [0] * self.num_layers)

    def _add_noise_interactions(self, items):
        """
        Adds noise to the interaction sequences.
        """
        copied_sequence = copy.deepcopy(items)
        insert_num = max(int(self.noise_ratio * len(copied_sequence)), 0)
        if insert_num == 0:
            return copied_sequence

        insert_idx = random.choices(range(len(copied_sequence)), k=insert_num)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.item_size - 2)
                inserted_sequence.append(item_id)
            inserted_sequence.append(item)
        return inserted_sequence
