import numpy as np
import operator


class ReplayBuffer:
    def __init__(self, size):
        assert size > 0
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx += 1
        if self._next_idx == self._maxsize:
            self._next_idx = 0

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        n = len(self._storage)
        idxes = [np.random.randint(n) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._it_max = MaxSegmentTree(it_capacity)

    def add(self, *args, **kwargs):
        idx = self._next_idx
        super(PrioritizedReplayBuffer, self).add(*args, **kwargs)

        max_priority = self._it_max.max()
        if max_priority <= 0:
            max_priority = 1.0

        self._it_sum[idx] = self._it_min[idx] = self._it_max[idx] = max_priority

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum()
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.rand() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_sum = self._it_sum.sum()
        p_min = self._it_min.min() / p_sum
        n = len(self._storage)
        max_weight = (p_min * n) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / p_sum
            weight = (p_sample * n) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        n = len(self._storage)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < n
            self._it_sum[idx] = \
                self._it_min[idx] = \
                self._it_max[idx] = min(1.0, priority) ** self._alpha


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2"
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        elif mid + 1 <= start:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2 * node, node_start, mid),
                self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

    def __setitem__(self, idx, val):
        assert val >= 0
        super(SumSegmentTree, self).__setitem__(idx, val)


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


class MaxSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MaxSegmentTree, self).__init__(
            capacity=capacity,
            operation=max,
            neutral_element=float('-inf')
        )

    def max(self, start=0, end=None):
        return super(MaxSegmentTree, self).reduce(start, end)

