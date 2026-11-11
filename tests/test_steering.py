"""Steering row regression tests using fixed logits, without model weights."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
from runtime import Runtime

class FixedModel:
    lm_head = SimpleNamespace(out_features=6)
    def __call__(self, input_ids, **kwargs):
        logits = torch.tensor([[[0., 3., 2.8, 2.6, 1., -100.]]]).repeat(len(input_ids), 1, 1)
        return SimpleNamespace(logits=logits, past_key_values=None)

class FakeRuntime(Runtime):
    def __init__(self):
        self.dev = 'cpu'
        self.lock = threading.RLock()
        self._push = None
        self.entries = []
        self.model = FixedModel()
        self.tok = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda _: 5,
                                   decode=lambda ids, **kwargs: ''.join(map(str, ids)))
    def ensure_loaded(self): pass
    def messages_ids(self, messages): return messages
    def _push_entry(self, *args):
        self.entries.append(args)
        return args

class SteeringTests(unittest.TestCase):
    def test_three_rows_retain_original_directions(self):
        rt = FakeRuntime()
        result = list(rt.steer_multi([(1, 2, .8, 'add')], [[1], [1, 2], [1]], max_new=3))
        self.assertEqual(result, [(i, '1') for _ in range(3) for i in range(3)] + [(None, None)])
        self.assertEqual(rt.entries[0][-1], (1, 0, -1))
        self.assertIsNone(rt._push)
    def test_sampling_controls_and_seed(self):
        rt = FakeRuntime()
        kwargs = dict(max_new=8, temperature=.8, repetition_penalty=1.2, seed=17, min_p=.1)
        first = list(rt.steer_multi([(1, 2, -.5, 'add')], [[1, 2]] * 3, **kwargs))
        self.assertEqual(first, list(rt.steer_multi([(1, 2, -.5, 'add')], [[1, 2]] * 3, **kwargs)))
        self.assertEqual({row for row, _ in first}, {0, 1, 2, None})
        self.assertEqual(rt.entries[0][-1], (1, 0, -1))
    def test_repetition_penalty(self):
        rt = FakeRuntime()
        self.assertEqual(list(rt.steer_multi([], [[1]] * 3, max_new=1, repetition_penalty=2)),
                         [(i, '2') for i in range(3)] + [(None, None)])
    def test_cancelled_before_generation(self):
        rt = FakeRuntime()
        cancel = threading.Event(); cancel.set()
        self.assertEqual(list(rt.steer_multi([], [[1]] * 3, cancel=cancel)), [(None, None)])
    def test_model_failure_clears_steering(self):
        rt = FakeRuntime()
        def fail(**kwargs): raise RuntimeError('failed')
        rt.model = fail
        with self.assertRaises(RuntimeError):
            list(rt.steer_multi([(1, 2, 1, 'mul')], [[1]] * 3))
        self.assertIsNone(rt._push)

if __name__ == '__main__': unittest.main()
