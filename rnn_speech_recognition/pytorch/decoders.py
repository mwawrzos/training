# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math

import torch

import torch.nn.functional as F
from model_rnnt import label_collate


class TransducerDecoder:
    """Decoder base class.

    Args:
        alphabet: An Alphabet object.
        blank_symbol: The symbol in `alphabet` to use as the blank during CTC
            decoding.
        model: Model to use for prediction.
    """

    def __init__(self, blank_index, model):
        self._model = model
        self._SOS = -1   # start of sequence
        self._blank_id = blank_index

    def _pred_step(self, label, hidden, device):
        if label == self._SOS:
            return self._model.predict(None, hidden, add_sos=False)
        if label > self._blank_id:
            label -= 1
        label = label_collate([[label]]).to(device)
        return self._model.predict(label, hidden, add_sos=False)

    def _joint_step(self, enc, pred, log_normalize=False):
        logits = self._model.joint(enc, pred)[:, 0, 0, :]
        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)

        return probs

    def _get_last_symb(self, labels):
        return self._SOS if labels == [] else labels[-1]


class RNNTGreedyDecoder(TransducerDecoder):
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """
    def __init__(self, blank_index, model, max_symbols_per_step=30):
        super().__init__(blank_index, model)
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step

    def decode(self, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
                TODO was (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        with torch.no_grad():
            # Apply optional preprocessing

            logits, out_lens = self._model.encode((x, out_lens))

            output = []
            for batch_idx in range(logits.size(0)):
                inseq = logits[batch_idx, :, :].unsqueeze(1)
                logitlen = out_lens[batch_idx]
                sentence = self._greedy_decode(inseq, logitlen)
                output.append(sentence)

        return output

    def _greedy_decode(self, x, out_len):
        training_state = self._model.training
        self._model.eval()

        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    self._get_last_symb(label),
                    hidden,
                    device
                )
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self._blank_id:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        self._model.train(training_state)
        return label


class RNNTBeamDecoder(TransducerDecoder):
    """A beam search decoder for transducer models.

    Based on Algorithm 1 from https://arxiv.org/abs/1211.3711.

    Args:
        alphabet: See `Decoder`.
        blank_symbol: See `Decoder`.
        model: See `Decoder`.
        beam_width: Width of the beam search.
        prune_threshold: Affects the list of symbols to consider when extending
            a prefix at each step. A prefix is not extended with a given symbol
            and added to the beam if the symbol has probability less than this
            threshold.
        norm_length: Normalise the probability of each prefix by the prefix
            length before selecting the most probable sequence for a given
            input.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
    """
    def __init__(self, blank_index, model, beam_width=8, length_norm=False,
                 max_symbols_per_step=100, prune_threshold=0.01):
        super().__init__(blank_index, model)
        self.beam_width = beam_width
        self.log_prune_threshold = math.log(prune_threshold)
        self.norm_length = length_norm
        self.max_symbols_per_step = max_symbols_per_step

        if max_symbols_per_step is not None:
            assert (isinstance(max_symbols_per_step, int)
                    and max_symbols_per_step > 0), \
                "max_symbols_per_step must be a positive integer or None"

    def decode(self, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (seq_len, batch, in_features).
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        output = []
        for batch_idx in range(x.size(1)):
            sample = x[:, batch_idx, :]
            try:
                sentence = self._beam_search(sample, out_lens[batch_idx])
            except AssertionError as e:
                logging.warning(e)
                sentence = []

            output.append(sentence)

        return output

    @staticmethod
    def _is_prefix(a, b):
        """Returns True if a is a proper prefix of b."""
        return len(a) < len(b) and a == b[:len(a)]

    @torch.no_grad()
    def _beam_search(self, sample, out_len):
        training_state = self._model.training
        self._model.eval()

        device = sample.device

        sample = sample.unsqueeze(1)
        fs, out_len = self._model.encode((sample, out_len))

        B = [Sequence()]
        for time_idx in range(out_len):
            f = fs[:, time_idx, :].unsqueeze(1)

            A = sorted(B, key=lambda a: len(a.labels), reverse=True)
            B = []
            for i, y in enumerate(A):
                for j in range(i + 1, len(A)):
                    y_hat = A[j]
                    if not self._is_prefix(y_hat.labels, y.labels):
                        continue
                    curlogp = y_hat.logp

                    preds = torch.cat(
                        y.pred_log[len(y_hat.labels):len(y.labels)]
                    )
                    logps = self._joint_step(
                        f, preds, log_normalize=True
                    )

                    extend_range = range(len(y_hat.labels), len(y.labels))
                    for i, k in enumerate(extend_range):
                        curlogp += float(logps[i][y.labels[k]])

                    y.logp = log_aplusb(y.logp, curlogp)
                    assert y.logp <= 0, "logp is not <=0"

            A.sort(key=lambda a: (-a.logp, len(a.labels)))
            while len(A) > 0 and (
                    len(B) < self.beam_width or
                    B[self.beam_width - 1].logp < A[0].logp
                    ):
                y_star = max(A, key=lambda a: (a.logp, len(a.labels)))
                A.remove(y_star)
                pred, hidden = self._pred_step(
                    self._get_last_symb(y_star.labels),
                    y_star.pred_hidden,
                    device
                )
                logp = self._joint_step(
                    f, pred, log_normalize=True
                )[0, :]
                _, indices = torch.sort(logp, descending=True)
                for k in indices.tolist():
                    if logp[k] <= self.log_prune_threshold:
                        break
                    yk = Sequence(y_star)
                    yk.logp += float(logp[k])
                    if k == self._blank_id:
                        if yk not in B:
                            yk.n_step_labels = 0
                            B.append(yk)
                        continue

                    yk.labels.append(k)
                    yk.times.append(time_idx)
                    yk.n_step_labels += 1
                    if self.max_symbols_per_step is not None:
                        if yk.n_step_labels == self.max_symbols_per_step:
                            continue
                    if yk not in A:
                        yk.pred_log.append(pred)
                        yk.pred_hidden = hidden
                        A.append(yk)
                A.sort(key=lambda a: (-a.logp, len(a.labels)))
                B.sort(key=lambda a: (-a.logp, len(a.labels)))

            B = B[:self.beam_width]

        if self.norm_length:
            B.sort(key=lambda a: -a.logp / max(len(a.labels), 0.1))

        logging.debug(
            f"BEAM DECODER, timesteps={B[0].times}, pred_idx={B[0].labels}")

        self._model.train(training_state)

        return B[0].labels


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


class Sequence:
    """A sequence in the beam search.

    Args:
        seq (Sequence): If not None, an existing Sequence to initialize from.
            If not None, `hidden` and `max_symbols` must be None else a
            `ValueError` is raised.
        pred_hidden:

    Attributes:
        pred_log: A list containing the prediction network output
            `torch.Tensor`s. `pred_log[i]` is the prediction network output
            that was used (input to joint network) when computing the `i`th
            symbol in `labels`.

        labels: A list of ints containing the label (index) of each symbol in
            the alphabet.

        times: A list of ints. `times[i]` is the time step that `labels[i]` was
            added to the Sequence. Note that this does not necessarily
            correspond to the input time steps due to downsampling (pooling,
            convolutions, striding, etc).

        pred_hidden: Hidden state of the prediction network for the current
            sequence.

        logp: Log probability of the sequence. Initialised to `math.log(1.0)`
            as all sequences should derive from a single starting sequence.

        n_step_labels: Number of labels added to `labels` during the current
            time step.
    """
    def __init__(self, seq=None, pred_hidden=None):
        if seq is None:
            self.pred_log = []
            self.labels = []
            self.times = []
            self.pred_hidden = pred_hidden
            self.logp = 0.0
            self.n_step_labels = 0
        else:
            if pred_hidden is not None:
                raise ValueError("pred_hidden must be None")
            self.pred_log = seq.pred_log[:]
            self.labels = seq.labels[:]
            self.times = seq.times[:]
            self.pred_hidden = seq.pred_hidden
            self.logp = seq.logp
            self.n_step_labels = seq.n_step_labels

    def __eq__(self, other):
        return self.labels == other.labels

    def __repr__(self):
        p = math.exp(self.logp)
        return (f"{self.__class__.__name__}(labels={self.labels}, "
                f"times={self.times}, "
                f"probability={p}, "
                f"n_step_labels={self.n_step_labels})")
