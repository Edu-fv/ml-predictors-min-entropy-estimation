from abc import ABC, abstractmethod
from math import log

import torch
import torch.nn.functional as F


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _block_size(model):
    return _base_model(model).config.block_size


def _check_binary_autoregressive(config):
    if not config.get("is_autoregressive", False):
        raise ValueError(
            "Autoregressive distillation requires --is_autoregressive so rollouts "
            "append binary tokens to a binary context."
        )


def _append_token(context, token, block_size):
    context = torch.cat([context, token.unsqueeze(1)], dim=1)
    return context[:, -block_size:]


@torch.no_grad()
def score_sequence(model, context, sequence):
    """Return log p_model(sequence | context) for binary autoregressive models."""
    current = context.clone()
    block_size = _block_size(model)
    log_probs = []

    for step in range(sequence.size(1)):
        token = sequence[:, step]
        logits = model(current).logits[:, -1, :]
        step_log_probs = F.log_softmax(logits, dim=-1)
        log_probs.append(step_log_probs.gather(1, token.unsqueeze(1)).squeeze(1))
        current = _append_token(current, token, block_size)

    return torch.stack(log_probs, dim=1).sum(dim=1)


@torch.no_grad()
def rollout(model, context, num_steps, mode="greedy", temperature=1.0):
    """Generate a binary continuation and score it under the generating model."""
    current = context.clone()
    block_size = _block_size(model)
    tokens = []
    log_probs = []

    for _ in range(num_steps):
        logits = model(current).logits[:, -1, :]
        scaled_logits = logits / temperature
        step_log_probs = F.log_softmax(scaled_logits, dim=-1)

        if mode == "greedy":
            token = step_log_probs.argmax(dim=-1)
        elif mode == "sample":
            probs = step_log_probs.exp()
            token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError(f"Unknown rollout mode: {mode}")

        tokens.append(token)
        log_probs.append(step_log_probs.gather(1, token.unsqueeze(1)).squeeze(1))
        current = _append_token(current, token, block_size)

    return torch.stack(tokens, dim=1), torch.stack(log_probs, dim=1).sum(dim=1)


@torch.no_grad()
def beam_search(model, context, num_steps, beam_width=8):
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    current_beam = context.unsqueeze(1)
    batch_size = context.size(0)
    block_size = _block_size(model)
    beam_scores = torch.zeros(batch_size, 1, device=context.device)
    beam_sequences = torch.empty(batch_size, 1, 0, dtype=torch.long, device=context.device)

    for _ in range(num_steps):
        batch_size, current_width, context_len = current_beam.shape
        flat_context = current_beam.reshape(batch_size * current_width, context_len)
        logits = model(flat_context).logits[:, -1, :]
        if logits.size(-1) != 2:
            raise ValueError(
                f"BSD requires a binary autoregressive model, got vocab_size={logits.size(-1)}"
            )
        log_probs = F.log_softmax(logits, dim=-1).reshape(batch_size, current_width, 2)

        candidate_scores = beam_scores.unsqueeze(-1) + log_probs
        next_width = min(beam_width, current_width * 2)
        flat_scores = candidate_scores.reshape(batch_size, current_width * 2)
        top_scores, top_idx = flat_scores.topk(next_width, dim=1)
        parent_idx = top_idx // 2
        next_token = top_idx % 2
        batch_idx = torch.arange(batch_size, device=context.device).unsqueeze(1)

        selected_context = current_beam[batch_idx, parent_idx]
        selected_sequence = beam_sequences[batch_idx, parent_idx]
        flat_selected_context = selected_context.reshape(batch_size * next_width, context_len)
        flat_next_token = next_token.reshape(-1)

        current_beam = _append_token(
            flat_selected_context, flat_next_token, block_size
        ).reshape(batch_size, next_width, -1)
        beam_sequences = torch.cat([selected_sequence, next_token.unsqueeze(-1)], dim=-1)
        beam_scores = top_scores

    return beam_sequences[:, 0, :], beam_scores[:, 0]


@torch.no_grad()
def evaluate_teacher_scored_rollouts(teacher, student, eval_data, config, device):
    """Compare teacher-greedy and student-greedy continuations under the teacher."""
    _check_binary_autoregressive(config)

    horizon = config["target_bits"]
    max_batches = config.get("distillation_eval_batches", 4)
    teacher_scores = []
    student_scores = []
    best_scores = []
    student_beats_teacher = []
    same_sequence = []

    teacher.eval()
    student.eval()

    for batch_idx, (x, _) in enumerate(eval_data):
        if batch_idx >= max_batches:
            break
        x = x.to(device)

        teacher_sequence, teacher_logprob = rollout(teacher, x, horizon, mode="greedy")
        student_sequence, _ = rollout(student, x, horizon, mode="greedy")
        student_logprob = score_sequence(teacher, x, student_sequence)
        best_logprob = torch.maximum(teacher_logprob, student_logprob)

        teacher_scores.append(teacher_logprob)
        student_scores.append(student_logprob)
        best_scores.append(best_logprob)
        student_beats_teacher.append((student_logprob > teacher_logprob).float())
        same_sequence.append((student_sequence == teacher_sequence).all(dim=1).float())

    if not teacher_scores:
        return {}

    teacher_scores = torch.cat(teacher_scores)
    student_scores = torch.cat(student_scores)
    best_scores = torch.cat(best_scores)
    student_beats_teacher = torch.cat(student_beats_teacher)
    same_sequence = torch.cat(same_sequence)
    nats_to_bits_per_token = -1.0 / (horizon * log(2.0))

    return {
        "ad_horizon": horizon,
        "ad_eval_samples": teacher_scores.numel(),
        "teacher_greedy_logprob": teacher_scores.mean().item(),
        "student_greedy_teacher_logprob": student_scores.mean().item(),
        "best_greedy_teacher_logprob": best_scores.mean().item(),
        "student_minus_teacher_logprob": (student_scores - teacher_scores).mean().item(),
        "student_beats_teacher_rate": student_beats_teacher.mean().item(),
        "student_teacher_same_sequence_rate": same_sequence.mean().item(),
        "teacher_greedy_entropy_ub": (teacher_scores * nats_to_bits_per_token).mean().item(),
        "student_greedy_entropy_ub": (student_scores * nats_to_bits_per_token).mean().item(),
        "best_greedy_entropy_ub": (best_scores * nats_to_bits_per_token).mean().item(),
    }


class DistillationStrategy(ABC):
    @abstractmethod
    def compute_loss(self, student, teacher, x, y, config, device): ...


class RADStrategy(DistillationStrategy):
    """REINFORCE autoregressive distillation with teacher log-likelihood rewards."""

    def __init__(
        self,
        num_steps=8,
        temperature=1.0,
        entropy_coef=0.0,
        baseline="teacher_greedy",
        normalize_advantages=True,
        **kwargs,
    ):
        self.num_steps = num_steps
        self.temperature = temperature
        self.entropy_coef = entropy_coef
        self.baseline = baseline
        self.normalize_advantages = normalize_advantages

    def compute_loss(self, student, teacher, x, y, config, device):
        _check_binary_autoregressive(config)

        current = x.clone()
        block_size = _block_size(student)
        policy_log_probs = []
        teacher_rewards = []
        entropies = []

        for _ in range(self.num_steps):
            student_logits = student(current).logits[:, -1, :] / self.temperature
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            student_probs = student_log_probs.exp()
            action = torch.multinomial(student_probs, num_samples=1).squeeze(1)

            policy_log_probs.append(
                student_log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            )
            entropies.append(-(student_probs * student_log_probs).sum(dim=-1))

            with torch.no_grad():
                teacher_logits = teacher(current).logits[:, -1, :]
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                teacher_rewards.append(
                    teacher_log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
                )

            current = _append_token(current, action, block_size)

        policy_log_probs = torch.stack(policy_log_probs, dim=1)
        teacher_rewards = torch.stack(teacher_rewards, dim=1)
        entropies = torch.stack(entropies, dim=1)

        sequence_rewards = teacher_rewards.sum(dim=1)
        if self.baseline == "teacher_greedy":
            with torch.no_grad():
                greedy_sequence, _ = rollout(teacher, x, self.num_steps, mode="greedy")
                baseline = score_sequence(teacher, x, greedy_sequence)
        elif self.baseline == "batch_mean":
            baseline = sequence_rewards.mean().expand_as(sequence_rewards)
        else:
            baseline = torch.zeros_like(sequence_rewards)

        advantages = sequence_rewards - baseline
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = advantages / (advantages.std(unbiased=False) + 1e-8)

        reinforce_loss = -(policy_log_probs.sum(dim=1) * advantages.detach()).mean()
        entropy_bonus = entropies.sum(dim=1).mean()

        return reinforce_loss - self.entropy_coef * entropy_bonus


class VADStrategy(DistillationStrategy):
    """Stop-gradient Gumbel-Softmax surrogate for autoregressive distillation."""

    def __init__(
        self,
        num_steps=8,
        tau_start=1.0,
        tau_end=0.2,
        tau_progression="linear",
        entropy_coef=0.0,
        teacher_kl_coef=0.02,
        straight_through=True,
        **kwargs,
    ):
        self.num_steps = num_steps
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_progression = tau_progression
        self.entropy_coef = entropy_coef
        self.teacher_kl_coef = teacher_kl_coef
        self.straight_through = straight_through
        self._current_tau = tau_start
        self._tau_delta = 0.0

    def set_alpha(self, total_steps):
        total_updates = max(1, total_steps * self.num_steps)
        if self.tau_progression == "exp":
            self._tau_delta = (self.tau_end / self.tau_start) ** (1.0 / total_updates)
        elif self.tau_progression == "linear":
            self._tau_delta = (self.tau_end - self.tau_start) / total_updates
        else:
            raise ValueError(f"Unknown tau progression: {self.tau_progression}")

    def step_tau(self):
        if self.tau_progression == "exp":
            next_tau = self._current_tau * self._tau_delta
        else:
            next_tau = self._current_tau + self._tau_delta
        self._current_tau = max(self.tau_end, next_tau)

    def compute_loss(self, student, teacher, x, y, config, device):
        _check_binary_autoregressive(config)

        current = x.clone()
        block_size = _block_size(student)
        rewards = []
        entropies = []
        kl_terms = []

        for _ in range(self.num_steps):
            student_logits = student(current).logits[:, -1, :]
            relaxed = F.gumbel_softmax(
                student_logits,
                tau=self._current_tau,
                hard=self.straight_through,
            )
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            student_probs = student_log_probs.exp()
            entropies.append(-(student_log_probs.exp() * student_log_probs).sum(dim=-1))

            with torch.no_grad():
                teacher_logits = teacher(current).logits[:, -1, :]
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

            rewards.append((relaxed * teacher_log_probs).sum(dim=-1))
            kl_terms.append(
                (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
            )
            hard_token = relaxed.argmax(dim=-1)
            current = _append_token(current, hard_token, block_size)
            self.step_tau()

        reward = torch.stack(rewards, dim=1).sum(dim=1).mean()
        entropy_bonus = torch.stack(entropies, dim=1).sum(dim=1).mean()
        teacher_kl = torch.stack(kl_terms, dim=1).sum(dim=1).mean()
        return -reward + self.teacher_kl_coef * teacher_kl - self.entropy_coef * entropy_bonus


class IRBCStrategy(DistillationStrategy):
    """Teacher sampling + local bit-flip refinement + behaviour cloning."""

    def __init__(self, num_steps=8, num_candidates=4, tau=0.7, refinement_passes=1, **kwargs):
        self.num_steps = num_steps
        self.num_candidates = num_candidates
        self.tau = tau
        self.refinement_passes = refinement_passes

    @torch.no_grad()
    def sample_sequences(self, teacher, x):
        samples = []
        for _ in range(self.num_candidates):
            sequence, _ = rollout(
                teacher,
                x,
                self.num_steps,
                mode="sample",
                temperature=self.tau,
            )
            samples.append(sequence)
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def refine_sequences(self, teacher, x, candidates):
        batch_size, num_candidates, horizon = candidates.shape
        flat_x = x.unsqueeze(1).expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates, -1
        )
        current = candidates.reshape(batch_size * num_candidates, horizon).clone()
        current_scores = score_sequence(teacher, flat_x, current)

        for _ in range(self.refinement_passes):
            for step in range(horizon):
                proposal = current.clone()
                proposal[:, step] = 1 - proposal[:, step]
                proposal_scores = score_sequence(teacher, flat_x, proposal)
                improved = proposal_scores > current_scores
                current[improved] = proposal[improved]
                current_scores[improved] = proposal_scores[improved]

        current = current.view(batch_size, num_candidates, horizon)
        current_scores = current_scores.view(batch_size, num_candidates)
        best_idx = current_scores.argmax(dim=1)
        return current[torch.arange(batch_size, device=x.device), best_idx]

    def compute_loss(self, student, teacher, x, y, config, device):
        _check_binary_autoregressive(config)

        with torch.no_grad():
            candidates = self.sample_sequences(teacher, x)
            targets = self.refine_sequences(teacher, x, candidates)

        current = x.clone()
        block_size = _block_size(student)
        losses = []

        for step in range(self.num_steps):
            token = targets[:, step]
            logits = student(current).logits[:, -1, :]
            losses.append(F.cross_entropy(logits, token, reduction="none"))
            current = _append_token(current, token, block_size)

        return torch.stack(losses, dim=1).mean()


class BSDStrategy(DistillationStrategy):
    """Beam-search teacher oracle with supervised student cloning."""

    def __init__(self, num_steps=8, beam_width=8, **kwargs):
        self.num_steps = num_steps
        self.beam_width = beam_width

    def compute_loss(self, student, teacher, x, y, config, device):
        _check_binary_autoregressive(config)

        with torch.no_grad():
            targets, _ = beam_search(teacher, x, self.num_steps, self.beam_width)

        current = x.clone()
        block_size = _block_size(student)
        losses = []

        for step in range(self.num_steps):
            token = targets[:, step]
            logits = student(current).logits[:, -1, :]
            losses.append(F.cross_entropy(logits, token, reduction="none"))
            current = _append_token(current, token, block_size)

        return torch.stack(losses, dim=1).mean()
