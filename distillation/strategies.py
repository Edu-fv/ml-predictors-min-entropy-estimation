from abc import ABC, abstractmethod

import torch


class DistillationStrategy(ABC):

    @abstractmethod
    def compute_loss(self, student, teacher, x, y, config, device): ...


class RADStrategy(DistillationStrategy):
    """REINFORCE-based Autoregressive Distillation.

    Student generates tokens autoregressively; teacher scores them.
    Loss drives student toward sequences with high joint probability
    under the teacher (MAP approximation for min-entropy).
    """

    def __init__(self, num_steps=5, reward_decay=1.0, temperature=1.0):
        self.num_steps = num_steps
        self.reward_decay = reward_decay
        self.temperature = temperature

    def compute_loss(self, student, teacher, x, y, config, device):
        batch_size = x.size(0)
        target_bits = config["target_bits"]
        eps = 1e-10

        log_probs = []
        joint_prob = torch.ones(batch_size, device=device)
        current_x = x.clone()

        for step in range(self.num_steps):
            output = student(current_x)
            logits = output.logits[:, -1, :] / self.temperature

            probs = torch.softmax(logits, dim=-1)
            actions = torch.distributions.Categorical(probs).sample()

            log_probs_all = torch.log_softmax(logits, dim=-1)
            step_logp = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_probs.append(step_logp)

            with torch.no_grad():
                ref_out = teacher(current_x)
                ref_probs = torch.softmax(ref_out.logits[:, -1, :], dim=-1)
                chosen_ref_probs = ref_probs.gather(-1, actions.unsqueeze(-1)).squeeze(
                    -1
                )
                joint_prob = joint_prob * (chosen_ref_probs**self.reward_decay)

            current_x = torch.cat([current_x[:, 1:], actions.unsqueeze(1)], dim=1)

        min_entropy_per_bit = -torch.log2(joint_prob + eps) / (
            self.num_steps * target_bits
        )
        rewards = -min_entropy_per_bit

        total_logp = sum(log_probs)
        advantages = rewards - rewards.mean()
        return -(total_logp * advantages).mean()


class VADStrategy(DistillationStrategy):
    """Variational AD via Gumbel-Softmax."""

    def __init__(self, num_steps=5, tau_start=1.0, tau_end=0.1, tau_progression="linear", **kwargs):
        self.num_steps = num_steps
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.alpha = 0
        self.__current_tau = tau_start
        self.tau_progression = tau_progression

    def set_alpha(self, total_steps):
        if self.tau_progression == "exp":
            self.alpha = (self.tau_end / self.tau_start) ** (1.0 / (total_steps * self.num_steps))
        elif self.tau_progression == "linear":
            self.alpha = (self.tau_end - self.tau_start)  / (total_steps * self.num_steps)
        
    def step_tau(self):
        if self.tau_progression == "exp":
            new_tau = self.__current_tau * self.alpha
        elif self.tau_progression == "linear":
            new_tau = self.__current_tau + self.alpha
        self.__current_tau = new_tau if new_tau >= self.tau_end else self.tau_end

    def compute_loss(self, student, teacher, x, y, config, device):
        batch_size = x.size(0)
        target_bits = config["target_bits"]
        eps = 1e-10
        log_probs = []
        joint_prob = torch.ones(batch_size, device=device)
        current_x = x.clone()
        rewards = []
        for step in range(self.num_steps):
            output = student(current_x)
            logits = output.logits[:, -1, :]
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
            y = torch.softmax((logits+gumbel_noise)/self.__current_tau, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            actions = torch.distributions.Categorical(probs).sample()
            with torch.no_grad():
                ref_out = teacher(current_x)
                ref_log_probs = torch.log_softmax(ref_out.logits[:, -1, :], dim=-1)
            current_x = torch.cat([current_x[:, 1:], actions.unsqueeze(1)], dim=1)
            reward = y*ref_log_probs
            rewards.append(reward)

        #reward_final = torch.stack(rewards, dim=1)
        advantages = reward - reward.mean()
        self.step_tau()
        return -advantages.mean()
        


class IRBCStrategy(DistillationStrategy):
    """Iterative Refinement + Behaviour Cloning (not yet implemented)."""

    def __init__(self, num_steps=5, num_candidates=16, tau=0.7, refinement_method="coordinate_ascent", **kwargs):
        self.num_steps = num_steps
        self.num_candidates = num_candidates
        self.tau = tau
        self.refinement_method = refinement_method


    @torch.no_grad()
    def sequence_logprob(self, model, x, candidates):
        B, C, T = candidates.shape

        x_len = x.shape[1]

        # repeat prompts
        x_rep = x.unsqueeze(1).repeat(1, C, 1)

        full_seq = torch.cat(
            [x_rep, candidates],
            dim=-1
        )

        # [B*C, L]
        full_seq = full_seq.reshape(B * C, -1)


        block_size = model.config.block_size

        full_seq = full_seq[:, -block_size:]


        seq_len = full_seq.shape[1]


        logits = model(full_seq).logits

        visible_T = min(T, seq_len - 1)

        # targets are last T tokens
        pred_logits = logits[:, -visible_T - 1: - 1, :]

        target = candidates[:, :, -visible_T:]
        target = target.reshape(B * C, visible_T)


        log_probs = torch.log_softmax(pred_logits, dim=-1)

        token_log_probs = log_probs.gather(
            -1,
            target.unsqueeze(-1)
        ).squeeze(-1)

        seq_log_probs = token_log_probs.sum(dim=-1)

        return seq_log_probs.view(B, C)
    


    @torch.no_grad()
    def sample_sequences(self, model, x, batch_size, seq_len, device):

        """# Repeat prompts for all candidates
        x_local = x.unsqueeze(1).repeat(1, self.num_candidates, 1)

        # [B, C, L] -> [B*C, L]
        x_local = x_local.reshape(batch_size * self.num_candidates, -1)

        generated = []

        for _ in range(target_bits):

            logits = model(x_local)
            # assume logits shape:
            # [B*C, seq_len, 2]

            next_logits = logits[:, -1, :]  # [B*C, 2]

            # temperature scaling
            next_logits = next_logits / self.tau

            probs = torch.functional.softmax(next_logits, dim=-1)

            next_bit = torch.multinomial(probs, num_samples=1)

            generated.append(next_bit)

            x_local = torch.cat([x_local, next_bit], dim=1)

        generated = torch.cat(generated, dim=1)

        generated = generated.view(
            batch_size,
            self.num_candidates,
            target_bits
        )

        return generated""" 
           
        with torch.no_grad():
            samples = []
            for candidate in range(self.num_candidates):
                current_sample = torch.ones(batch_size,seq_len, device=device, dtype=torch.long)
                for bit in range(seq_len):
                    teacher_out = model(x)
                    teacher_logits = teacher_out.logits[:, -1, :]
                    sample_probs = torch.softmax(teacher_logits/self.tau, dim=-1)
                    current_sample[:, bit] = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
                samples.append(current_sample)
        return torch.stack(samples, dim=1)


    def coordinate_ascent(self, model, x, candidates):
        B, C, T = candidates.shape

        current = candidates.clone()

        current_scores = self.sequence_logprob(
            model,
            x,
            current,
        )

        for _ in range(self.num_steps):

            for bit_idx in range(T):

                proposal = current.clone()

                # flip bit
                proposal[:, :, bit_idx] = (
                    1 - proposal[:, :, bit_idx]
                )

                proposal_scores = self.sequence_logprob(
                    model,
                    x,
                    proposal,
                )

                improved = proposal_scores > current_scores

                current[improved, :] = proposal[improved, :]
                current_scores[improved] = proposal_scores[improved]

        return current, current_scores

    def refine_samples(self, model, x, candidates):
        if self.refinement_method == "coordinate_ascent":
            return self.coordinate_ascent(model, x, candidates)

    def compute_loss(self, student, teacher, x, y, config, device):
        """raise NotImplementedError(
            "IRBC: teacher sampling + cloning not yet implemented."
        )
        """        
        batch_size = config["batch_size"]
        target_bits = config["target_bits"]
        seq_len = config["seqlen"]
        
        candidates = self.sample_sequences(teacher, x, batch_size, seq_len, device)

        refined_candidates, scores = self.refine_samples(teacher, x, candidates)

        best_idx = scores.argmax(dim=1)

        B = x.shape[0]

        best_sequences = refined_candidates[
            torch.arange(B),
            best_idx
        ]



        B, T = best_sequences.shape

        # teacher-forced input
        full_input = torch.cat(
            [x, best_sequences],
            dim=1
        )


        block_size = student.config.block_size

        full_input = full_input[:, -block_size:]


        seq_len = full_input.shape[1]



        logits = student(full_input).logits

        prompt_len = x.shape[1]

        # logits predicting target bits
        pred_logits = logits[:, prompt_len - 1:-1, :]

        # [B, T, 2]


        loss = torch.nn.CrossEntropyLoss()
        output = loss(
            pred_logits.reshape(-1, 2),
            best_sequences.reshape(-1),
        )

        return output.mean()

