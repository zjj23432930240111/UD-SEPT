# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Uncertainty-Aware Action Filtering (UAAF) module for LLM-PySC2.

Provides:
- ActionVoter: Extracts action signatures from multiple LLM responses,
  computes Shannon entropy, and performs majority voting.
- UncertaintyFilter: Unified strategy handler (Raw/SC/UAAF).
- ExperimentTracker: Records step-level and game-level metrics for analysis.
"""

import math
import json
import time
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from loguru import logger


class ActionVoter:
    """Extracts action signatures from LLM responses and performs voting."""

    def __init__(self, translator_a):
        """
        Args:
            translator_a: An instance of DefaultTranslatorA (or subclass)
                          used to parse LLM text into action structures.
        """
        self.translator = translator_a

    def extract_signature(self, raw_text: str) -> str:
        """Extract the first action name from an LLM response as a signature.

        This provides a simplified representation for voting comparison.
        If parsing fails, returns 'No_Operation'.

        Args:
            raw_text: Raw LLM output text.

        Returns:
            A string signature (action name) for comparison across calls.
        """
        try:
            action_lists, _, _ = self.translator.translate(raw_text)
            if action_lists and len(action_lists) > 0:
                # Collect all action names from all teams
                all_actions = []
                for team_actions in action_lists:
                    if team_actions:
                        for action in team_actions:
                            all_actions.append(action.get('name', 'No_Operation'))
                if all_actions:
                    return all_actions[0]  # Use first action as signature
        except Exception:
            pass
        return 'No_Operation'

    def extract_all_signatures(self, raw_text: str) -> List[str]:
        """Extract all action names from an LLM response.

        Args:
            raw_text: Raw LLM output text.

        Returns:
            List of action name strings from all teams.
        """
        try:
            action_lists, _, _ = self.translator.translate(raw_text)
            all_actions = []
            for team_actions in action_lists:
                if team_actions:
                    for action in team_actions:
                        all_actions.append(action.get('name', 'No_Operation'))
            return all_actions if all_actions else ['No_Operation']
        except Exception:
            return ['No_Operation']

    def vote(self, raw_texts: List[str]) -> Tuple[str, Dict, List[str], float, float]:
        """Perform majority voting on multiple LLM responses.

        Args:
            raw_texts: List of raw LLM output texts (from multiple calls).

        Returns:
            chosen_text: The raw text corresponding to the winning signature.
            distribution: Dict mapping signature -> count.
            signatures: List of extracted signatures for each call.
            entropy: Shannon entropy of the signature distribution.
            consistency_ratio: Proportion of the most common signature.
        """
        signatures = [self.extract_signature(t) for t in raw_texts]
        counts = Counter(signatures)
        total = len(signatures)

        # Find the winning signature
        chosen_sig, max_count = counts.most_common(1)[0]

        # Compute Shannon entropy
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)

        consistency_ratio = max_count / total

        # Select the raw text that produced the winning signature
        chosen_text = None
        for sig, text in zip(signatures, raw_texts):
            if sig == chosen_sig:
                chosen_text = text
                break

        # Safety fallback: if chosen_text is None (shouldn't happen), use first
        if chosen_text is None:
            chosen_text = raw_texts[0]

        return chosen_text, dict(counts), signatures, round(entropy, 4), round(consistency_ratio, 4)

    def build_fallback_text(self, action_name: str = 'Hold_Position') -> str:
        """Generate a fallback LLM response text for safe default actions.

        Args:
            action_name: Name of the fallback action (e.g., 'Hold_Position', 'Stop', 'No_Operation').

        Returns:
            A formatted text mimicking LLM output structure, with the fallback action.
        """
        return f"Actions:\n    Team-1:\n        <{action_name}()>\n"


class UncertaintyFilter:
    """Unified strategy handler for Raw, Self-Consistency, and UAAF.

    Implements the core UAAF mechanism:
    - Raw: Accept first LLM output directly (no filtering).
    - SC (Self-Consistency): Vote across 3 calls, select majority action.
    - UAAF: Vote across 3-4 calls, if entropy > threshold, trigger fallback action.
    """

    def __init__(self, voter: ActionVoter, config):
        """
        Args:
            voter: An ActionVoter instance.
            config: AgentConfig with strategy-related fields.
        """
        self.voter = voter
        self.config = config

    def filter(self, raw_texts: List[str], strategy: str) -> Tuple[str, Dict]:
        """Apply the appropriate filtering strategy.

        Args:
            raw_texts: List of raw LLM response texts.
            strategy: 'raw', 'sc', or 'uaaf'.

        Returns:
            chosen_text: The selected (or fallback) LLM response text.
            metrics: Dict containing step-level uncertainty metrics.
        """
        fallback_triggered = False

        if strategy == 'raw':
            # Raw strategy: take the first response directly
            chosen_text = raw_texts[0]
            # Still compute metrics for logging purposes
            _, distribution, signatures, entropy, consistency = self.voter.vote(raw_texts)
            chosen_signature = self.voter.extract_signature(chosen_text)

        elif strategy == 'sc':
            # Self-Consistency: majority vote, no filtering
            chosen_text, distribution, signatures, entropy, consistency = self.voter.vote(raw_texts)
            chosen_signature = self.voter.extract_signature(chosen_text)

        elif strategy == 'uaaf':
            # UAAF: vote + entropy-based filtering
            chosen_text, distribution, signatures, entropy, consistency = self.voter.vote(raw_texts)
            chosen_signature = self.voter.extract_signature(chosen_text)

            # Check entropy threshold
            threshold = getattr(self.config, 'entropy_threshold', 1.0)
            if entropy > threshold:
                fallback_action = getattr(self.config, 'fallback_action_name', 'Hold_Position')
                chosen_text = self.voter.build_fallback_text(fallback_action)
                chosen_signature = fallback_action
                fallback_triggered = True
                logger.warning(
                    f"[UAAF] Entropy {entropy} > threshold {threshold}, "
                    f"fallback to {fallback_action}. Distribution: {distribution}"
                )

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'raw', 'sc', or 'uaaf'.")

        metrics = {
            'strategy': strategy,
            'num_calls': len(raw_texts),
            'signatures': signatures,
            'distribution': distribution,
            'chosen_signature': chosen_signature,
            'entropy': entropy,
            'consistency_ratio': consistency,
            'fallback_triggered': fallback_triggered,
        }

        return chosen_text, metrics


class ExperimentTracker:
    """Records step-level and game-level metrics for UAAF experiments.

    Data is saved as:
    - Step-level: JSONL file (one line per decision step)
    - Game-level: JSON summary file (aggregated metrics per game)
    """

    def __init__(self, results_dir: str, task: int, level: int, strategy: str, game_id: int):
        """
        Args:
            results_dir: Directory to save results.
            task: Task number (1, 3, 4, 8).
            level: Difficulty level (1, 2).
            strategy: Strategy name ('raw', 'sc', 'uaaf').
            game_id: Game index within this configuration.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.level = level
        self.strategy = strategy
        self.game_id = game_id
        self.step_records = []
        self._game_started = False
        self._start_time = time.time()

        # File name prefix
        self.prefix = f"t{task}_l{level}_{strategy}_g{game_id}"

    def record_step(self, step: int, metrics: Dict, agent_name: str = 'unknown'):
        """Record a decision step's uncertainty metrics.

        Args:
            step: Decision loop step number.
            metrics: Dict from UncertaintyFilter.filter().
            agent_name: Name of the SubAgent that made this decision.
        """
        record = {
            'step': step,
            'timestamp': time.time(),
            'agent_name': agent_name,
            **metrics
        }
        self.step_records.append(record)

    def finalize_game(self, result: str, total_steps: int = 0,
                      total_tokens_in: int = 0, total_tokens_out: int = 0,
                      total_time: float = 0.0):
        """Finalize game recording: save step-level JSONL and game-level summary JSON.

        Args:
            result: 'win', 'tie', or 'lose'.
            total_steps: Total game steps.
            total_tokens_in: Total input tokens consumed.
            total_tokens_out: Total output tokens consumed.
            total_time: Total game time in seconds.
        """
        # Compute aggregated metrics from step records
        entropies = [r['entropy'] for r in self.step_records]
        consistencies = [r['consistency_ratio'] for r in self.step_records]
        fallbacks = [r['fallback_triggered'] for r in self.step_records]

        # Count action validity: how often chosen_signature matches a real action
        valid_actions = []
        for r in self.step_records:
            sig = r.get('chosen_signature', 'No_Operation')
            valid_actions.append(sig != 'No_Operation')

        summary = {
            'task': self.task,
            'level': self.level,
            'strategy': self.strategy,
            'game_id': self.game_id,
            'result': result,
            'total_steps': total_steps,
            'total_tokens_in': total_tokens_in,
            'total_tokens_out': total_tokens_out,
            'total_time_sec': round(total_time, 2),
            'num_decision_steps': len(self.step_records),
            'avg_entropy': float(np.mean(entropies)) if entropies else 0.0,
            'max_entropy': float(np.max(entropies)) if entropies else 0.0,
            'min_entropy': float(np.min(entropies)) if entropies else 0.0,
            'std_entropy': float(np.std(entropies)) if len(entropies) > 1 else 0.0,
            'avg_consistency_ratio': float(np.mean(consistencies)) if consistencies else 0.0,
            'fallback_ratio': sum(fallbacks) / len(fallbacks) if fallbacks else 0.0,
            'valid_action_ratio': sum(valid_actions) / len(valid_actions) if valid_actions else 0.0,
            'timestamp': time.time(),
        }

        # Save step-level JSONL
        step_path = self.results_dir / f"{self.prefix}_steps.jsonl"
        with open(step_path, 'w', encoding='utf-8') as f:
            for r in self.step_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        # Save game-level summary JSON
        sum_path = self.results_dir / f"{self.prefix}_summary.json"
        with open(sum_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"[UAAF] Game saved: {self.prefix} result={result}, "
                     f"avg_entropy={summary['avg_entropy']:.4f}, "
                     f"fallback_ratio={summary['fallback_ratio']:.4f}")

        # Reset for next game
        self.step_records.clear()
