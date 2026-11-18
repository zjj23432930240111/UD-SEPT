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
UAAF Experiment Batch Runner.

Features:
- One command to run all experiments
- tqdm progress bar with task/strategy/game info
- Checkpoint-based resume: re-run the same command to continue from where it stopped
- Parallel execution support (--parallel N)
- Configurable tasks, levels, strategies, num_games

Usage:
  # Full 4800-game run with 2 parallel processes
  python -m llm_pysc2.bin.run_uaaf_experiments --num_games 200 --parallel 2

  # Resume after interruption (same command, checkpoint auto-detected)
  python -m llm_pysc2.bin.run_uaaf_experiments --num_games 200 --parallel 2

  # Quick test: 5 games, 1 task, 1 strategy
  python -m llm_pysc2.bin.run_uaaf_experiments --tasks 1 --levels 1 --strategies raw --num_games 5

  # Custom checkpoint location
  python -m llm_pysc2.bin.run_uaaf_experiments --checkpoint my_checkpoint.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


CHECKPOINT_FILE = "uaaf_checkpoint.json"


def load_checkpoint(path):
    """Load checkpoint file, return dict of completed games."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"completed": {}, "failed": {}}


def save_checkpoint(path, ckpt):
    """Save checkpoint to file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)


def is_completed(ckpt, task, level, strategy, game_id):
    """Check if a specific game has been completed."""
    key = f"{task}_{level}_{strategy}_{game_id}"
    return key in ckpt["completed"]


def mark_completed(ckpt, task, level, strategy, game_id):
    """Mark a game as completed."""
    key = f"{task}_{level}_{strategy}_{game_id}"
    ckpt["completed"][key] = {
        "timestamp": time.time(),
        "task": task, "level": level, "strategy": strategy, "game_id": game_id
    }


def mark_failed(ckpt, task, level, strategy, game_id, reason="unknown"):
    """Mark a game as failed (will be retried on next run)."""
    key = f"{task}_{level}_{strategy}_{game_id}"
    ckpt["failed"][key] = {
        "timestamp": time.time(),
        "reason": reason,
        "task": task, "level": level, "strategy": strategy, "game_id": game_id
    }


def run_single_game(task, level, strategy, game_id, results_dir, api_key, timeout=1800, max_retries=2):
    """Run a single game via subprocess calling pysc2.bin.agent.

    Args:
        task: Task number (1, 3, 4, 8).
        level: Difficulty level (1, 2).
        strategy: Strategy name ('raw', 'sc', 'uaaf').
        game_id: Game index.
        results_dir: Directory for saving results.
        api_key: DeepSeek API key.
        timeout: Timeout per game in seconds.
        max_retries: Max retries on SC2 websocket failure.

    Returns:
        (success: bool, task, level, strategy, game_id)
    """
    env = os.environ.copy()
    env['UAAF_TASK'] = str(task)
    env['UAAF_LEVEL'] = str(level)
    env['UAAF_STRATEGY'] = strategy
    env['UAAF_GAME_ID'] = str(game_id)
    env['UAAF_RESULTS_DIR'] = results_dir
    env['DEEPSEEK_API_KEY'] = api_key

    map_name = f"pvz_task{task}_level{level}"

    cmd = [
        sys.executable, "-m", "pysc2.bin.agent",
        "--map", map_name,
        "--agent_race", "protoss",
        "--parallel", "1",
        "--game_steps_per_episode", "3000",
        "--agent", "llm_pysc2.bin.experiment_uaaf.MainAgentUAAF"
    ]

    def _kill_sc2():
        """Kill all SC2 processes and wait for cleanup."""
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/IM', 'SC2_x64.exe'],
                               capture_output=True, timeout=10)
            else:
                subprocess.run(['pkill', '-9', 'SC2_x64'],
                               capture_output=True, timeout=10)
        except Exception:
            pass
        time.sleep(5)  # Wait for SC2 to fully release ports

    last_result = None
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd, env=env, timeout=timeout,
                capture_output=True, text=True
            )
            last_result = result
            if result.returncode == 0:
                return (True, task, level, strategy, game_id)
            else:
                # Check if SC2 websocket failure
                err_text = (result.stderr or '') + (result.stdout or '')
                is_sc2_failure = "Failed to connect to the SC2 websocket" in err_text
                if is_sc2_failure and attempt < max_retries:
                    print(f"  [SC2 DEAD] t={task} l={level} s={strategy} g={game_id} "
                          f"(attempt {attempt+1}/{max_retries+1}) - restarting SC2")
                    _kill_sc2()
                    continue
                print(f"  [FAIL] Game t={task} l={level} s={strategy} g={game_id}: "
                      f"returncode={result.returncode}")
                if result.stderr:
                    err_lines = result.stderr.strip().split('\n')
                    for line in err_lines[-3:]:
                        print(f"    {line}")
                return (False, task, level, strategy, game_id)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] Game t={task} l={level} s={strategy} g={game_id}: "
                  f"exceeded {timeout}s timeout")
            return (False, task, level, strategy, game_id)
        except Exception as e:
            print(f"  [ERROR] Game t={task} l={level} s={strategy} g={game_id}: {e}")
            return (False, task, level, strategy, game_id)

    # All retries exhausted for SC2 failure
    if last_result is not None:
        print(f"  [FAIL after {max_retries} retries] Game t={task} l={level} s={strategy} g={game_id}")
    return (False, task, level, strategy, game_id)


def main():
    parser = argparse.ArgumentParser(
        description="Run UAAF experiments across tasks/levels/strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--tasks', default='1,3,4,8',
                        help='Comma-separated task IDs (default: 1,3,4,8)')
    parser.add_argument('--levels', default='1,2',
                        help='Comma-separated level IDs (default: 1,2)')
    parser.add_argument('--strategies', default='raw,sc,uaaf',
                        help='Comma-separated strategies (default: raw,sc,uaaf)')
    parser.add_argument('--num_games', type=int, default=200,
                        help='Number of games per configuration (default: 200)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel game processes (default: 1)')
    parser.add_argument('--checkpoint', default=CHECKPOINT_FILE,
                        help='Checkpoint file path (default: uaaf_checkpoint.json)')
    parser.add_argument('--results_dir', default='uaaf_results',
                        help='Results directory (default: uaaf_results)')
    parser.add_argument('--api_key', default='sk-764a21b2e23d4b848bbc7347b2f413db',
                        help='DeepSeek API key')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per game in seconds (default: 3600)')
    args = parser.parse_args()

    # Parse arguments
    tasks = [int(t) for t in args.tasks.split(',')]
    levels = [int(l) for l in args.levels.split(',')]
    strategies = [s.strip() for s in args.strategies.split(',')]
    game_timeout = args.timeout

    # Load checkpoint
    ckpt = load_checkpoint(args.checkpoint)
    num_completed = len(ckpt["completed"])
    num_failed = len(ckpt["failed"])

    # Build game list
    all_games = []
    for task in tasks:
        for level in levels:
            for strategy in strategies:
                for g in range(args.num_games):
                    all_games.append((task, level, strategy, g))

    total = len(all_games)
    remaining = [g for g in all_games if not is_completed(ckpt, *g)]

    print(f"\n{'='*60}")
    print(f"  UAAF Experiment Batch Runner")
    print(f"{'='*60}")
    print(f"  Tasks: {tasks}")
    print(f"  Levels: {levels}")
    print(f"  Strategies: {strategies}")
    print(f"  Games per config: {args.num_games}")
    print(f"  Total games: {total}")
    num_in_scope = total - len(remaining)
    print(f"  Already completed (checkpoint): {num_completed}")
    print(f"  Already completed (in scope): {num_in_scope}/{total}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Results dir: {args.results_dir}")
    print(f"{'='*60}\n")

    if len(remaining) == 0:
        print("All experiments already completed! No games to run.")
        return

    # Run games
    pbar = tqdm(total=total, initial=total - len(remaining), desc="UAAF Experiments",
                unit="game", bar_format="{l_bar}{bar:30}{r_bar}")

    if args.parallel > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all remaining games
            futures = {}
            for (task, level, strategy, g) in remaining:
                future = executor.submit(
                    run_single_game, task, level, strategy, g,
                    args.results_dir, args.api_key, game_timeout
                )
                futures[future] = (task, level, strategy, g)

            for future in as_completed(futures):
                (task, level, strategy, g) = futures[future]
                try:
                    success, t, l, s, gid = future.result()
                    if success:
                        mark_completed(ckpt, t, l, s, gid)
                        pbar.set_postfix_str(f"t{t}l{l}s{s}g{gid} OK")
                    else:
                        mark_failed(ckpt, t, l, s, gid, "subprocess_failed")
                        pbar.set_postfix_str(f"t{t}l{l}s{s}g{gid} FAIL")
                except Exception as e:
                    mark_failed(ckpt, task, level, strategy, g, str(e))
                    pbar.set_postfix_str(f"t{task}l{level}s{strategy}g{g} ERR")

                save_checkpoint(args.checkpoint, ckpt)
                pbar.update(1)

    else:
        # Sequential execution
        for (task, level, strategy, g) in remaining:
            pbar.set_postfix_str(f"t{task}l{level}s{strategy}g{g}")
            success, t, l, s, gid = run_single_game(
                task, level, strategy, g, args.results_dir, args.api_key, game_timeout
            )
            if success:
                mark_completed(ckpt, t, l, s, gid)
            else:
                mark_failed(ckpt, t, l, s, gid, "subprocess_failed")

            save_checkpoint(args.checkpoint, ckpt)
            pbar.update(1)

    pbar.close()

    # Final summary
    final_completed = len(ckpt["completed"])
    final_failed = len(ckpt["failed"])
    print(f"\n{'='*60}")
    print(f"  Experiment Summary")
    print(f"{'='*60}")
    print(f"  Completed: {final_completed}/{total}")
    print(f"  Failed: {final_failed}")
    print(f"  Remaining: {total - final_completed - final_failed}")
    print(f"  Results saved to: {args.results_dir}")
    print(f"  Checkpoint: {args.checkpoint}")
    if final_failed > 0:
        print(f"\n  Failed games (will be retried on next run):")
        for key, info in list(ckpt["failed"].items())[:10]:
            print(f"    {key}: {info.get('reason', 'unknown')}")
        if len(ckpt["failed"]) > 10:
            print(f"    ... and {len(ckpt['failed']) - 10} more")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
