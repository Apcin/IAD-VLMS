
import os
import re
import copy
import math
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime

from collections import deque

from reward_process import location_reward , type_reward , description_reward

logger = logging.getLogger(__name__)

# ── Reasoning Judge (HTTP client to judge_server.py) ────────────────────────
# Supports multiple judge URLs for parallel scoring (comma-separated)
# e.g. JUDGE_URLS="http://127.0.0.1:5100,http://127.0.0.1:5101"

JUDGE_URLS = [u.strip() for u in os.environ.get("JUDGE_URLS", os.environ.get("JUDGE_URL", "http://127.0.0.1:5100")).split(",")]

# Gemini vision judge URL (for grouped scoring with images)
GEMINI_JUDGE_URL = os.environ.get("GEMINI_JUDGE_URL", "http://127.0.0.1:5200")

# Number of generations per prompt (set from training args, default 4)
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))

# ── Flag-gated reward controls (REPORT2.md §2.6 redesign / §2.7 ablation arms) ──
# All env parsing happens at CALL time (never at import time) so that
#   (1) a crash-restart of the production job with the flags unset is
#       numerically byte-identical to the shipped behavior, and
#   (2) unit tests can toggle the envs between calls without re-importing.
# With every flag unset, accuracy_reward/consistency_reward execute the shipped
# code paths untouched.
#
#   REWARD_GATE_TAGS=1  — verdict gate (§2.6 "minimal one-line version"):
#                         type/location credit (s_type+s_loc)/2 is paid ONLY
#                         when the predicted verdict matches gold.
#   REWARD_BAND_FORM=1  — full §2.6 redesign (implies the gate):
#                         R_answer = 0                                if verdict wrong
#                                  = 1 + b*(q - 1/2)*1[gt=='yes']     if verdict correct
#                         q = (s_type_cont + s_loc)/2, b = 0.1,
#                         s_type_cont = RAW embedding similarity (the
#                         threshold staircase is bypassed). Overrides
#                         GATE/COMPONENTS for the answer term; the format
#                         term is unchanged by it (§2.6 "R_format unchanged").
#   REWARD_COMPONENTS   — comma list, default "verdict,format,type,loc"
#                         (§2.7 arm switches). Each sub-signal is multiplied
#                         by its switch, so an omitted component contributes
#                         exactly 0 (e.g. "verdict,format" = S-style arm).

_REWARD_BAND_B = 0.1  # §2.6 band half-width b
_DEFAULT_COMPONENTS = frozenset(("verdict", "format", "type", "loc"))


def _reward_flags():
    """Parse the reward-control envs at call time.

    Returns (band, gate, components). band implies gate (§2.6 full form
    contains the verdict gate by construction)."""
    band = os.environ.get("REWARD_BAND_FORM", "") == "1"
    gate = band or os.environ.get("REWARD_GATE_TAGS", "") == "1"
    comps_raw = os.environ.get("REWARD_COMPONENTS")
    if comps_raw is None:
        comps = _DEFAULT_COMPONENTS
    else:
        comps = frozenset(c.strip() for c in comps_raw.split(",") if c.strip())
    return band, gate, comps


def _raw_type_similarity(predicted, actual):
    """§2.6 band form: s_type_cont = the RAW embedding similarity.

    Reuses the exact embedding-similarity call the shipped staircase uses
    (AnomalyRewardCalculator normalization + NomicEmbeddingModel.similarity)
    but skips the _threshold_map discretization. Empty inputs score 0.0,
    mirroring compute_reward's guard."""
    if not predicted or not actual:
        return 0.0
    calc = type_reward.AnomalyRewardCalculator()
    return calc.nomic.similarity(calc._normalize_text(predicted), calc._normalize_text(actual))


def _accuracy_reward_flagged(content, sol, band, gate, comps):
    """Flag-aware answer-term reward for ONE completion (REPORT2.md §2.6/§2.7).

    Reached only when at least one control flag is active; the all-unset
    production path runs the shipped loop body in accuracy_reward untouched.
    Mirrors the shipped structure (tag extraction, /2 normalization,
    except-pass semantics) so the only differences are the flag effects."""
    reward = 0.0
    try:
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        gt = ground_truth.lower()

        answer_match = re.search(r'<answer>(.*?)</answer>', content)
        ans = answer_match.group(1).strip().lower() if answer_match else None
        verdict_correct = ans is not None and ans == gt

        if band:
            # §2.6 full redesign. Precedence: overrides GATE/COMPONENTS for
            # the answer term. Minimal-faithful reading of the report: a
            # missing/malformed tag contributes sub-signal 0 (the shipped
            # "no tag -> no credit" convention), and gt outside {yes,no}
            # scores 0 exactly as the shipped code does.
            if verdict_correct and gt == "yes":
                s_type_cont = 0.0
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                if gpt_type_match and gt_type_match:
                    s_type_cont = _raw_type_similarity(
                        gpt_type_match.group(1).strip().lower(),
                        gt_type_match.group(1).strip().lower())

                s_loc = 0.0
                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                if gpt_location_match and gt_location_match:
                    s_loc = location_reward.map_location_to_region(
                        gpt_location_match.group(1).strip().lower(),
                        gt_location_match.group(1).strip().lower())

                q = (s_type_cont + s_loc) / 2.0
                reward = 1.0 + _REWARD_BAND_B * (q - 0.5)
            elif verdict_correct and gt == "no":
                reward = 1.0  # 1[gt=='yes'] zeroes the band term for gt=no
            # verdict wrong (or gt not yes/no): reward stays 0.0
            return reward

        # ── gate / component-switch path (§2.6 minimal version, §2.7 arms) ──
        sw_verdict = 1.0 if "verdict" in comps else 0.0
        sw_type = 1.0 if "type" in comps else 0.0
        sw_loc = 1.0 if "loc" in comps else 0.0

        if gt == "no":
            # Shipped code pays no tag credit on gt=no, so the gate is vacuous
            # here; only the verdict switch applies.
            if ans == "no":
                reward = sw_verdict * 1.0

        elif gt == "yes":
            total_reward = 0.0
            max_reward = 2.0

            gpt_type_match = re.search(r'<type>(.*?)</type>', content)
            gt_type_match = re.search(r'<type>(.*?)</type>', sol)
            if gpt_type_match and gt_type_match:
                type_calculator = type_reward.AnomalyRewardCalculator()
                type_score = type_calculator.compute_reward(
                    gpt_type_match.group(1).strip().lower(),
                    gt_type_match.group(1).strip().lower())
                total_reward += sw_type * type_score

            gpt_location_match = re.search(r'<location>(.*?)</location>', content)
            gt_location_match = re.search(r'<location>(.*?)</location>', sol)
            if gpt_location_match and gt_location_match:
                location_score = location_reward.map_location_to_region(
                    gpt_location_match.group(1).strip().lower(),
                    gt_location_match.group(1).strip().lower())
                total_reward += sw_loc * location_score

            tag_credit = total_reward / max_reward
            if gate and not verdict_correct:
                # §2.6 minimal version: (s_type+s_loc)/2 is paid only when the
                # predicted verdict matches gold — kills the hedging exploit.
                tag_credit = 0.0
            reward = tag_credit
            if ans == "yes":
                reward += sw_verdict * 1.0

    except Exception:
        pass  # mirror the shipped except-pass semantics
    return reward


def _call_judge_single_server(url, items):
    """Send a batch to a single judge server. Returns list of scores."""
    endpoint = f"{url}/batch"
    data = json.dumps({"items": items}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("scores", [0.5] * len(items))
    except Exception as e:
        logger.warning(f"[JUDGE] HTTP call to {url} failed: {e}")
        return [0.5] * len(items)


def _call_judge_batch(items):
    """Split items across available judge servers and score in parallel."""
    if not items:
        return []
    n_servers = len(JUDGE_URLS)
    if n_servers == 1:
        return _call_judge_single_server(JUDGE_URLS[0], items)

    # Split items round-robin across servers
    chunks = [[] for _ in range(n_servers)]
    index_map = [[] for _ in range(n_servers)]  # track original indices
    for i, item in enumerate(items):
        server_idx = i % n_servers
        chunks[server_idx].append(item)
        index_map[server_idx].append(i)

    # Call all servers in parallel
    with ThreadPoolExecutor(max_workers=n_servers) as executor:
        futures = []
        for server_idx in range(n_servers):
            if chunks[server_idx]:
                futures.append((server_idx, executor.submit(_call_judge_single_server, JUDGE_URLS[server_idx], chunks[server_idx])))

    # Reassemble scores in original order
    scores = [0.5] * len(items)
    for server_idx, future in futures:
        chunk_scores = future.result()
        for local_i, score in enumerate(chunk_scores):
            scores[index_map[server_idx][local_i]] = score

    return scores


def reasoning_reward(completions, solution, **kwargs):
    """Judge the quality of <think> reasoning via external judge server."""
    contents = [completion[0]["content"] for completion in completions]
    product_list = kwargs.get("product", [])
    items = []
    valid_indices = []

    for i, (content, sol) in enumerate(zip(contents, solution)):
        # Extract think block
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
        if not think_match or len(think_match.group(1).strip()) < 20:
            continue

        # Extract GT info from solution
        sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol)
        gt_label = sol_answer_match.group(1).strip().lower() if sol_answer_match else "no"
        gt_type_match = re.search(r'<type>(.*?)</type>', sol)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else ""
        gt_loc_match = re.search(r'<location>(.*?)</location>', sol)
        gt_location = gt_loc_match.group(1).strip() if gt_loc_match else ""

        # Extract model answer
        model_ans_match = re.search(r'<answer>(.*?)</answer>', content)
        model_answer = model_ans_match.group(1).strip() if model_ans_match else ""

        product = product_list[i] if i < len(product_list) else "unknown"

        items.append({
            "think_text": think_match.group(1).strip(),
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_location": gt_location,
            "product": product,
            "model_answer": model_answer,
        })
        valid_indices.append(i)

    # Batch call to judge
    scores = _call_judge_batch(items) if items else []

    # Build rewards array: 0.0 for invalid, scored for valid
    rewards = [0.0] * len(contents)
    for idx, score in zip(valid_indices, scores):
        rewards[idx] = score

    return rewards

GEMINI_CLIENT_MAX_RETRIES = 3
GEMINI_CLIENT_TIMEOUT = 180  # server-side retry cascade can take ~120s

def _call_gemini_judge_grouped(groups):
    """Send grouped completions to Gemini vision judge server with retries."""
    endpoint = f"{GEMINI_JUDGE_URL}/batch_grouped"
    data = json.dumps({"groups": groups}).encode("utf-8")

    for attempt in range(GEMINI_CLIENT_MAX_RETRIES):
        try:
            req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=GEMINI_CLIENT_TIMEOUT) as resp:
                raw = resp.read()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error(f"[GEMINI-JUDGE] JSON decode failed, raw response: {raw[:500]}")
                    continue
                scores = result.get("group_scores", [])
                if not scores:
                    logger.warning(f"[GEMINI-JUDGE] Empty group_scores, raw response: {raw[:500]}")
                return scores
        except Exception as e:
            logger.warning(f"[GEMINI-JUDGE] HTTP call failed (attempt {attempt+1}/{GEMINI_CLIENT_MAX_RETRIES}): {e}")
            if attempt < GEMINI_CLIENT_MAX_RETRIES - 1:
                import time as _time
                _time.sleep(1)
    logger.error(f"[GEMINI-JUDGE] All {GEMINI_CLIENT_MAX_RETRIES} HTTP retries exhausted")
    return None


def reasoning_reward_gemini(completions, solution, **kwargs):
    """Judge reasoning quality via Gemini with vision — grouped per prompt."""
    contents = [completion[0]["content"] for completion in completions]
    product_list = kwargs.get("product", [])
    image_path_list = kwargs.get("image_path", [])

    n = len(contents)
    num_gen = NUM_GENERATIONS

    # Group completions by prompt (every num_gen consecutive entries share a prompt)
    num_prompts = n // num_gen
    groups = []
    group_valid_indices = []  # track which completions in each group are valid

    for g in range(num_prompts):
        start = g * num_gen
        end = start + num_gen

        # Extract GT info from the first solution in the group (all same)
        sol = solution[start]
        sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol)
        gt_label = sol_answer_match.group(1).strip().lower() if sol_answer_match else "no"
        gt_type_match = re.search(r'<type>(.*?)</type>', sol)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else ""
        gt_loc_match = re.search(r'<location>(.*?)</location>', sol)
        gt_location = gt_loc_match.group(1).strip() if gt_loc_match else ""

        product = product_list[start] if start < len(product_list) else "unknown"
        image_path = image_path_list[start] if start < len(image_path_list) else ""

        # Build completions for this group
        group_completions = []
        valid_mask = []
        for i in range(start, end):
            content = contents[i]
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
            model_ans_match = re.search(r'<answer>(.*?)</answer>', content)
            model_answer = model_ans_match.group(1).strip() if model_ans_match else ""

            if think_match and len(think_match.group(1).strip()) >= 20:
                group_completions.append({
                    "think_text": think_match.group(1).strip(),
                    "model_answer": model_answer,
                })
                valid_mask.append(True)
            else:
                # Still include a placeholder so indices align
                group_completions.append({
                    "think_text": "(no reasoning provided)",
                    "model_answer": model_answer,
                })
                valid_mask.append(False)

        # Derive GT mask path for RealIAD: same filename, .jpg -> .png
        gt_mask_path = ""
        if gt_label == "yes" and image_path and image_path.lower().endswith(".jpg"):
            candidate = image_path[:-4] + ".png"
            if os.path.isfile(candidate):
                gt_mask_path = candidate

        groups.append({
            "image_path": image_path,
            "gt_mask_path": gt_mask_path,
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_location": gt_location,
            "product": product,
            "completions": group_completions,
        })
        group_valid_indices.append(valid_mask)

    # Call Gemini judge with all groups
    group_scores = _call_gemini_judge_grouped(groups)

    # Build flat rewards array
    rewards = [0.0] * n
    if group_scores and len(group_scores) == num_prompts:
        for g in range(num_prompts):
            start = g * num_gen
            scores = group_scores[g]
            valid_mask = group_valid_indices[g]
            for j in range(num_gen):
                if j < len(scores) and valid_mask[j]:
                    rewards[start + j] = scores[j]
                # invalid completions stay at 0.0
    else:
        # Fallback: 0.5 for all valid completions
        logger.warning(f"[GEMINI-JUDGE] Fallback — unexpected result: group_scores={group_scores}, expected {num_prompts} groups")
        for g in range(num_prompts):
            start = g * num_gen
            for j in range(num_gen):
                idx = start + j
                content = contents[idx]
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                if think_match and len(think_match.group(1).strip()) >= 20:
                    rewards[idx] = 0.5

    return rewards


def _call_gemini_judge_ranked(groups):
    """Send grouped completions to the Gemini RANK endpoint with HTTP retries.
    Returns list (one per group) of dense-rank lists (1=best, ties allowed) or None per group."""
    endpoint = f"{GEMINI_JUDGE_URL}/batch_ranked"
    data = json.dumps({"groups": groups}).encode("utf-8")

    for attempt in range(GEMINI_CLIENT_MAX_RETRIES):
        try:
            req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=GEMINI_CLIENT_TIMEOUT) as resp:
                raw = resp.read()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error(f"[GEMINI-RANK] JSON decode failed, raw response: {raw[:500]}")
                    continue
                rankings = result.get("group_rankings", [])
                if not rankings:
                    logger.warning(f"[GEMINI-RANK] Empty group_rankings, raw response: {raw[:500]}")
                return rankings
        except Exception as e:
            logger.warning(f"[GEMINI-RANK] HTTP call failed (attempt {attempt+1}/{GEMINI_CLIENT_MAX_RETRIES}): {e}")
            if attempt < GEMINI_CLIENT_MAX_RETRIES - 1:
                import time as _time
                _time.sleep(1)
    logger.error(f"[GEMINI-RANK] All {GEMINI_CLIENT_MAX_RETRIES} HTTP retries exhausted")
    return None


def _dense_ranks_to_rewards(dense_ranks):
    """Hardcoded dense-rank -> reward in [0,1]: best (rank 1) -> 1.0, worst -> 0.0, ties share.
    reward = (max_rank - rank) / (max_rank - min_rank). If every completion ties (single rank
    level), returns None -> no gradient (caller assigns a neutral constant)."""
    lo, hi = min(dense_ranks), max(dense_ranks)
    if hi == lo:
        return None
    return [(hi - r) / (hi - lo) for r in dense_ranks]


def reasoning_reward_gemini_rank(completions, solution, **kwargs):
    """Rank-based reasoning reward. Gemini RANKS the G completions per prompt (ties allowed,
    dense ranks); rewards are hardcoded from the rank order, not from Gemini scores.
    Invalid completions (no/short <think>) are forced to 0.0. A group where the judge fails or
    calls everything equal contributes no within-group gradient (neutral constant)."""
    contents = [completion[0]["content"] for completion in completions]
    product_list = kwargs.get("product", [])
    image_path_list = kwargs.get("image_path", [])

    n = len(contents)
    num_gen = NUM_GENERATIONS
    num_prompts = n // num_gen

    groups = []
    group_valid_indices = []
    for g in range(num_prompts):
        start = g * num_gen
        end = start + num_gen

        sol = solution[start]
        sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol)
        gt_label = sol_answer_match.group(1).strip().lower() if sol_answer_match else "no"
        gt_type_match = re.search(r'<type>(.*?)</type>', sol)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else ""
        gt_loc_match = re.search(r'<location>(.*?)</location>', sol)
        gt_location = gt_loc_match.group(1).strip() if gt_loc_match else ""

        product = product_list[start] if start < len(product_list) else "unknown"
        image_path = image_path_list[start] if start < len(image_path_list) else ""

        group_completions = []
        valid_mask = []
        for i in range(start, end):
            content = contents[i]
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
            model_ans_match = re.search(r'<answer>(.*?)</answer>', content)
            model_answer = model_ans_match.group(1).strip() if model_ans_match else ""
            if think_match and len(think_match.group(1).strip()) >= 20:
                group_completions.append({"think_text": think_match.group(1).strip(), "model_answer": model_answer})
                valid_mask.append(True)
            else:
                group_completions.append({"think_text": "(no reasoning provided)", "model_answer": model_answer})
                valid_mask.append(False)

        gt_mask_path = ""
        if gt_label == "yes" and image_path and image_path.lower().endswith(".jpg"):
            candidate = image_path[:-4] + ".png"
            if os.path.isfile(candidate):
                gt_mask_path = candidate

        groups.append({
            "image_path": image_path,
            "gt_mask_path": gt_mask_path,
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_location": gt_location,
            "product": product,
            "completions": group_completions,
        })
        group_valid_indices.append(valid_mask)

    group_rankings = _call_gemini_judge_ranked(groups)

    rewards = [0.0] * n
    ok = bool(group_rankings) and len(group_rankings) == num_prompts
    for g in range(num_prompts):
        start = g * num_gen
        valid_mask = group_valid_indices[g]
        dense = group_rankings[g] if ok else None
        vals = _dense_ranks_to_rewards(dense) if (dense and len(dense) == num_gen) else None
        for j in range(num_gen):
            if not valid_mask[j]:
                rewards[start + j] = 0.0            # no/short reasoning -> always worst
            elif vals is None:
                rewards[start + j] = 0.5            # judge failed or all-equal -> neutral, no fake gradient
            else:
                rewards[start + j] = vals[j]

    # optional debug dump (JSONL) for inspecting real completions + ranks during a test run
    if os.environ.get("RANK_DEBUG_LOG"):
        try:
            with open(os.environ["RANK_DEBUG_LOG"], "a") as _f:
                for g in range(num_prompts):
                    start = g * num_gen
                    _f.write(json.dumps({
                        "product": groups[g]["product"],
                        "gt_label": groups[g]["gt_label"],
                        "gt_type": groups[g]["gt_type"],
                        "dense": group_rankings[g] if ok else None,
                        "rewards": [round(rewards[start + j], 3) for j in range(num_gen)],
                        "valid": group_valid_indices[g],
                        "answers": [groups[g]["completions"][j].get("model_answer", "") for j in range(num_gen)],
                        "think_preview": [groups[g]["completions"][j].get("think_text", "")[:200] for j in range(num_gen)],
                    }) + "\n")
        except Exception:
            pass

    return rewards


def consistency_reward(completions, solution, **kwargs):
    # env-gated smoke-test visibility: dump first completions per call
    if os.environ.get("GRPO_COMPLETION_LOG"):
        try:
            with open(os.environ["GRPO_COMPLETION_LOG"], "a") as _f:
                for _c, _s in list(zip(completions, solution))[:2]:
                    _f.write("==== COMPLETION ====\n" + _c[0]["content"][:600] + "\n---- GT ----\n" + str(_s)[:200] + "\n")
        except Exception:
            pass
    pattern_no = r"^(?!.*<location>)(?!.*<type>).*<think>.*?</think>\s*<answer>.*?</answer>.*$"  # normal pattern    
    pattern_yes = r".*<think>.*?</think>\s*<location>.*?</location>\s*<type>.*?</type>\s*<answer>.*?</answer>.*"  # abnormal pattern
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # yes or no
        if ground_truth.lower() == "yes":
            # yes --> pattern_yes
            match_result = re.fullmatch(pattern_yes, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
        elif ground_truth.lower() == "no":
            # no --> pattern_no
            match_result = re.fullmatch(pattern_no, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
        else:
            rewards.append(0.0)

    # REPORT2.md §2.7 component switch for 'format', parsed at CALL time.
    # With REWARD_COMPONENTS unset the switch is on and the shipped rewards
    # list is returned untouched (crash-restart no-op). REWARD_BAND_FORM
    # deliberately does NOT alter this term (§2.6: "R_format unchanged");
    # only omitting 'format' from REWARD_COMPONENTS zeroes it.
    _, _, _comps = _reward_flags()
    if "format" not in _comps:
        rewards = [r * 0.0 for r in rewards]
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """answer, location, type"""
    # REPORT2.md §2.6/§2.7 flag-gated controls, parsed at CALL time (never at
    # import time). With all flags unset — the production default — the shipped
    # loop body below executes untouched, so a crash-restart with no envs set
    # behaves byte-identically.
    _band, _gate, _comps = _reward_flags()
    _flags_active = _band or _gate or _comps != _DEFAULT_COMPONENTS
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        if _flags_active:
            rewards.append(_accuracy_reward_flagged(content, sol, _band, _gate, _comps))
            continue
        reward = 0.0
        try:

            #resolve ground_truth
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            gt = ground_truth.lower()

            # gt == "no"
            if gt == "no":
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match:
                    ans = content_match.group(1).strip().lower()
                    if ans == "no":
                        reward = 1.0
            
            # gt == "yes"
            elif gt == "yes":
                total_reward = 0.0
                max_reward = 2.0 
                
                # ----- type reward -----
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                if gpt_type_match and gt_type_match:
                    gpt_type = gpt_type_match.group(1).strip().lower()
                    gt_type = gt_type_match.group(1).strip().lower()
                    type_calculator = type_reward.AnomalyRewardCalculator()
                    type_score = type_calculator.compute_reward(gpt_type, gt_type)
                    total_reward += type_score
                else:
                    pass # reward += 0

                # location reward
                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                if gpt_location_match and gt_location_match:
                    gpt_location = gpt_location_match.group(1).strip().lower()
                    gt_location = gt_location_match.group(1).strip().lower()
                    location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                    total_reward += location_score
                else:
                    pass

                reward = total_reward / max_reward # 0~1

                # answer reward
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match:
                    ans = answer_match.group(1).strip().lower()
                    if ans == "yes":
                        reward += 1.0
                    
                
                
        except Exception:
            pass  # reward = 0   
        rewards.append(reward)            
    return rewards





def consistency_reward_cot(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        

        if model_answer != ground_truth:

            rewards.append(0.0)
            continue
        

        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                consistency_reward = 1.0  
            elif tag_count == 2:
                consistency_reward = 0.7 
            elif tag_count == 1:
                consistency_reward = 0.4 
            else:
                consistency_reward = 0.0 
        else:
            consistency_reward = 0.0
        
        rewards.append(consistency_reward)
    
    return rewards

def format_consistency_reward_cot(completions, solution, **kwargs):
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        
        if model_answer != ground_truth:
            rewards.append(0.0)
            continue
        
        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            format_consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                format_consistency_reward = 1.0
            elif tag_count == 2:
                format_consistency_reward = 0.7
            elif tag_count == 1:
                format_consistency_reward = 0.4
            else:  # tag_count == 0
                format_consistency_reward = 0.0
        else:

            format_consistency_reward = 0.0
        
        rewards.append(format_consistency_reward)
    
    return rewards


def accuracy_reward_cot_wo_type(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
  
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 


                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                gpt_location = gpt_location_match.group(1).strip().lower()
                gt_location = gt_location_match.group(1).strip().lower()
                location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                total_reward += location_score

                reward = total_reward / max_reward


                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                    
                  
                
        except Exception:
            pass 
        rewards.append(reward)            
    return rewards

def accuracy_reward_cot_wo_location(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0
            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 
                
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                gpt_type = gpt_type_match.group(1).strip().lower()
                gt_type = gt_type_match.group(1).strip().lower()
                type_calculator = type_reward.AnomalyRewardCalculator()
                type_score = type_calculator.compute_reward(gpt_type, gt_type)
                total_reward += type_score

                reward = total_reward / max_reward
                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                
        except Exception:
            pass    
        rewards.append(reward)            
    return rewards

def format_reward_cot_base(completions, solution, **kwargs):
    pattern = r".*<think>.*?</think><answer>.*?</answer>.*" 
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        match_result = re.fullmatch(pattern, content, re.DOTALL)
        rewards.append(1.0 if match_result else 0.0)
    return rewards

def accuracy_reward_cot_base(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                answer_match = re.search(r'<answer>(.*?)</answer>', content)

                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0

        except Exception:
            pass  
        rewards.append(reward)            
    return rewards

def wo_format(completions, solution, **kwargs):
    rewards = 0
    return rewards
