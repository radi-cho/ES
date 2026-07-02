import re
import jax
import jax.numpy as jnp
import numpy as np
import reasoning_gym

from pathlib import Path
from datasets import load_dataset
from functools import lru_cache
from math_verify import parse as mv_parse, verify as mv_verify

@lru_cache(maxsize=2048)
def _parse_gold_cached(gold_text: str):
    return mv_parse(gold_text)

# ----------------------------
# ReasoningGym dataset names
# ----------------------------
RG_DATASETS = [
    "ab", "acre", "advanced_geometry", "aiw", "arc_1d", "arc_agi", "base_conversion", "basic_arithmetic",
    "bf", "binary_alternation", "binary_matrix", "bitwise_arithmetic", "boxnet", "caesar_cipher",
    "calendar_arithmetic", "chain_sum", "circuit_logic", "codeio", "coin_flip", "color_cube_rotation",
    "complex_arithmetic", "composite", "count_bits", "count_primes", "countdown", "course_schedule",
    "cryptarithm", "decimal_arithmetic", "decimal_chain_sum", "dice", "emoji_mystery", "family_relationships",
    "figlet_font", "fraction_simplification", "futoshiki", "game_of_life", "game_of_life_halting", "gcd",
    "graph_color", "group_anagrams", "gsm_symbolic", "intermediate_integration", "isomorphic_strings",
    "jugs", "kakurasu", "knight_swap", "knights_knaves", "largest_island", "lcm", "leg_counting",
    "letter_counting", "letter_jumble", "list_functions", "mahjong_puzzle", "manipulate_matrix", "maze",
    "mini_sudoku", "modulo_grid", "n_queens", "needle_haystack", "number_filtering", "number_format",
    "number_sequence", "number_sorting", "palindrome_generation", "palindrome_partitioning",
    "polynomial_equations", "polynomial_multiplication", "pool_matrix", "power_function",
    "prime_factorization", "products", "propositional_logic", "puzzle24", "quantum_lock", "ransom_note",
    "rearc", "rectangle_count", "rotate_matrix", "rotten_oranges", "rubiks_cube", "rush_hour",
    "self_reference", "sentence_reordering", "shortest_path", "simple_equations", "simple_geometry",
    "simple_integration", "sokoban", "spell_backward", "spiral_matrix", "string_insertion",
    "string_manipulation", "string_splitting", "string_synthesis", "sudoku", "survo", "syllogism",
    "time_intervals", "tower_of_hanoi", "tsumego", "word_ladder", "word_sequence_reversal",
    "word_sorting", "zebra_puzzles",
]


# ----------------------------
# Helpers
# ----------------------------
def safe_decode(tokens, tokenizer):
    """Decode up to the first 0 token, else full sequence."""
    try:
        stop_tokens = np.flatnonzero(tokens == 0)
        if stop_tokens.size > 0:
            tokens = tokens[:stop_tokens[0]]
        return tokenizer.decode(tokens)
    except BaseException:
        return ""


def decode_tokens(tokens, tokenizer):
    """Decode a token sequence without RWKV-specific EOS truncation."""
    try:
        return tokenizer.decode(np.asarray(tokens, dtype=np.int32).tolist())
    except BaseException:
        return ""


def _generation_stop_ids(tokenizer):
    if hasattr(tokenizer, "generation_stop_ids"):
        return tokenizer.generation_stop_ids()
    stop = {0}
    inner = getattr(tokenizer, "tok", None)
    if inner is not None:
        if inner.eos_token_id is not None:
            stop.add(int(inner.eos_token_id))
        if inner.pad_token_id is not None:
            stop.add(int(inner.pad_token_id))
    return stop


def decode_generation_tokens(tokens, tokenizer):
    """Decode generated tokens, stripping trailing pad markers only."""
    try:
        stop_ids = _generation_stop_ids(tokenizer)
        arr = np.asarray(tokens, dtype=np.int32)
        while arr.size > 0 and int(arr[-1]) in stop_ids:
            arr = arr[:-1]
        return tokenizer.decode(arr.tolist())
    except BaseException:
        return ""


def _extract_countdown_chat_generation(context, tok_seq, prompt_len, tokenizer):
    """Extract model continuation; prefer string split to avoid token-boundary drift."""
    gen_tokens = np.asarray(tok_seq[prompt_len:], dtype=np.int32)
    stop_ids = _generation_stop_ids(tokenizer)
    while gen_tokens.size > 0 and int(gen_tokens[-1]) in stop_ids:
        gen_tokens = gen_tokens[:-1]

    prompt_tokens = np.asarray(tok_seq[:prompt_len], dtype=np.int32)
    full = decode_tokens(np.concatenate([prompt_tokens, gen_tokens]), tokenizer)
    if full.startswith(context):
        generation_from_text = full[len(context):]
        if generation_from_text.strip():
            return generation_from_text
    return decode_tokens(gen_tokens, tokenizer)


def get_padded_prompt(single_prompt, generation_length):
    single_prompt = single_prompt[:generation_length]
    return single_prompt + [0] * (generation_length - len(single_prompt))


def strip_thoughts(txt: str) -> str:
    boundary_markers = ("Now produce output:", "###")
    for marker in boundary_markers:
        if marker in txt:
            txt = txt.split(marker, 1)[1]
            break
    i = txt.find("</think>")
    if i != -1:
        txt = txt[i + len("</think>"):]
    answer_marker = "Answer:"
    if answer_marker in txt:
        txt = txt.split(answer_marker, 1)[1]
        # Take the first non-empty line after the marker to avoid explanations
        for line in txt.splitlines():
            line = line.strip()
            if line:
                return line
        return ""
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(("User:", "Assistant:")):
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        if any(ch.isdigit() for ch in line):
            return line
    return txt


def normalize_answer_text(text: str) -> str:
    """Normalize unicode fractions/symbols to ASCII for parsing."""
    if not text:
        return text
    replacements = {
        "⅓": "1/3",
        "⅔": "2/3",
        "¼": "1/4",
        "½": "1/2",
        "¾": "3/4",
        "⅕": "1/5",
        "⅖": "2/5",
        "⅗": "3/5",
        "⅘": "4/5",
        "⅙": "1/6",
        "⅚": "5/6",
        "⅛": "1/8",
        "⅜": "3/8",
        "⅝": "5/8",
        "⅞": "7/8",
        "÷": "/",
        "×": "*",
        "⌊": "",
        "⌋": "",
        "=": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.strip()


def make_rg_prompt(question: str) -> str:
    return (
        "User: You are a helpful assistant. You first think about the reasoning process "
        f"in your mind and then provide the user with the answer. Question: {question}.\n"
        "Assistant:\n"
        "Return ONLY the final expression.\n"
        "Do not repeat the question.\n"
        "Use all the numbers exactly once.\n"
        "Start immediately with:\n"
        "###\n"
        "Now produce output:\n"
    )


# ----------------------------
# Base Task
# ----------------------------
class BanditTask:
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        self.encoding_tokenizer = encoding_tokenizer
        self.decoding_tokenizer = decoding_tokenizer
        self.max_num_steps = max_num_steps

    def __len__(self):
        raise NotImplementedError("len not implemented")

    def get_input(self, indices):
        raise NotImplementedError("get_input not implemented")

    def get_batch_fitness(self, indices, full_generations):
        raise NotImplementedError("get_batch_fitness not implemented")


# ----------------------------
# Toy tasks
# ----------------------------
class ToyTask(BanditTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps, single_fitness):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self._batch_fitness = jax.jit(jax.vmap(single_fitness))

    def __len__(self):
        return -1

    def get_input(self, indices):
        return jnp.zeros(indices.shape + (self.max_num_steps,), dtype=jnp.int32)

    def get_batch_fitness(self, indices, full_generations):
        # skip the first token (0) for scoring
        return self._batch_fitness(full_generations[:, 1:])


class FastZero(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(
            encoding_tokenizer,
            decoding_tokenizer,
            max_num_steps,
            lambda generated_tokens: -jax.numpy.nonzero(
                generated_tokens == 0,
                size=1,
                fill_value=generated_tokens.shape[0] * 2
            )[0][0].astype(jnp.float32),
        )


class UniqueTok(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(
            encoding_tokenizer,
            decoding_tokenizer,
            max_num_steps,
            lambda generated_tokens: jnp.sum(
                jnp.where(
                    jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts == 0,
                    0, 1
                )
            ).astype(jnp.float32),
        )


class RepTok(ToyTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(
            encoding_tokenizer,
            decoding_tokenizer,
            max_num_steps,
            lambda generated_tokens: jnp.max(
                jnp.unique_counts(generated_tokens, size=generated_tokens.shape[0]).counts
            ).astype(jnp.float32),
        )


class Digits(BanditTask):
    def __len__(self):
        return -1

    def get_input(self, indices):
        return jnp.zeros(indices.shape + (self.max_num_steps,), dtype=jnp.int32)

    def get_batch_fitness(self, indices, full_generations):
        numpy_tokens = np.array(full_generations)
        num_digits = [
            sum(c.isdigit() for c in safe_decode(numpy_tokens[i, 1:], self.decoding_tokenizer))
            for i in range(numpy_tokens.shape[0])
        ]
        return jnp.array(num_digits, dtype=jnp.float32)


# ----------------------------
# GSM8K (regex-based simple scorer)
# ----------------------------
def make_conversation(example):
    return {"prompt": f"User: {example['question']}\n\nAssistant: <think"}


def extract_predicted_answer(text):
    regex_pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore = [",", r"\$", r"(?s).*#### ", r"\.$"]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()
        for rgx in regexes_to_ignore:
            text = re.sub(rgx, "", text)
        return text
    else:
        # No match
        return None


def extract_ground_truth(text):
    return text.split("####")[-1].strip()


def check_accuracy(generated_ans, solution):
    ground_truth_answer = extract_ground_truth(solution)
    model_answer = extract_predicted_answer(generated_ans.strip())
    return 1.0 if (model_answer == ground_truth_answer) else 0.0


def single_fitness(generated_answer, true_answer, i):
    find_idx = generated_answer.find("</think>")
    if find_idx == -1:
        return 0.0
    true_idx = find_idx + len("</think>")
    return check_accuracy(generated_answer[true_idx:], true_answer)


class GSM8KTrain(BanditTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset("openai/gsm8k", "main", split="train").map(make_conversation)

    def __len__(self):
        return len(self.dataset)

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.encoding_tokenizer.encode(
                    self.dataset[i.item() % len(self.dataset)]["prompt"]
                ),
                self.max_num_steps,
            )
            for i in indices
        ])

    def get_batch_fitness(self, indices, full_generations):
        batch_answers = [self.dataset[i.item() % len(self.dataset)]["answer"] for i in indices]
        rewards = []
        np_full = np.array(full_generations)
        for i, tok_seq in enumerate(np_full):
            gen_ans = safe_decode(tok_seq, self.decoding_tokenizer)
            reward = 0.0 if len(gen_ans) == 0 else single_fitness(gen_ans, batch_answers[i], i)
            rewards.append(reward)
        return jnp.array(rewards, dtype=jnp.float32)


class GSM8KTest(GSM8KTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset("openai/gsm8k", "main", split="test").map(make_conversation)


class GSM8KSFT(GSM8KTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset("openai/gsm8k", "main", split="train").map(make_conversation)

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.encoding_tokenizer.encode(
                    self.dataset[i.item() % len(self.dataset)]["prompt"] +
                    self.dataset[i.item() % len(self.dataset)]["answer"]
                ),
                self.max_num_steps,
            )
            for i in indices
        ])

# ----------------------------
# ReasoningGym tasks
# ----------------------------
class ReasoningGymTrain(BanditTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=10000, seed=42, dataset_type="countdown"):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset_size = dataset_size
        self.dataset = reasoning_gym.create_dataset(dataset_type, size=dataset_size, seed=seed)

    def __len__(self):
        return self.dataset_size

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.decoding_tokenizer.encode(
                    make_rg_prompt(self.dataset[i.item() % len(self.dataset)]["question"])
                ),
                self.max_num_steps,
            )
            for i in indices
        ])

    def get_batch_fitness(self, indices, full_generations):
        rewards = []
        np_full = np.array(full_generations)
        L = len(self.dataset)
        for idx, tok_seq in zip(indices, np_full):
            gen_ans = safe_decode(tok_seq, self.decoding_tokenizer)
            gen_ans = strip_thoughts(gen_ans).strip()
            gen_ans = normalize_answer_text(gen_ans)
            if len(gen_ans) == 0:
                reward = 0.0
            else:
                entry = self.dataset[idx.item() % L]
                reward = self.dataset.score_answer(answer=gen_ans, entry=entry)
            rewards.append(reward)
        return jnp.array(rewards, dtype=jnp.float32)


class ReasoningGymTest(BanditTask):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=1000, seed=42, dataset_type="countdown"):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset_size = dataset_size
        self.dataset = reasoning_gym.create_dataset(dataset_type, size=dataset_size, seed=seed)


# ----------------------------
# AIME tasks (Math-Verify)
# ----------------------------
def _build_aime_prompt(example):
    q = example.get("problem")
    prompt = (
        "User: You are a helpful assistant. You first think about the reasoning "
        "process in your mind and then provide the user with the answer. Most answers "
        "are simple numbers, fractions or expressions. "
        f"Question: {q}\nAssistant: <think"
    )
    return {**example, "prompt": prompt}


def _get_aime_answer(example):
    return example.get("answer")


def _score_with_math_verify(pred_text, gold_text):
    if not pred_text:
        return 0.0
    try:
        gold = _parse_gold_cached(gold_text)
        pred = mv_parse(pred_text)   
        return 1.0 if mv_verify(gold, pred) else 0.0
    except Exception:
        return 0.0


class AIMETrain(BanditTask):
    """Train on DeepScaleR preview (AIME-style)."""
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset(
            "agentica-org/DeepScaleR-Preview-Dataset", split="train"
            # "agentica-org/DeepScaleR-Preview-Dataset", split="train"
        ).map(_build_aime_prompt)

    def __len__(self):
        return len(self.dataset)

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.encoding_tokenizer.encode(
                    self.dataset[i.item() % len(self.dataset)]["prompt"]
                ),
                self.max_num_steps,
            )
            for i in indices
        ])

    def get_batch_fitness(self, indices, full_generations):
        np_full = np.array(full_generations)
        L = len(self.dataset)
        golds = [_get_aime_answer(self.dataset[i.item() % L]) for i in indices]
        rewards = []
        for j, seq in enumerate(np_full):
            gen = safe_decode(seq, self.decoding_tokenizer)
            gen = strip_thoughts(gen).strip()
            rewards.append(_score_with_math_verify(gen, golds[j]))
        return jnp.array(rewards, dtype=jnp.float32)


class AIMETrainAIME(BanditTask):
    """Train on DeepScaleR preview (AIME-style)."""
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset(
            "gneubig/aime-1983-2024", split="train"
        ).map(_build_aime_prompt)

    def __len__(self):
        return len(self.dataset)

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.encoding_tokenizer.encode(
                    self.dataset[i.item() % len(self.dataset)]["prompt"]
                ),
                self.max_num_steps,
            )
            for i in indices
        ])

    def get_batch_fitness(self, indices, full_generations):
        np_full = np.array(full_generations)
        L = len(self.dataset)
        golds = [_get_aime_answer(self.dataset[i.item() % L]) for i in indices]
        rewards = []
        for j, seq in enumerate(np_full):
            gen = safe_decode(seq, self.decoding_tokenizer)
            gen = strip_thoughts(gen).strip()
            rewards.append(_score_with_math_verify(gen, golds[j]))
        return jnp.array(rewards, dtype=jnp.float32)


class AIME24Test(AIMETrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset("HuggingFaceH4/aime_2024", split="train").map(_build_aime_prompt)


class AIME25Test(AIMETrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = load_dataset("math-ai/aime25", split="test").map(_build_aime_prompt)


# ----------------------------
# Countdown variants
# ----------------------------
class CountdownNTrainRG(ReasoningGymTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=256, seed=42):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         dataset_size=dataset_size, seed=seed, dataset_type="countdown")
        # Override dataset with constraints:
        self.dataset = reasoning_gym.create_dataset(
            "countdown", size=dataset_size, seed=seed, min_numbers=3, max_numbers=4
        )


class CountdownNValRG(ReasoningGymTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=128, seed=1337):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         dataset_size=dataset_size, seed=seed, dataset_type="countdown")
        self.dataset = reasoning_gym.create_dataset(
            "countdown", size=dataset_size, seed=seed, min_numbers=3, max_numbers=4
        )


_COUNTDOWN_JSON_PATH = Path(__file__).resolve().parents[3] / "data" / "countdown.json"


def _load_countdown_json_dataset(seed: int, dataset_size: int | None = None):
    ds = load_dataset("json", data_files=str(_COUNTDOWN_JSON_PATH), split="train")
    ds = ds.shuffle(seed=seed)
    if dataset_size is not None:
        ds = ds.select(range(min(dataset_size, len(ds))))
    return ds


_COUNTDOWN_CHAT_THINK_CLOSE = r"(?:<\/think>|<\/redacted_thinking>)"


def _countdown_chat_format_reward(response: str) -> float:
    think_regex = rf"<think>.*?{_COUNTDOWN_CHAT_THINK_CLOSE}"
    answer_regex = r"<answer>.*?<\/answer>"
    full_format_regex = rf"^<think>.*?{_COUNTDOWN_CHAT_THINK_CLOSE}\n<answer>.*?<\/answer>$"

    think_match = re.search(think_regex, response, re.DOTALL)
    answer_match = re.search(answer_regex, response, re.DOTALL)
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    if full_format_match:
        return 1.0
    reward = 0.0
    if think_match:
        reward += 0.1
    if answer_match:
        reward += 0.5
    return reward


def _countdown_chat_answer_reward(response: str, numbers, target) -> float:
    answer_regex = r"<answer>(.*?)<\/answer>"
    all_matches = re.findall(answer_regex, response, re.DOTALL)
    if not all_matches:
        return 0.0

    answer_content = all_matches[-1].strip()
    allowed_chars = r"^[0-9+\-*/() ]+$"
    if not answer_content or not re.match(allowed_chars, answer_content):
        return 0.0

    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    try:
        result = eval(answer_content, {"__builtins__": None}, {})
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def _countdown_chat_score(response: str, numbers, target) -> float:
    format_reward = _countdown_chat_format_reward(response)
    answer_reward = _countdown_chat_answer_reward(response, numbers, target)
    return format_reward * 0.1 + answer_reward


def _countdown_chat_score_generation(generation: str, numbers, target) -> float:
    """Score only the model continuation (prompt already ends with <think>)."""
    format_reward = _countdown_chat_format_reward("<think>" + generation)
    answer_reward = _countdown_chat_answer_reward(generation, numbers, target)
    return format_reward * 0.1 + answer_reward


class CountdownChatTrain(BanditTask):
    """Countdown with eggroll-vllm chat/XML prompt format (countdown.json)."""

    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=256, seed=0):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps)
        self.dataset = _load_countdown_json_dataset(seed=seed, dataset_size=dataset_size)
        self.dataset_size = len(self.dataset)

    def __len__(self):
        return self.dataset_size

    def get_input(self, indices):
        return jnp.array([
            get_padded_prompt(
                self.encoding_tokenizer.encode(self.dataset[i.item() % len(self.dataset)]["context"]),
                self.max_num_steps,
            )
            for i in indices
        ])

    def get_batch_fitness(self, indices, full_generations):
        np_full = np.array(full_generations)
        rewards = []
        for idx, tok_seq in zip(indices, np_full):
            example = self.dataset[idx.item() % len(self.dataset)]
            numbers = example["numbers"]
            target = example["target"]
            context = example["context"]
            prompt_len = len(self.encoding_tokenizer.encode(context))
            generation = _extract_countdown_chat_generation(
                context, tok_seq, prompt_len, self.decoding_tokenizer
            )
            rewards.append(_countdown_chat_score_generation(generation, numbers, target))
        return jnp.array(rewards, dtype=jnp.float32)


class CountdownChatVal(CountdownChatTrain):
    def __init__(self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                 dataset_size=256, seed=12345):
        super().__init__(encoding_tokenizer, decoding_tokenizer, max_num_steps,
                         dataset_size=dataset_size, seed=seed)


# ----------------------------
# Dynamic RG class builders
# ----------------------------
def _to_camel(s: str) -> str:
    return "".join(part.capitalize() for part in s.split("_"))


def _make_rg_train_class(dataset_name: str):
    cls_name = f"{_to_camel(dataset_name)}TrainRG"
    return type(
        cls_name,
        (ReasoningGymTrain,),
        {
            "__init__": (lambda self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                               dataset_size=10000, seed=42, _ds=dataset_name:
                         ReasoningGymTrain.__init__(
                             self,
                             encoding_tokenizer,
                             decoding_tokenizer,
                             max_num_steps,
                             dataset_size=dataset_size,
                             seed=seed,
                             dataset_type=_ds,
                         ))
        }
    )


def _make_rg_validation_class(dataset_name: str):
    cls_name = f"{_to_camel(dataset_name)}ValRG"
    return type(
        cls_name,
        (ReasoningGymTrain,),
        {
            "__init__": (lambda self, encoding_tokenizer, decoding_tokenizer, max_num_steps,
                               dataset_size=2000, seed=1337, _ds=dataset_name:
                         ReasoningGymTrain.__init__(
                             self,
                             encoding_tokenizer,
                             decoding_tokenizer,
                             max_num_steps,
                             dataset_size=dataset_size,
                             seed=seed,
                             dataset_type=_ds,
                         ))
        }
    )


# ----------------------------
# Registry
# ----------------------------
all_tasks = {
    "fastzero": FastZero,
    "uniquetok": UniqueTok,
    "reptok": RepTok,
    "digits": Digits,
    "gsm8k": GSM8KTrain,
    "gsm8ksft": GSM8KSFT,
    "countdownn": CountdownNTrainRG,
    "countdown_chat": CountdownChatTrain,
    "aime24": AIMETrainAIME,
    "aime25": AIMETrainAIME,
}

validation_tasks = {
    "fastzero": FastZero,
    "uniquetok": UniqueTok,
    "reptok": RepTok,
    "digits": Digits,
    "gsm8k": GSM8KTest,
    "gsm8ksft": GSM8KTest,
    "countdownn": CountdownNValRG,
    "countdown_chat": CountdownChatVal,
    "aime24": AIME24Test,
    "aime25": AIME25Test,
}

for _name in RG_DATASETS:
    TrainClass = _make_rg_train_class(_name)
    ValClass = _make_rg_validation_class(_name)
    all_tasks[_name] = TrainClass
    validation_tasks[_name] = ValClass