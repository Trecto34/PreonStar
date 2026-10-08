"""Write the golden files for POST /v1/systemone: tests/ekbasis_eval/golden/*.json and golden/labels.json.

For each request (21 cases, see CASES) it records:
  - request: the JSON body as the client sends it;
  - mode: "single" (one question), "independent" (several questions, read_once false: one prompt each) or
    "read_once" (several questions, one token sequence, letter_adapter.py _read_once_ids layout);
  - per prompt: user_json (decision_core.render_user, or the follow-up JSON), prompt_text (the full chat text the server
    must build), token_ids (tokenizers, add_special_tokens=False, as the reference does), label_ids (the one-token id of
    each option label, in option order), answer_position (index of the last prompt token, where the logits are read);
  - read_once: the concatenated text, its token ids, and the answer positions in that sequence.

Two independent checks are run while writing (the script aborts on a mismatch):
  1. the literal ChatML wrapper in common.py equals the HF chat_template.jinja render (enable_thinking=False, rendered with
     jinja2 under transformers' environment settings) for every prompt and read-once suffix;
  2. the read-once token ids equal the concatenation of the separately tokenised pieces (recorded as
     "pieces_equal_whole_text" for the concatenated text, which is informational).

Requires: tokenizers (already installed system-wide, used read-only) and jinja2. Downloads tokenizer.json (20 MB) and
chat_template.jinja from caiovicentino1/Ekbasis-27B into the cache, not into the repository.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import decision_core as dc  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

GOLDEN = os.path.join(HERE, "golden")
X = "⁣"  # placeholder used to split the template, as letter_adapter.py does

STATE_GIT = ("Current branch: main\nBranches (last commit message): main (init), dev (wip)\n"
             "git status: nothing to commit, working tree clean\nStash entries: 0")
STATE_LIST = "Files in the working directory: " + ", ".join(f"file{i:02d}.txt (v{i % 5})" for i in range(26))


def noul(text, criteria=None, qtype="noul"):
    q = {"type": qtype, "instructions": text}
    if criteria is not None:
        q["criteria"] = criteria
    return q


GIT_Q = "Do these commands permanently lose any uncommitted work?"
# (name, description, [(question name, question)], state, read_once)
CASES = [
    ("01_noul_basic", "noul, no criteria: shown as true/false, options yes/no",
     [("lost", noul(GIT_Q))], STATE_GIT, False),
    ("02_noul_criteria", "noul with criteria.true and criteria.false",
     [("gone", noul("Is the file removed?", {"true": "the file is gone", "false": "the file still exists"}))],
     STATE_GIT, False),
    ("03_noul_null_description", "noul criteria.true null: the option shows its own name, yes",
     [("q", noul("Is it done?", {"true": None, "false": "not done"}))], STATE_GIT, False),
    ("04_boolean_type", "boolean is handled like noul",
     [("green", noul("Is the build green?", {"true": "all tests pass"}, qtype="boolean"))], STATE_GIT, False),
    ("05_choice_3_dict_order", "choice, 3 options, dict order kept (not sorted)",
     [("branch", {"type": "choice", "instructions": "Which branch should the commit go to?",
                  "criteria": {"main": "production", "dev": "integration", "feature/x": "personal work"}})],
     STATE_GIT, False),
    ("06_choice_26_last_Z", "choice with exactly 26 options: last label is Z",
     [("pick", {"type": "choice", "instructions": "Pick the file to delete.",
                "criteria": {f"file{i:02d}.txt": f"candidate {i}" for i in range(26)}})], STATE_LIST, False),
    ("07_choice_27_first_AA", "27 options: the 27th label is AA",
     [("pick", {"type": "choice", "instructions": "Pick the file to delete.",
                "criteria": {f"file{i:02d}.txt": f"candidate {i}" for i in range(27)}})], STATE_LIST, False),
    ("08_choice_160_max", "160 options (the one-pass maximum)",
     [("pick", {"type": "choice", "instructions": "Pick the file to delete.",
                "criteria": {f"file{i:03d}.txt": f"candidate {i}" for i in range(160)}})], STATE_LIST, False),
    ("09_score_dict_order", "score, dict criteria: keys in JSON order, not sorted",
     [("risk", {"type": "score", "instructions": "How risky is it?",
                "criteria": {"5": "very high", "1": "very low", "3": "medium"}})], STATE_GIT, False),
    ("10_score_list", "score, list criteria: labels are the indices 0, 1, 2",
     [("risk", {"type": "score", "instructions": "How many files are at risk?",
                "criteria": ["none", "some", "many"]})], STATE_GIT, False),
    ("11_score_null_description", "score dict with a null description: shows the key",
     [("risk", {"type": "score", "instructions": "Rate it.", "criteria": {"0": None, "1": "one"}})],
     STATE_GIT, False),
    ("12_unicode_state", "non-ASCII state and instructions are emitted raw",
     [("q", noul("Está ok? Résumé: sim.", {"true": "São Paulo, 日本語 🙂"}))],
     {"city": "São Paulo", "note": "日本語", "emoji": "🙂", "units": "µm ±5°"}, False),
    ("13_nested_state", "nested dict, list, bool, null and floats in the state",
     [("q", noul("Is the stock below the reorder point?"))],
     {"sku": {"id": 4411, "tags": ["a", "b"], "ok": True, "note": None},
      "levels": [{"w": 1.5, "n": 0}, {"w": 1e-07, "n": -2}], "ratio": 0.333333}, False),
    ("14_list_state", "a JSON list as the state (not a string)",
     [("q", noul("Is any item missing?"))], [1, "two", {"three": 3}, None, True], False),
    ("15_control_chars", "control characters, quotes and backslashes in a string state",
     [("q", noul("Is the escaping right?"))], 'line1\nline2\t"q" \\ \u0001 \u007f end\r', False),
    ("16_u2028_and_emoji", "U+2028 and emoji in criteria descriptions",
     [("pick", {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "x y", "b": "🙂 ok"}})],
     STATE_GIT, False),
    ("17_read_once_3q", "read_once with noul, choice and score (3 questions)",
     [("lost", noul(GIT_Q)),
      ("branch", {"type": "choice", "instructions": "Which branch?", "criteria": {"main": "prod", "dev": "int"}}),
      ("risk", {"type": "score", "instructions": "How risky?", "criteria": ["low", "high"]})],
     STATE_GIT, True),
    ("18_read_once_1q_ignored", "read_once true with one question: flag ignored, single prompt",
     [("lost", noul(GIT_Q))], STATE_GIT, True),
    ("19_independent_2q", "two questions, read_once false: one full prompt each",
     [("lost", noul(GIT_Q)),
      ("pick", {"type": "choice", "instructions": "Pick one.", "criteria": {"x": "first", "y": "second"}})],
     STATE_GIT, False),
    ("20_read_once_2q_40opts", "read_once, second question with 40 options (AA onward)",
     [("lost", noul(GIT_Q)),
      ("pick", {"type": "choice", "instructions": "Pick the file.",
                "criteria": {f"f{i}": f"option {i}" for i in range(40)}})],
     STATE_LIST, True),
    ("21_missing_instructions", "question without instructions: empty criterion text",
     [("q", {"type": "noul"})], STATE_GIT, False),
]


def template_env(path):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)  # transformers' settings

    def raise_exception(message):
        raise ValueError(message)
    env.globals["raise_exception"] = raise_exception
    with open(path, encoding="utf-8") as f:
        return env.from_string(f.read())


def render(tpl, messages, gen):
    return tpl.render(messages=messages, add_generation_prompt=gen, enable_thinking=False, tools=None)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    tok_path = c.model_file("tokenizer.json")
    tpl_path = c.model_file("chat_template.jinja")
    tpl = template_env(tpl_path)
    tok = Tokenizer.from_file(tok_path)

    def enc(text):
        return tok.encode(text, add_special_tokens=False).ids

    # Label table: every label must be exactly one token, bare (no leading space, no BOS).
    label_ids = {}
    for lab in dc.LABELS:
        ids = enc(lab)
        if len(ids) != 1:
            raise SystemExit(f"label {lab!r} is {len(ids)} tokens; decision_core would be wrong")
        label_ids[lab] = ids[0]
    if len(label_ids) != 588 or "BQ" in label_ids:
        raise SystemExit("label table differs from the 588-label reference")
    os.makedirs(GOLDEN, exist_ok=True)
    provenance = {
        "model_repo": c.MODEL_REPO, "decision_core_sha256": sha256_file(os.path.join(HERE, "decision_core.py")),
        "chat_template_sha256": sha256_file(tpl_path), "tokenizer_json_sha256": sha256_file(tok_path),
        "tokenizers_version": __import__("tokenizers").__version__,
    }
    with open(os.path.join(GOLDEN, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"provenance": provenance, "count": len(label_ids),
                   "checked": {k: label_ids[k] for k in ("A", "Z", "AA", "ZZ")},
                   "labels": [{"label": k, "token_id": v} for k, v in label_ids.items()]},
                  f, ensure_ascii=False, indent=1)
        f.write("\n")

    for name, desc, qs, state, read_once in CASES:
        questions = [(qn, q) for qn, q in qs]
        body = c.build_request(state, questions, read_once)
        qlist = [q for _, q in questions]
        if len(qlist) == 1:
            mode = "single"
        else:
            mode = "read_once" if read_once else "independent"
        user_messages = lambda q: [{"role": "system", "content": dc.SYSTEM},  # noqa: E731
                                   {"role": "user", "content": c.user_text(state, q)}]
        steps = []
        if mode in ("single", "independent"):
            texts = [c.chat_prompt(state, q) for q in qlist]
            for (qn, q), text in zip(questions, texts):
                if render(tpl, user_messages(q), True) != text:
                    raise SystemExit(f"{name}: literal ChatML differs from chat_template.jinja")
                ids = enc(text)
                steps.append({"question": qn, "user_json": c.user_text(state, q), "prompt_text": text,
                              "token_ids": ids, "label_ids": [label_ids[dc.LABELS[i]] for i in range(len(dc.options_of(q)))],
                              "answer_position": len(ids) - 1})
            entry = {"name": name, "description": desc, "mode": mode, "request": body, "steps": steps}
        else:
            pieces = c.readonce_pieces(state, qlist)
            # The same suffix built from the template, as letter_adapter._read_once_ids does.
            u = render(tpl, [{"role": "user", "content": X}], True)
            user_open, gen = u.split(X)
            a = render(tpl, [{"role": "user", "content": X}, {"role": "assistant", "content": X}], False)
            end_assist = a.split(X)[-1]
            tpl_pieces = [render(tpl, user_messages(qlist[0]), True)] + [
                "?" + end_assist + user_open + dc.render_followup(q, dc.options_of(q)) + gen for q in qlist[1:]]
            if tpl_pieces != pieces:
                raise SystemExit(f"{name}: read-once pieces differ from the template layout")
            ids, pos, total = [], [], 0
            per = []
            for p in pieces:
                pi = enc(p)
                per.append(pi)
                ids += pi
                total += len(pi)
                pos.append(total - 1)
            whole = enc("".join(pieces))
            for i, ((qn, q), p) in enumerate(zip(questions, pieces)):
                steps.append({"question": qn, "user_json": (c.user_text(state, q) if i == 0 else dc.render_followup(q, dc.options_of(q))),
                              "label_ids": [label_ids[dc.LABELS[i]] for i in range(len(dc.options_of(q)))], "answer_position": pos[i]})
            entry = {"name": name, "description": desc, "mode": mode, "request": body, "steps": steps,
                     "read_once_prompt_text": "".join(pieces), "read_once_token_ids": ids,
                     "read_once_pieces": pieces, "pieces_equal_whole_text": whole == ids}
        with open(os.path.join(GOLDEN, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"{name}: mode={mode} questions={len(questions)} "
              f"tokens={sum(len(s.get('token_ids', [])) for s in steps) or len(entry.get('read_once_token_ids', []))}")
    print(f"wrote {len(CASES)} cases and labels.json to {GOLDEN}")


if __name__ == "__main__":
    main()
