"""Biasing list 的產生與讀取。

論文 3.1：每句的 biasing list = 句中所有 rare words（正樣本 E+）
          + 從 rare word pool 隨機採樣、不在句中的 rare words（負樣本 E-）。
論文 2.1：COALA 額外放一個特殊 token <unbiased> 作為 e_0。
          若句中無任何 rare word，e_0 就是唯一正樣本；否則 e_0 是負樣本。

注意兩個詞表的用途不同（見 is21_deep_bias/README.md）：
  common_words_5k.txt  → 判定哪些字是 rare word（取補集）
  all_rare_words.txt   → 採樣干擾詞 E- 的詞池
"""

import collections
import json
import random
from pathlib import Path

from rare_words import (
    RARE_WORDS_FILE,
    load_common_words,
    load_rare_words,
    find_rare_words,
)

UNBIASED = "<unbiased>"

OFFICIAL_REF_DIR = Path("/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref")


def load_rare_pool(path=RARE_WORDS_FILE):
    """回傳排序好的 rare word list（採樣需要有序容器才能重現）。"""
    return sorted(load_rare_words(path))


def make_biasing_list(positives, rare_pool, n_distractors, rng):
    """回傳 biasing list = 正樣本 + n_distractors 個干擾詞，字母排序。

    不含 <unbiased>。長度 = len(positives) + n_distractors。
    """
    pos = sorted(set(positives))
    pos_set = set(pos)

    # 多抽 len(pos) 個，濾掉不小心抽到的正樣本後仍保證足夠
    sampled = rng.sample(rare_pool, n_distractors + len(pos))
    negatives = [w for w in sampled if w not in pos_set][:n_distractors]

    return sorted(pos + negatives)


def make_stage1_list(positives, rare_pool, rng, max_pos=5, n_total=10):
    """Stage 1 prompt：最多取 max_pos 個正樣本，補干擾詞到 n_total。

    論文 3.4：'prompts are constructed by sampling up to 5 positive
    entities within a 10-entity set' —— 刻意不給全部正樣本，
    避免模型過度依賴 biasing 資訊。
    """
    pos = sorted(set(positives))
    if len(pos) > max_pos:
        pos = sorted(rng.sample(pos, max_pos))
    return make_biasing_list(pos, rare_pool, n_total - len(pos), rng)


def add_unbiased(biasing_list):
    """把 e_0 = <unbiased> 放在清單最前面，回傳完整的 E。"""
    return [UNBIASED] + list(biasing_list)


def split_pos_neg(full_list, positives):
    """依論文 2.1 切出 E+ / E- 的索引。

    - 句中有 rare word → E+ = 那些 rare word，<unbiased> 屬於 E-
    - 句中沒有 rare word → E+ = {<unbiased>}，其餘全是 E-
    """
    pos_set = set(positives)
    if pos_set:
        pos_idx = [i for i, e in enumerate(full_list) if e in pos_set]
    else:
        pos_idx = [full_list.index(UNBIASED)]
    pos_idx_set = set(pos_idx)
    neg_idx = [i for i in range(len(full_list)) if i not in pos_idx_set]
    return pos_idx, neg_idx


def load_official_ref(split, n):
    """讀官方 ref/{split}.biasing_{n}.tsv。

    回傳 {utt_id: {"text":..., "positives":[...], "biasing_list":[...]}}
    """
    path = OFFICIAL_REF_DIR / f"{split}.biasing_{n}.tsv"
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            utt_id, text, pos_json, list_json = line.split("\t")
            out[utt_id] = {
                "text": text,
                "positives": json.loads(pos_json),
                "biasing_list": json.loads(list_json),
            }
    return out


def main():
    rare_pool = load_rare_pool()
    common_set = load_common_words()
    print(f"rare pool (採樣 E- 用): {len(rare_pool)} 個詞")
    print(f"common set (判定 E+ 用): {len(common_set)} 個詞")

    # --- 1. 產生器示範 + 可重現性 ---
    text = ("the air and the earth are curiously mated and intermingled "
            "as if the one were the breath of the other")
    positives = find_rare_words(text, common_set)
    print(f"\n正樣本 E+ = {positives}")

    lst_a = make_biasing_list(positives, rare_pool, 500, random.Random(42))
    lst_b = make_biasing_list(positives, rare_pool, 500, random.Random(42))
    lst_c = make_biasing_list(positives, rare_pool, 500, random.Random(43))

    print(f"500 干擾詞 + {len(positives)} 正樣本 = 長度 {len(lst_a)}")
    print(f"正樣本都在清單裡      : {set(positives) <= set(lst_a)}")
    print(f"seed 42 兩次是否相同  : {lst_a == lst_b}")
    print(f"seed 42 vs 43 是否相同: {lst_a == lst_c}")
    print(f"前 8 個: {lst_a[:8]}")

    full = add_unbiased(lst_a)
    pos_idx, neg_idx = split_pos_neg(full, positives)
    print(f"加 <unbiased> 後長度  : {len(full)}  (e_0 = {full[0]})")
    print(f"|E+| = {len(pos_idx)}, |E-| = {len(neg_idx)}")

    # 沒有 rare word 的句子
    empty_text = ("when i was a young man i thought paul was making "
                  "too much of his call")
    empty_pos = find_rare_words(empty_text, common_set)
    full2 = add_unbiased(
        make_biasing_list(empty_pos, rare_pool, 500, random.Random(42))
    )
    p2, n2 = split_pos_neg(full2, empty_pos)
    print(f"\n無 rare word 的句子: E+ = {[full2[i] for i in p2]}, "
          f"|E-| = {len(n2)}")

    # --- 2. Stage 1 prompt 示範 ---
    many = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    s1 = make_stage1_list(many, rare_pool, random.Random(0))
    print(f"\nStage 1 清單（7 個正樣本，最多取 5）: 長度 {len(s1)}, "
          f"實際含正樣本 {len(set(s1) & set(many))} 個")

    # --- 3. 用官方檔驗證 ---
    ref = load_official_ref("test-clean", 500)
    n_superset_ok = n_match_ok = 0
    len_dist = collections.Counter()   # 清單長度 - 正樣本數 的分布
    mismatches = []

    for utt_id, r in ref.items():
        n_pos = len(r["positives"])
        n_all = len(r["biasing_list"])
        len_dist[n_all - n_pos] += 1

        if set(r["positives"]) <= set(r["biasing_list"]):
            n_superset_ok += 1

        mine = find_rare_words(r["text"], common_set)
        if set(mine) == set(r["positives"]):
            n_match_ok += 1
        elif len(mismatches) < 5:
            mismatches.append((utt_id, r["positives"], mine))

    total = len(ref)
    print(f"\n官方 test-clean.biasing_500.tsv 共 {total} 列")
    print(f"  E+ 是 E 的子集合          : {n_superset_ok}/{total}")
    print(f"  我的 find_rare_words 一致 : {n_match_ok}/{total}")
    print(f"  清單長度 - 正樣本數 的分布 (只印前 8 種):")
    for k, v in sorted(len_dist.items())[:8]:
        print(f"      {k:>5} -> {v} 列")
    for utt_id, official, mine in mismatches:
        print(f"    MISMATCH {utt_id}")
        print(f"      官方: {official}")
        print(f"      我的: {mine}")


if __name__ == "__main__":
    main()