"""重新產生「物件 總覽」與「函式 總覽」（從各筆記的 frontmatter 與第一行說明抽取）。

用法（在 repo 根目錄）：python3 obsidian/.tools/build_indexes.py
新增檔案時，把檔名加進下面的 FILE_ORDER。
"""
import pathlib
import re

VAULT = pathlib.Path(__file__).resolve().parent.parent

FILE_ORDER = [
    ("資料層", ["parse_trans", "rare_words", "build_manifest", "figure1", "biasing", "dataset"]),
    ("模型層", ["modules", "prompt", "lora_setup", "model", "projector", "losses"]),
    ("訓練與評估", ["overfit_test", "train_stage1", "evaluate"]),
    ("工具", ["check_models", "check_shape", "dump_tb"]),
]
NO_FUNCTION_FILES = ["test"]

OBJ_CATS = [
    ("常數", "寫死在程式裡、不會變的值（模型名稱、路徑、超參數、prompt 文字）"),
    ("張量", "`torch.Tensor`（以及波形 `numpy.ndarray`）：模型裡真正流動的數字"),
    ("資料結構", "Python 的 `str` / `list` / `dict` / `set`：清單、樣本、batch、設定"),
    ("模型與工具", "有行為的物件：模型零件、tokenizer、optimizer、DataLoader 等"),
    ("資料檔", "磁碟上的檔案：輸入資料、詞表、輸出結果"),
]

# 沒有「**功能**：」那一行的短方法，手動給說明
OVERRIDE = {
    "AudioAdapter.__init__()": "建 `Linear(1280·stack, 576)`",
    "CTCHead.__init__()": "建 `Linear(576, 29)`",
    "CTCHead.forward()": "`(B, L, 576)` → [[ctc_logits]] `(B, L, 29)`",
    "DiscriminativeProjector.__init__()": "建 `Linear(576, D′) → ReLU → Linear(D′, 1)`",
}
KIND_ORDER = {"類別": 0, "方法": 1, "函式": 2, "內部函式": 3, "main": 4}


def front(t):
    m = re.match(r"---\n(.*?)\n---\n", t, re.S)
    d = {}
    if m:
        for line in m.group(1).split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def first_desc(t):
    m = re.search(r"\*\*(?:存什麼|是什麼|功能)\*\*：(.+)", t)
    return m.group(1).strip() if m else ""


def cell(s):
    return s.replace("|", "\\|")


def build_objects():
    out = ["---", "tags: [COALA, 物件, 總覽]", "---", "", "# 物件 總覽", "",
           "← [[COALA 索引]] · [[函式 總覽]]", "",
           "**第一層：原子物件。** 每篇記一個有名字的東西「存什麼、長什麼樣、什麼型別、從哪來到哪去」。", "",
           "## 收錄規則", "",
           "- **有自己的筆記**：常數、跨函式流動的 tensor / 資料結構（函式的輸入輸出、類別屬性）、模型零件、資料檔",
           "- **只列在函式筆記的「內部物件」表**：只活在一個函式裡的暫存變數（迴圈變數、計數器）",
           "- **同名不同義**用括號區分：例如 [[h (adapter 輸出)]] vs [[h (projector 輸入)]]、三種 `out`",
           "- 程式裡的其他變數名寫在筆記的 `aliases`（Obsidian 搜尋得到）",
           "- shape 以目前預設 **stack=2** 為準（音訊位置數 750）；stack=4 時是 375", ""]
    total = 0
    for cat, desc in OBJ_CATS:
        files = sorted((VAULT / "物件" / cat).glob("*.md"), key=lambda p: p.stem.lower())
        total += len(files)
        out += [f"## {cat}（{len(files)}）", "", desc, ""]
        if cat == "張量":
            out += ["| 物件 | shape | dtype | 存什麼 |", "|---|---|---|---|"]
        elif cat == "資料檔":
            out += ["| 物件 | 位置 | 存什麼 |", "|---|---|---|"]
        else:
            out += ["| 物件 | 型別 | 存什麼 |", "|---|---|---|"]
        for p in files:
            t = p.read_text(encoding="utf-8")
            f = front(t)
            d = cell(first_desc(t))
            if cat == "張量":
                out.append(f"| [[{p.stem}]] | `{cell(f.get('shape', ''))}` | {cell(f.get('dtype', ''))} | {d} |")
            elif cat == "資料檔":
                out.append(f"| [[{p.stem}]] | `{cell(f.get('位置', ''))}` | {d} |")
            else:
                out.append(f"| [[{p.stem}]] | `{cell(f.get('型別', ''))}` | {d} |")
        out.append("")
    (VAULT / "物件" / "物件 總覽.md").write_text("\n".join(out), encoding="utf-8")
    return total


def build_functions():
    out = ["---", "tags: [COALA, 函式, 總覽]", "---", "", "# 函式 總覽", "",
           "← [[COALA 索引]] · [[物件 總覽]]", "",
           "**第二層：函式與類別。** 每篇記「功能、輸入、輸出、內部用到哪些物件、呼叫誰、被誰呼叫」，並附原始碼。", "",
           "## 命名規則", "",
           "| 種類 | 筆記名 | 例 |", "|---|---|---|",
           "| 函式 | `名稱()` | [[ctc_compress()]] |",
           "| 類別 | `類別名` | [[COALAModel]] |",
           "| 方法 | `類別名.方法()` | [[COALAModel.forward_stage1()]] |",
           "| 多個檔案同名（`main`、`parse_args`） | `檔名.名稱()` | [[model.main()]] |", "",
           "> 原始碼是寫筆記當下的快照。程式改了之後，以 `src/` 為準。", ""]
    total = 0
    for layer, names in FILE_ORDER:
        out += [f"## {layer}", ""]
        for name in names:
            files = list((VAULT / "函式" / name).glob("*.md"))
            files.sort(key=lambda p: (KIND_ORDER.get(front(p.read_text(encoding="utf-8")).get("種類", ""), 9), p.stem))
            total += len(files)
            out += [f"### [[{name}.py]]（{len(files)}）", "", "| 名稱 | 種類 | 功能 |", "|---|---|---|"]
            for p in files:
                t = p.read_text(encoding="utf-8")
                k = front(t).get("種類", "")
                d = OVERRIDE.get(p.stem) or cell(first_desc(t))
                out.append(f"| [[{p.stem}]] | {k} | {d} |")
            out.append("")
    for name in NO_FUNCTION_FILES:
        out += [f"### [[{name}.py]]（0）", "", "沒有函式（頂層實驗腳本）。", ""]
    (VAULT / "函式" / "函式 總覽.md").write_text("\n".join(out), encoding="utf-8")
    return total


if __name__ == "__main__":
    print(f"物件 {build_objects()} 篇、函式 {build_functions()} 篇")
