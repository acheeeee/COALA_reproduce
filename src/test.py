from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
inst = "Please determine if the text appears in the input speech. Text:"

for e in ["quilter", " quilter"]:
    ids = tok(e, add_special_tokens=False)["input_ids"]
    print(repr(e), ids, tok.convert_ids_to_tokens(ids))

whole = tok(inst + " quilter", add_special_tokens=False)["input_ids"]
print(tok.convert_ids_to_tokens(whole)[-4:])

