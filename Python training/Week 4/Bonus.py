import tiktoken

sentence = "Yoho Technologies in Chennai helps learners build LLM applications in 2026!"

enc_new = tiktoken.get_encoding("o200k_base")
enc_old = tiktoken.get_encoding("cl100k_base")

for name, enc in [("o200k_base", enc_new), ("cl100k_base", enc_old)]:
    ids = enc.encode(sentence)
    pieces = [enc.decode([token_id]) for token_id in ids]
    print(f"=== {name} ({len(ids)} tokens) ===")
    print("IDs:   ", ids)
    print("Pieces:", pieces)
    print()