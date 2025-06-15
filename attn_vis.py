from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import matplotlib.pyplot as plt

model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set tokenizer source and target language
tokenizer.src_lang = "en_XX"
src_text = "The agreement on the European Economic Area was signed in August 1992."
encoded = tokenizer(src_text, return_tensors="pt")

# Set forced language token for target (French)
forced_bos_token_id = tokenizer.lang_code_to_id["fr_XX"]

# Generate with attention
with torch.no_grad():
    outputs = model.generate(
        **encoded,
        forced_bos_token_id=forced_bos_token_id,
        output_attentions=True,
        return_dict_in_generate=True
    )

# Get attention and tokens
cross_attn = outputs.cross_attentions[-1][0]  # (num_heads, tgt_len, src_len)
attn_matrix = cross_attn.mean(dim=0).squeeze().detach().numpy()
# normalize
attn_matrix = (attn_matrix - attn_matrix.min()) / (attn_matrix.max() - attn_matrix.min())
src_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
tgt_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])

# Plot and save
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(attn_matrix, cmap="gist_yarg")
ax.set_xticks(range(len(tgt_tokens)))
ax.set_xticklabels(tgt_tokens, rotation=90)
ax.set_yticks(range(len(src_tokens)))
ax.set_yticklabels(src_tokens)

plt.title("mBART Cross-Attention Heatmap (EN → FR)", pad=20)
plt.tight_layout()
plt.savefig("mbart_en_fr_attention3.png")
print("Saved to mbart_en_fr_attention.png")
