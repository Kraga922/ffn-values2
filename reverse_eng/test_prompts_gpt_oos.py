import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 🔧 Choose your model type here
model_type = 'llama'

# 🔍 Model mapping — you can easily switch here
model_map = {
    'llama': "openai/gpt-oss-20b",
    # 'llama-8b': "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # 'llama-70b': "meta-llama/Meta-Llama-3-70B-Instruct",
    # 'llama-70b-4bit': "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    # 'gemma': "google/gemma-2-2b",
}

# 📌 Set model ID
model_id = model_map.get(model_type)

if not model_id:
    raise ValueError(f"Unknown model_type '{model_type}'. Check the model_map dictionary.")

# 🧠 Load tokenizer and model
print(f"\n🔄 Loading model: {model_id}...\n")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16  # You can change to float16 if needed
)

# 🪄 Create text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 💬 Interactive prompt loop
print("\n✅ Model is ready! Type your prompt below.")
print("Type 'exit' or press Ctrl+C to quit.\n")

try:
    while True:
        prompt = input("📝 Prompt:\n> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("👋 Exiting.")
            break

        output = generator(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        print("\n🤖 Response:\n" + output[0]["generated_text"] + "\n" + "-"*50)

except KeyboardInterrupt:
    print("\n👋 Interrupted. Exiting.")
