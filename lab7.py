from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

# ---------- Code Generation ----------

generation_prompt = """
Write a Python function to calculate
the factorial of a number.

Python code:
"""

output = generator(
    generation_prompt,
    max_length=100,
    num_return_sequences=1,
    do_sample=False
)

print("Generated Code:")
print(output[0]["generated_text"])


# ---------- Code Debugging ----------

debug_prompt = """
Find and correct the error in the following Python code:

def factorial(n):
    if n == 0:
        return 0
    return n * factorial(n - 1)

Explain the correction and provide the fixed code.
"""

debug_output = generator(
    debug_prompt,
    max_length=150,
    num_return_sequences=1,
    do_sample=False
)

print("\nDebugging Output:")
print(debug_output[0]["generated_text"])