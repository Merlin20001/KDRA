from kdra.core.ontology.normalizer import VectorOntologyNormalizer

norm = VectorOntologyNormalizer(model_name='BAAI/bge-small-en-v1.5')

def test_dedupe(name1, name2):
    id1 = norm.normalize_and_dedupe("Method", name1)
    id2 = norm.normalize_and_dedupe("Method", name2)
    print(f"'{name1}' -> {id1}")
    print(f"'{name2}' -> {id2}")
    if id1 == id2:
        print("✅ Matched!\n")
    else:
        print("❌ Did not match\n")

print("Testing Exact & Fuzzy Match...")
test_dedupe("Large Language Model", "Large Language Models")
test_dedupe("Transformer", "Transformers")
test_dedupe("GPT-4", "GPT4")

print("Testing Vector Cross-Match (Semantics)...")
test_dedupe("Neural Network Architecture", "Artificial Neural Net")

