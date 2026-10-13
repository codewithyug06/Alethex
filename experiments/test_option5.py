import alethex

print("Testing Option 5: Distributable Python Library Interface")
chunks = [
    {"id": "1", "timestamp": "2024-01-01T00:00:00", "text": "Alice works at Meta in London."},
    {"id": "2", "timestamp": "2024-06-01T00:00:00", "text": "Alice relocated to SF and joined Anthropic."}
]

filtered = alethex.filter_context(chunks)
print("\n--- RAG Filtering Results ---")
for item in filtered:
    print(f"[{item['status'].upper()}] Source {item['source_id']}: {item['text']} (is_valid={item['is_valid']})")

print("\nOption 5 library test passed successfully!")
