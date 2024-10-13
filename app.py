from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text, max_length=130):
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Take input from the user
input_text = input("Enter the text to be summarized: ")
summarized_text = summarize_text(input_text)

print("\nSummarized Text:\n", summarized_text)
