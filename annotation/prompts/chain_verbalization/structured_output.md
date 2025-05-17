You will receive a JSON object containing a sentence that expands a causal event chain from a news article. Your task is to ensure this JSON is properly formatted according to the required schema.

## Input
A JSON object containing a sentence field, possibly with formatting inconsistencies.

## Output Format
Reformat the input into this exact JSON structure:

```json
{
  "sentence": "Copy the sentence exactly as provided in the input - do not modify"
}
```

## Important Rules
1. COPY THE SENTENCE VERBATIM - do not modify, summarize, or rephrase it
2. Ensure valid JSON syntax with proper quotes, commas, and brackets
3. Remove any nested quotes or escape characters that might have been introduced

Your only task is to parse and reformat the JSON without altering the sentence in any way.