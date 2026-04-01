You will receive a JSON object containing a causal relation prediction and the reasoning behind the prediction. Your 
task is to ensure this JSON is properly formatted according to the required schema.

## Input
A JSON object containing a reason field and a relation field, possibly with formatting inconsistencies.

## Output Format
Reformat the input into this exact JSON structure:

```json
{
"reason": "your reasoning for the answer",
"relation": "causal/none"
}
```

## Important Rules
1. COPY THE reason and relation VERBATIM - do not modify, summarize, or rephrase it
2. Ensure valid JSON syntax with proper quotes, commas, and brackets
3. Remove any nested quotes or escape characters that might have been introduced

Your only task is to parse and reformat the JSON without changing the content.