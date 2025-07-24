You will receive a JSON object containing narrative cluster analysis results. Your task is to ensure this JSON is properly formatted according to the required schema.

## Input
A JSON object containing framing analysis results, possibly with formatting inconsistencies.

## Output Format
Reformat the input into this exact JSON structure:

```json
{
  "framing_pattern_detected": true/false,
  "theme": "[Exact theme mentioned in the input json]",
  "framing_elements": {
    "issue": "[Exact string mentioned in the inpuit json, null if missing]",
    "evaluation": "[Exact string mentioned in the inpuit json, null if missing]",
    "resolution": "[Exact string mentioned in the inpuit json, null if missing]"
  }
}

Important Rules

1. COPY ALL CONTENT VERBATIM - do not modify, summarize, or rephrase any sentences or substrings
2. Ensure valid JSON syntax with proper quotes, commas, and brackets
3. Convert boolean values to proper JSON format (true/false, not "true"/"false")
4. Ensure framing_elements values are either properly quoted strings or null (not "null")
5. Remove any nested quotes or escape characters that might have been introduced
6. If any required fields are missing, set them to appropriate default values:
  - framing_pattern_detected: false
  - theme: "[original sentence if present]"
  - framing_elements: {"issue": null, "evaluation": null, "resolution": null}

Your only task is to parse and reformat the JSON structure without altering any sentence content.