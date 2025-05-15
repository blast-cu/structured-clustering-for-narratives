You will receive a JSON object containing character analysis and stance assessment from a news article. Your task is to reformat it into the required schema without changing any content.

## Input
A JSON object with character analysis and stance information, possibly with formatting inconsistencies.

## Output Format
Reformat the input into this exact JSON structure:

```json
{
  "characters": [
    {
      "character_group": "[Character group from predefined list]",
      "specific_entity": "[Exact entity mentioned]",
      "role": "[Hero/Victim/Threat/Neutral]",
      "justification": "[Copy the justification exactly as provided in the input - do not modify]"
    }
  ],
  "stance": "[Pro/Anti/Neutral]",
  "stance_justification": "[Copy the stance justification exactly as provided - do not modify]"
}
```

## Important Rules
1. COPY ALL JUSTIFICATIONS VERBATIM - do not modify, summarize, or rephrase them
2. Use only the exact character groups mentioned in the original analysis
3. Use only "Hero", "Victim", "Threat", or "Neutral" for roles and copy as mentioned in the original analysis
4. Use only "Pro", "Anti", or "Neutral" for stance and copy as mentioned in the original analysis
7. Ensure valid JSON with proper quotes, commas, and brackets

Your only task is to parse and reformat the JSON without changing the content or reasoning.