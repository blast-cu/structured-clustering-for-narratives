You will receive a JSON object containing character analysis and stance assessment from a news article. Your task is to reformat it into the required schema without changing any content.

## Input
A JSON object with character analysis and stance information, possibly with formatting inconsistencies.

## Output Format
Reformat the json in the input into the exact following JSON structure:

```json
{
  "characters": [
    {
      "entity": "[Exact entity mentioned in the input json]",
      "character_group": "[Character group mentioned in the input json]",
      "role": "[Hero/Victim/Threat/Neutral as mentioned in the input json]"
    }
  ],
  "stance": "[Pro/Anti/Neutral as mentioned in the input json]"
}
```

## Important Rules
1. Strictly follow the input json and only fix schema and formatting issues.
2. There can be multiple characters in the input json. Keep the same number of characters as the input
3. Do not add any characters or stance that are not in the input json.
4. Copy each character exactly once
5. Copy values exactly as written in the input
6. Remove any extra fields
7. Output only valid JSON, nothing else

Your only task is to parse and reformat the JSON without changing the content.