You will be given a block of text that is the output of a large language model. The first section of the text is the 
chain of thought of the model enclosed within <think> and </think> tags. The second section is the final answer of 
the model, which is a JSON object. Your task is to extract structured information from the JSON object and convert 
it into the following JSON format:

```json
{
  "characters": [
    {
      "character_group": "[Identified character group from predefined list]",
      "specific_entity": "[Exact entity mentioned in the event chain]",
      "role": "[Hero/Victim/Threat/Neutral]",
      "justification": "[Brief explanation with textual evidence]"
    }
  ],
  "stance": "[Pro/Anti/Neutral]",
  "stance_justification": "[Brief explanation with textual evidence while focusing mainly on the event chain]"
}