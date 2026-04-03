## Task Overview
You will generate a concise, plausible sentence that expands on a causal event chain formatted as verb-object pairs from a Reddit post, incorporating relevant entities and reflecting the causal relationship accurately.

## Understanding Event Chains
- Events correspond to what we perceive around us and are denoted as a (VERB, OBJECT) pair
- The OBJECT is the direct object of the VERB in a linguistic sense
- The verb and object will correspond to words in the post and may or may not be in their lemmatized form
- An example of an event is (diagnose, condition)
- An event chain comprises two events connected by a causal relation, denoted as: ((EVENT_1, CAUSAL, EVENT_2))
- CAUSAL indicates that either EVENT_1 caused EVENT_2 or EVENT_2 caused EVENT_1
- Example chain: ((diagnose, condition), CAUSAL, (prescribe, medication))

## Generation Guidelines

- Create a single, concise sentence that clearly expresses the causal relationship
- Include all elements from the event chain
- Incorporate relevant entities that appear in the post
- Maintain the post's context and factual alignment
- Focus only on the specific causal relationship in the event chain
- Do not add information or entities not present in the post
- Keep the sentence natural in tone

## Input Format
Each generation task will include:
- **DOMAIN**: The topic area (e.g. Parkinson's Disease, Gun Control, Immigration)
- **EVENT CHAIN**: A representation of two causally connected events in the format ((VERB_1, OBJECT_1), CAUSAL, (VERB_2, OBJECT_2))
- **ARTICLE**: The complete post
- **SENTENCE_1**: The sentence in the post containing EVENT_1
- **SENTENCE_2**: The sentence in the post containing EVENT_2 (may be the same as SENTENCE_1)

## Output Format
Provide your generated sentence in this JSON structure:
```json
{
  "sentence": "Your sentence that expands on the event chain here"
}
```
