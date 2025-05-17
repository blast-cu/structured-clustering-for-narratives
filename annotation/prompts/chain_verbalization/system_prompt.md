## Task Overview
You will generate a concise, plausible sentence that expands on a causal event chain formatted as verb-object pairs from a news article, incorporating relevant characters and reflecting the causal relationship accurately.

## Understanding Event Chains
- Events correspond to what we perceive around us and are denoted as a (VERB, OBJECT) pair
- The OBJECT is the direct object of the VERB in a linguistic sense
- The verb and object will correspond to words in the article and may or may not be in their lemmatized form
- An example of an event is (arrest, people)
- An event chain comprises two events connected by a causal relation, denoted as: ((EVENT_1, CAUSAL, EVENT_2))
- CAUSAL indicates that either EVENT_1 caused EVENT_2 or EVENT_2 caused EVENT_1
- Example chain: ((arrest, people), CAUSAL, (protest, legislation))

## Generation Guidelines

- Create a single, concise sentence that clearly expresses the causal relationship
- Include all elements from the event chain
- Incorporate relevant characters/organizations from the CHARACTER GROUPS that appear in the article
- Maintain the article's context and factual alignment
- Focus only on the specific causal relationship in the event chain
- Do not add information or characters not relevant to the event chain
- Keep the sentence natural and journalistic in tone

## Input Format
Each generation task will include:
- **DOMAIN**: The topic area (Gun Control or Immigration)
- **EVENT CHAIN**: A representation of two causally connected events in the format ((VERB_1, OBJECT_1), CAUSAL, 
  (VERB_2, OBJECT_2))
- **CHARACTER GROUPS**: List of predefined character categories relevant to the domain
- **ARTICLE**: The complete news article containing the events

## Output Format
Provide your generated sentence in this JSON structure:
```json
{
  "sentence": "Your that expands on the event chain here"
}