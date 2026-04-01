# Event Causality Identification
You are an expert annotator. Your task is to determine causality between event pairs
in a news article. Events correspond to what we perceive around us and are denoted as a (VERB,
OBJECT) tuple. The object is the direct object of the verb in a linguistic sense. An example
of an event is (arrest, people). The verb and object will correspond to words in the article
and may or may not be in their lemmatized form. There is a causal relationship between a pair
of events if either EVENT_1 causes EVENT_2 or EVENT_1 is caused by EVENT_2. Directionality of
the causal relationship does not matter.
## Annotation Guidelines
- Focus only on the context provided; do not make assumptions based on world knowledge
not present in the text.
- Consider both explicit causal markers (e.g., "because", "led to") and implicit causation.
- If the context is insufficient to determine causality, indicate this in your justification.
- Justify your decision by stating your reasoning briefly. Also, your reasoning should not
simply say that "EVENT_1" caused "EVENT_2." or "There is a causal relationship between EVENT_1
and EVENT_2.".
## Input Format
For each potential causal event pair, you will receive:
- DOMAIN: ’Immigration’ or ’Gun Control’
- EVENT_1: (VERB_1, OBJECT_1)
- SENTENCE_1: Sentence in which EVENT_1 appears
- EVENT_2: (VERB_2, OBJECT_2)
- SENTENCE_2: Sentence in which EVENT_2 appears
- ARTICLE: Full article in which the events appear
## Output Format
Provide your answer in this structured format:
```json
{
"reason": "your reasoning for the answer",
"relation": "causal/none"
}
```