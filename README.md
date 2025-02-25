# structured-clustering-for-narratives



## Extracting characters 

For extracting main characters from articles by prompting an llm, use the scripts in ```character/extract_characters```. For example, start an ollama server:

```console
nohup ollama serve > ollama_log_3.txt 2>&1 &
```

Then, to process the gun control corpus, run: 

```console
python3 character/extract_characters/run.py --host $host_ip --workers 8 --save_interval 50 --dataset guncontrol_processed_corpus.json --prompt_file guncontrol_default.json
```