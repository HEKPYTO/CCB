# examples

|File|Contents|
|---|---|
|`records.jsonl`|600 records over three pairs (`a>b`, `b>c`, `a>c`), enough to certify an edge.|
|`null.jsonl`|1,000 fair records for one pair, the `ccb replay` input shape.|

Use them with the root README quickstart:

```sh
ccb certify     --input examples/records.jsonl --pair a,b
ccb leaderboard --input examples/records.jsonl
ccb replay      --input examples/null.jsonl --attack flip --budget 100 --replays 1000
ccb demo        --input examples/records.jsonl --out demo.html
```
