#!/bin/bash
# Judge M3 conditions as they appear, until the steering conditions are judged too.
source .venv/bin/activate
while true; do
  python src/e1_judge.py results/mech/m3_stories.json results/mech/m3_judged.json --quality
  n=$(python -c "import json;print(len(json.load(open('results/mech/m3_judged.json'))))" 2>/dev/null || echo 0)
  [ "$n" -ge 11 ] && break
  sleep 120
done
echo M3_JUDGE_DONE
