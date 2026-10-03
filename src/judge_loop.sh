#!/bin/bash
# Repeatedly judge whatever conditions exist until all expected E1 conditions are judged.
source .venv/bin/activate
while true; do
  if [ -f results/e1/stories.json ]; then python src/e1_judge.py results/e1/stories.json results/e1/judged.json --quality; fi
  n=$(python -c "import json;d=json.load(open('results/e1/judged.json'));print(len(d))" 2>/dev/null || echo 0)
  [ "$n" -ge 10 ] && break
  sleep 60
done
echo JUDGE_LOOP_DONE
