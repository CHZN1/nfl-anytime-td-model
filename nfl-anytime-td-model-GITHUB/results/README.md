# Results

After all games for a week are final, grade the saved prediction snapshots with:

```bash
python grading/grade_predictions.py --season 2026 --week 1
```

The grader writes weekly single-projection and parlay result CSVs into this directory. It matches ESPN player IDs first, falls back to normalized name/team matching, and does not silently count unresolved box-score matches as losses.

Prospective 2026 results should be added only after the corresponding pregame prediction snapshot already exists.
