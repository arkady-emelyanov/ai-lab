# Drops the panels whose queries use one of $drop (metric names or parts of
# them, of components switched off), then the rows left without panels.
# jq --argjson drop '["idrac_", ...]' -f drop-panels.jq dashboard.json
def disabled: ([.targets[]?.expr // empty] | join(" ")) as $e | any($drop[]; . as $m | $e | contains($m));
.panels |= (map(select(.type == "row" or (disabled | not)))
  | . as $p
  | [range(length) | select($p[.].type != "row" or (. + 1 < ($p | length) and $p[. + 1].type != "row")) | $p[.]])
