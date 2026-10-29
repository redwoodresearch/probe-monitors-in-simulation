#!/usr/bin/env bash
# Laptop: the census cadence for d9 (10-15 ~07:00): census_commit.sh at 09:22:30 PT and then every 25 min until 14:00 PT (d9 10-15 ~09:20: extended for ug and the sim-honest quarters),
# so a committed census <= 30 min old exists for every start. One line per run to
# nm_launch/census_loop.log. census_commit.sh holds the census lock, so it never overlaps another census.
E=$(cd "$(dirname "$0")" && pwd)
next=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 09:22:30' +%s)
END=$(TZ=America/Los_Angeles date -j -f '%Y-%m-%d %H:%M:%S' '2026-10-15 14:00:00' +%s)
while (( next <= END )); do
  while (( $(date +%s) < next )); do sleep 20; done
  out=$(perl -e 'alarm 900; exec @ARGV' bash "$E/census_commit.sh" 2>&1 | head -2 | tr '\n' ' ')
  echo "$(TZ=America/Los_Angeles date '+%F %T PT') $out" >> "$E/census_loop.log"
  next=$((next + 1500))
done
