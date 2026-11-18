# Research services

Services that only collect research data. They are not part of the trading install (`infra/install_host.sh` and
`infra/sync_systemd.sh` handle `infra/systemd/` only), never read a key, and run at the lowest priority.

## market-recorder

`data/downloads/market_recorder.py`: liquidations (the `forceOrder` stream; the public archive keeps none) and a
best bid/ask snapshot of every USDT perpetual every 5 minutes (the archive's bookTicker stopped 2024-03-30), into
`/root/lab_cache/market_recorder` (one file per UTC day, closed days compressed and listed in `manifest.jsonl`; every
span not recorded in `gaps.csv`; `status.json`). Data from 2026-11-13 on is forward data: research reads it only in
a study registered before it looks.

It runs from its own worktree pinned to a master commit, so branch work in the main checkout never changes it:

```bash
git -C /root/quant_system fetch origin master
git -C /root/quant_system worktree add --detach /root/qs-services origin/master      # once
cp infra/research/market-recorder.service /etc/systemd/system/ && systemctl daemon-reload
systemctl enable --now market-recorder
# update to a newer master:
git -C /root/qs-services checkout --detach origin/master && systemctl restart market-recorder
# stop for good:
systemctl disable --now market-recorder
```
