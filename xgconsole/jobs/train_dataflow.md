title: Training data flow made visible: traces and a diagram page
executor: dataflow-scout
track: research
status: done
started: 2026-08-20 11:59
updated: 2026-08-20 12:38
slurm: 
link: 
page: train_dataflow
upstreams: filter_retrain
motivation: Owner ask: understand during training which data comes in, to which functions in which file, and what is saved where. The uncapped continue-train made the cost of this opacity concrete (a 22-hour epoch diagnosed by reading /proc instead of any instrument), so the deliverable is a data-flow page grounded in real traces of the live run rather than code reading alone.
log:
- 2026-08-20 11:59 [master] Job started.
- 2026-08-20 12:03 [dataflow-scout] Measured the live training run from the outside, with reads only, so the run itself was never touched: every sample pulls about 9.2 megabytes off the shared disk, and 8.4 of those are the single conditioning file.
- 2026-08-20 12:16 [dataflow-scout] Ran the same training script locally on three shapes to time it from the inside, and confirmed the whole path end to end: it loads, trains, runs the quick check, and writes three checkpoint files of 2.6 gigabytes each per save.
- 2026-08-20 12:26 [dataflow-scout] Traced every file the training loop touches and timed each part of a step: reading and unpacking the data is about 8 percent of the time, the rest is the model itself, so the long epochs are a compute cost and not a disk cost.
- 2026-08-20 12:37 [dataflow-scout] Page published and checked live on a wide screen and a phone: it walks the whole path from the four files on disk, through the function that reads them, to the three checkpoint files a save writes.
- 2026-08-20 12:38 [dataflow-scout] Page published and checked live on a wide screen and on a phone, in both the light and dark settings.
- 2026-08-20 12:38 [dataflow-scout] The training run reads 9.23 megabytes for every shape it trains on, and 91 percent of that is a single conditioning file whose second half is a block of zeros the code throws away immediately, which is 266 gigabytes an epoch of pure waste. The expected conclusion does not follow: reading and unpacking a shape costs about 91 milliseconds out of 1.10 seconds, so tidying that file is worth doing for storage and for the shared disk, but it would take under an hour and a half off a 19 hour epoch. The epochs are long because the run pushes 63,129 shapes through the model one at a time with nothing batched and nothing read ahead. Capping the epoch length is the change that would actually help, and the setting for it already exists and is unused.
outcome: The training run reads 9.23 megabytes for every shape it trains on, and 91 percent of that is a single conditioning file whose second half is a block of zeros the code throws away immediately, which is 266 gigabytes an epoch of pure waste. The expected conclusion does not follow: reading and unpacking a shape costs about 91 milliseconds out of 1.10 seconds, so tidying that file is worth doing for storage and for the shared disk, but it would take under an hour and a half off a 19 hour epoch. The epochs are long because the run pushes 63,129 shapes through the model one at a time with nothing batched and nothing read ahead. Capping the epoch length is the change that would actually help, and the setting for it already exists and is unused.
