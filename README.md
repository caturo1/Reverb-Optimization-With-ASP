# Reverb-Optimization-With-ASP
Code and misc for my bachelor thesis "Digital Signal Processing with ASP: Optimizing Audio Parameters of a Reverb Effect"


From the root directory, run the application with:
```
bash $ python -m src.application --audio-file name
```

You can set the additional flags:
* --display:
    If set, displays information about internal procedures, audio characteristics, artifacts and time statistics
* --dynamic:
    If set, this adds only relevant nogoods depending on the detected artifacts; otherwise the reverbrated audio is analyzed for any artifact in bulk mode and adds nogoods bundled

The benchmark can be run just by calling the benchmark script and writes results into ./benchmark/results:
```
bash $ ./benchmark.sh
```