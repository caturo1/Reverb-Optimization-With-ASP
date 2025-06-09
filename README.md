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

A demonstration of four examplary audio files is available in the demonstration folder by opening the 'index.html' locally in your browser. It is based on the [A/B Audio Player by mattbartley](https://github.com/mattbartley/AB-Audio-Player/tree/main) project, with just minor adjustments made by myself. The vocal files are taken from [VocalSet: A Singing Voice Dataset](https://zenodo.org/records/1193957) and the individual instrument files from the the [Musical Instrument's Sound Dataset](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset).

