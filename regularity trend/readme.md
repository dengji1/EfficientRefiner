# Reproducing HPWL and Regularity Trends

We provide instructions to reproduce the HPWL and Regularity trends shown in our paper.

## Initial Layouts

Initial placements for the superblue1–superblue3 circuits are generated using DreamPlace. The generated `.pl` files can be found in the `/pl` subdirectory.

## Refinement Command

To refine an initial layout (e.g., `superblue1.pl`), run the following command:

```bash
python main.py --benchmark="superblue1" --legalize=False --place_cells=False --save_curve=True --output_dir="results"
```

After the code finishes, the necessary data for plotting the HPWL and Regularity trends can be found in the log file `results/superblue1/superblue1.csv`.

We also provide the `.csv` files generated from our runs, which are used to plot the trends, in the `/logs` directory.
