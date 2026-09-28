# ALPACA Bubble Postprocessor

Postprocessing tool for extracting bubble dynamics data from simulations performed with ALPACA (Adaptive Level-set PArallel Code Alpaca), a multiresolution compressible multiphase flow solver. (see https://gitlab.lrz.de/nanoshock/ALPACA).

The script is implemented in ParaView Python and generates processed datasets, metadata, line profiles, and visualization images from ALPACA simulation outputs.

## Requirements

- ParaView 5.11.2
- Python 3
- NumPy
- Pandas
- SciPy

Install required Python packages:

```bash
pip install numpy pandas scipy
```

Run the script using ParaView's Python interpreter:

```bash
pvpython LS_postprocess.py
```

## Input Structure

The script expects an ALPACA case directory containing:

```text
case/
├── domain/
├── subdomain/
├── *.xml
└── *_monitor_quantities.csv
```

## Usage

```bash
pvpython LS_postprocess.py <inputfolder> <outputfolder> <frequency> <pressure_amplitude> <radius>
```

### Arguments

- `inputfolder` : ALPACA case directory
- `outputfolder` : Output directory
- `frequency` : Acoustic frequency [Hz]
- `pressure_amplitude` : Pressure amplitude [Pa]
- `radius`: Initial bubble radius [m]

### Example

```bash
pvpython postprocess.py \
    case    \
    results \
    100e3   \
    1.5e5   \
    4e-5
```

## Output

The output directory contains:

```text
results/
├── metadata.json
├── bubble_data.csv
├── bubble_data_medres.csv
├── bubble_data_highres.csv
├── domain_data.csv
├── timesteps_full.csv
├── timesteps_sub.csv
├── profile_x_full/
├── profile_y_full/
├── profile_x_sub/
├── profile_y_sub/
├── images_full/
└── images_sub/
```

## Author

Dániel Nagy  
Department of Hydrodynamic Systems  
Budapest University of Technology and Economics
