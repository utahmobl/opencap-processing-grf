# OpenCap Processing GRF

This repository enables the post-processing of human movement kinematics collected using [OpenCap](opencap.ai). You can run kinematic analyses, download multiple sessions using scripting, and run muscle-driven simulations to estimate kinetics. Building on the original OpenCap processing repository, this codebase implements the hybrid machine learning–simulation pipeline described in Miller et al., improving estimation of ground reaction forces, joint moments, and joint contact forces.

## Publications
More information is available in our [preprint](https://www.biorxiv.org/XXXX): 

Miller EY, Tan T, Falisse A, Uhlrich SD, 2025. Integrating Machine Learning with Musculoskeletal Simulation Improves OpenCap Video-Based Dynamics Estimation 

Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: Human movement dynamics from smartphone videos. PLoS Comput Biol 19(10): e1011462. https://doi.org/10.1371/journal.pcbi.1011462. *contributed equally <br> <br>


## Install requirements
### General
1. Install [Anaconda](https://www.anaconda.com/)
1. Open Anaconda prompt
2. Create environment (python 3.11 recommended): `conda create -n opencap-processing-grf python=3.11`
3. Activate environment: `conda activate opencap-processing-grf`
4. Install OpenSim: `conda install -c opensim-org opensim=4.5=py311np123`
    - Test that OpenSim was successfully installed:
        - Start python: `python`
        - Import OpenSim: `import opensim`
            - If you don't get any error message at this point, you should be good to go.
        - You can also double check which version you installed : `opensim.GetVersion()`
        - Exit python: `quit()`
    - Visit this [webpage](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53116061/Conda+Package) for more details about the OpenSim conda package.
5. (Optional): Install an IDE such as Spyder: `conda install spyder`
6. Clone the repository to your machine: 
    - Navigate to the directory where you want to download the code: eg. `cd Documents`. Make sure there are no spaces in this path.
    - Install git if you have not already: 'conda install -c conda-forge git'
    - Clone the repository: `git clone https://github.com/utahmobl/opencap-processing-grf.git`
    - Navigate to the directory: `cd opencap-processing-grf`
7. Install required packages: `python -m pip install -r requirements.txt`
8. Run `python createAuthenticationEnvFile.py`
    - An environment variable (`.env` file) will be saved after authenticating.    
    
### Simulations
1. **Windows only**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
    - The Community variant is sufficient and is free for everyone.
    - During the installation, select the *workload Desktop Development with C++*.
    - The code was tested with the 2017, 2019, and 2022 Community editions.
2. **Linux only**: Install OpenBLAS libraries
    - `sudo apt-get install libopenblas-base`

    
## Examples
- Run `batchDownload_ForGRFTrack.py` for example of how download data and predict GRFs/COPs
- Run `example_walking_opensimAD_GRF.py` for examples of how to generate hybrid simulations




