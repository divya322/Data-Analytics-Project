## Local installation

The goal is to create a Python environment with the required dependencies.

1. Install Miniconda (example for Linux):
```
# Download and run the installer
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts, then reload the shell
$ source ~/.bashrc
```

2. If needed, initialize the shell for conda:
```
$ conda init
# close and reopen the terminal, or:
$ source ~/.bashrc
```

3. Create and activate the environment:
```
# Create an environment named "DADH" with Python (change the version if needed)
$ conda create -n DADH python=3.14
$ conda activate DADH
```

4. Install the project's dependencies:
```
$ conda install pandas
```