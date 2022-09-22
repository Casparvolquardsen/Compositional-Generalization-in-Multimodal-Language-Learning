# Dataset generation from simulation

This directory contains the sources used to generate the Multimodal-Robot-Simulation dataset. This is implemented for [CoppeliaSim](https://www.coppeliarobotics.com "CoppeliaSim") educational version using [PyRep](https://github.com/stepjam/PyRep "PyRep GitHub"). The sources were originally provided by Aaron Eisermann and adapted by Caspar Volquardsen.

The original dataset downloaded for the experiments in the paper [1] can be accessed by members of the University of Hamburg (UHH) with the following [link](https://unihamburgde-my.sharepoint.com/:u:/g/personal/caspar_volquardsen_studium_uni-hamburg_de/EQiwFjBtBv9ClUW_429NZp0Byw79Pto7hFXSRJkXqlF_Pg?e=qgMWag "OneDrive"). If you are not from the UHH you can contact [me](caspar.volquardsen@uni-hamburg.de "caspar.volquardsen@uni-hamburg.de") via e-mail to get access.

## How to use

Run "python3 data_gen_balance.py 5 demo_data -rules ./demo_data/rules.txt" in this directory for a demonstration of the data generation.

Use "python3 data_gen_balance.py -h" to view the commandline arguments for data_gen_balancy.py.

Use the commandline argument -rules ./path/to/file.txt to specify the action sequences you want to generate.
	rules.txt defines the objects and their possible colors that will be included in the simulation.
	It is also possible to exclude certain action sequences by defining Exceptions:
		Exceptions:
		green apple - pick  #Never pick up the green apple, this also prevents the putting down action
		.* banana - left  #Never push a banana to the left
		.* - right #Push no objects to the right

Each action sequence will be saved in a seperate folder within data specified save location. An action sequence is saved 
in frames, where each frame is saved as two files, a .png file that contains the image from the vision sensor and 
a .txt file hat contains the 6 joint positions followed by the one-hot encoded word vector of the teacher output.
The word list for the one-hot vector can be found in the data.info file.
Each sequence folder contains an action.info file that tells the number of frames simulated for this action as well as 
the number of  frames simulated in the entire run of the simulator. It will also contain a line stating "Contains frames 
of unsuccessful action" if contains frames of an unsuccessful simulation attempt of an action (eg. the arm got stuck). 
In that case, it also lists the frame(s) at which the arm was reset into its default position for a repeated attempt.
		
	

Requires [CoppeliaSim](https://www.coppeliarobotics.com/) and [PyRep](https://github.com/stepjam/PyRep).
