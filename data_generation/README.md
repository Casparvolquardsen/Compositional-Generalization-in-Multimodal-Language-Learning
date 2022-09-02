run "python3 data_gen_balance.py 5 demo_data -rules ./demo_data/rules.txt" in this directory for a demonstration of the data generation.

use "python3 data_gen_balance.py -h" to view the commandline arguments for data_gen_balancy.py.

use the commandline argument -rules ./path/to/file.txt to specify the action sequences you want to generate.
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
		
	

requires CoppeliaSim (https://www.coppeliarobotics.com/) and PyRep (https://github.com/stepjam/PyRep).
