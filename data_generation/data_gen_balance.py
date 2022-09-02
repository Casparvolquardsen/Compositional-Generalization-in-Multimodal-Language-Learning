from pyrep.objects import VisionSensor
from pyrep.robots.arms.arm import Arm
from pyrep.objects.dummy import Dummy
from os.path import dirname, join, abspath

# import pyvirtualdisplay

import cv2, os, argparse, re, itertools

import numpy as np

import sim
from sim import Action, Push, Place, Wait, Pick, Remove

import time, datetime


def generate(save_directory, scene_file, actionManager,
             num_actions, max_null_frames,
             framerate,
             resolution, headless):
    sim.initialize(scene_file, headless)

    agent = ManipulatorPro()
    tip = agent.get_tip()
    target = Dummy("mp_target")
    vision_sensor = VisionSensor("Vision_sensor")
    vision_sensor.set_resolution(resolution)

    sim.setup_ik(agent, target)

    max_actions = num_actions  # Number of actions to generate
    start_time = time.time()  # Start time of the program

    na_frames = 0  # Counts the number of frames with the default word as output. This is used to end the program if it deadlocks
    max_na_frames = max_null_frames

    joint_backup = agent.get_joint_positions()  # reset the joint positions to these values if the arm gets stuck
    target_backup = target.get_position()  # reset target to this position of arm gets stuck

    canceled_flag = False
    resets_list = []
    cancel_counter = [0, 0]  # 1. # of actions containing canceled frames 2. # of canceled actions
    cancel_list = []  # saves which actions have been canceled

    frame_counter = [0, 0]  # 1. frames in current directory 2. all frames
    sentence = []

    save_intervall = 1 / framerate
    save_intervall_running = save_intervall

    i_action = 0  # How many actions have been generated
    action_directory = init_subdir(save_directory, i_action, frame_counter, sentence)

    actionManager.spawn()
    actionManager.execute()

    running = True
    while running:
        s = sim.sim_update()  # perform a simulation step and receive action status
        if s == Action.SAVE:
            close_subdir(action_directory, i_action, frame_counter, canceled_flag,
                         cancel_counter, cancel_list, sentence, resets_list)
            canceled_flag = False
            resets_list = []

            i_action += 1
            if i_action < max_actions:
                action_directory = init_subdir(SAVE_DIRECTORY, i_action, frame_counter, sentence)
                printlog("New subdir:", action_directory)

                if i_action % args.intervall == 0:
                    save_info(save_directory, actionManager, resolution,
                              frame_counter, cancel_counter, cancel_list, joint_backup,
                              i_action, interrupted=na_frames > max_na_frames)

                actionManager.advance()
                actionManager.spawn()
                actionManager.execute()
            else:
                running = False

        elif s == Action.CANCELED:
            print("canceled")
            cancel_counter[1] += 1
            canceled_flag = True
            target.set_position(target_backup)
            agent.set_joint_positions(joint_backup)
            resets_list.append(frame_counter[0])

            actionManager.force_reset()
            actionManager.execute()

        save_intervall_running -= sim.delta_time
        if save_intervall_running <= 0:
            capture_perception(action_directory, agent, vision_sensor, frame_counter, sentence)
            save_intervall_running = save_intervall

            if sim.Teacher.getCurrentWordIndex() == 0:
                na_frames += 1
            else:
                na_frames = 0

        if na_frames > max_na_frames:
            running = False
            print("Simulation stopped after", na_frames, "frames of N/A")

    close_subdir(action_directory, i_action, frame_counter, canceled_flag,
                 cancel_counter, cancel_list, sentence, resets_list)
    canceled_flag = False
    resets_list = []

    end_time = time.time()
    print("Simulated the arm performing", i_action, "actions for a total duration of",
          time.strftime("%H:%M:%S", time.gmtime(sim.total_time)))
    print("The simulator was running for", time.strftime("%H:%M:%S", time.gmtime(end_time - start_time)))

    save_info(save_directory, actionManager, resolution,
              frame_counter, cancel_counter, cancel_list, joint_backup,
              i_action, interrupted=na_frames > max_na_frames)

    print("ENDED")

    sim.stop()
    print("SIMULATION STOPPED")
    """
    if HEADLESS:
        print("STOPPING VIRTUAL DISPLAY...")
        display.stop()
        print("STOPPED VIRTUAL DISPLAY")
    """


class ManipulatorPro(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, 'mp', num_joints=6, base_name="mp")


"""
Capture the current frame and save it to the specified directory

The current image from the vision sensor will be saved to a file frame_xxxxxx.png
The joint positions and the current one-hot-encoded language vector are stored together in frame_xxxxxx.txt
"""


def capture_perception(save_directory, agent, vision_sensor, frame_counter, sentence):
    image = vision_sensor.capture_rgb()
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(save_directory, "frame_" + "{:06d}".format(frame_counter[0]) + ".png"), image)
    joints = agent.get_joint_positions()
    language = sim.Teacher.getCurrentOneHot()

    if language[0] != 1:
        sentence.append(sim.Teacher.getCurrentWordIndex())
    np.savetxt(os.path.join(save_directory, "frame_{:06d}".format(frame_counter[0]) + ".txt"), joints + language)

    frame_counter[0] += 1
    frame_counter[1] += 1


"""
Initialize a new sub-directory to store an action into

The global frame_counter and sentence variables are reset.

Returns:
    The path to the newly created sub-directory
"""


def init_subdir(save_directory, i, frame_counter, sentence):
    printlog("Initializing action directory")
    frame_counter[0] = 0
    sentence.clear()
    action_directory = os.path.join(save_directory, "sequence_{:04d}".format(i))
    if not os.path.exists(action_directory):
        os.mkdir(action_directory)
    printlog("Returning action dir:", action_directory)
    return action_directory


"""
Close a sub-directory

Saves information about the finished action into the directory
"""


def close_subdir(directory, i_action, frame_counter, canceled_flag, cancel_counter,
                 cancel_list, sentence, resets_list):
    np.savetxt(os.path.join(directory, "action.info"), [frame_counter[0], canceled_flag], fmt="%d")
    with open(os.path.join(directory, "action.info"), "w") as file:
        file.writelines(["Frames: ", str(frame_counter)])
        if canceled_flag:
            file.writelines(["\nContains frames of unsuccessful action"])
            file.writelines(["\nResetting joints after frames: ", str(resets_list)])
        file.writelines(["\nSentence: ", sim.Teacher.to_sentence(sentence)])
    if canceled_flag:
        cancel_counter[0] += 1
        cancel_list.append(i_action - 1)  # -1 to match folder names since i_action starts at 1
    # with open(os.path.join(directory, "action.info"), "w") as file:
    # file.write("Frames: " + str(frame_counter - frame_start))


"""
Save information about the generated data into data.info file
"""


def save_info(directory, actionManager, resolution, frame_counter,
              cancel_counter, cancel_list, joint_backup, i_action, interrupted=False):
    with open(os.path.join(directory, "data.info"), 'w') as file:
        file.write("Timestamp: " + str(datetime.datetime.today()))
        if interrupted:
            file.write("\nProgram did not finish")
        file.writelines(["\nFrames: " + str(frame_counter[1]), "\nWord list: ", sim.Teacher.dic.__str__()])

        file.write("\nDefined exceptions for:")
        for i, exc in enumerate(actionManager.exceptions[0]):
            file.write("\n\t" + exc + " - " + str(actionManager.exceptions[1][i]))

        file.write("\n\nDefined objects and colors:")
        for i, o in enumerate(actionManager.combinations):
            file.write("\n\t" + str(o))

        file.writelines(["\n\nImage Shape: ", str(resolution[0]) + " x " + str(resolution[1])])
        file.writelines(["\nInput Shape: ", str(6 + len(sim.Teacher.dic) + resolution[0] * resolution[1] * 3)])
        file.writelines(["\nOutput Shape: ", str(len(sim.Teacher.dic))])
        file.writelines(["\nActions canceled: ", str(cancel_counter[0]) + " (" + str(cancel_counter[1]) + ") - " + str(
            cancel_counter[0] / i_action) + "%"])
        file.writelines(["\nJoints reset position: ", str(joint_backup)])
        file.writelines(["\n\nAction distribution:"] + ["\n\t" + e + ": " + str(sim.Teacher.actions[e]) for e in
                                                        sorted(sim.Teacher.actions)])

    np.savetxt(os.path.join(directory, "errors.np"), cancel_list, fmt="%d")


def printlog(*args):
    if VERBOSE:
        print(*args)


class ActionManager:

    def __init__(self, combinations, exceptions, max_objects_in_scene):
        self.combinations = combinations
        self.max_objects = max_objects_in_scene

        self.cur_objects = []
        self.cur_actions = []

        self.c0 = np.random.permutation(self.combinations)
        self.c1 = np.random.permutation(self.combinations)

        self.exceptions = exceptions

    def spawn(self):
        printlog("spawn")
        for i in range(len(self.cur_objects), self.max_objects):
            obj = sim.SceneObject.create(self.c0[0][1], self.c0[0][0])
            self.c0 = self.c0[1:]
            if len(self.c0) < 1:
                self.c0 = self.c1
                self.c1 = np.random.permutation(self.combinations)
            # pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            # Action.add(sim.CombinedAction([Place(obj, pos), Wait(np.random.uniform(.3, .5))]), non_blocking = True)

            a = [0, 1, 2]
            for i, exc in enumerate(self.exceptions[0]):  # Remove specified exceptions from the action list
                if re.match(exc, obj.description().replace(",", " ")):
                    for v in self.exceptions[1][i]:
                        try:
                            a.remove(v)
                        except ValueError:
                            pass
            np.random.shuffle(a)
            self.cur_actions.append(a)
            self.cur_objects.append(obj)

    def advance(self):
        printlog("advancing")

        self.cur_actions[0].pop(0)

        if len(self.cur_actions[0]) == 0:  # all actions executed
            Action.add(Remove(self.cur_objects.pop(0)))
            self.cur_actions.pop(0)

    def execute(self):
        printlog("exec")

        for obj in self.cur_objects:
            if not obj.active:
                pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
                Action.add(sim.CombinedAction([Place(obj, pos), Wait(np.random.uniform(.3, .5))]), non_blocking=True)

        printlog("Entering execute")
        if self.cur_actions[0][0] == -1:
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            Action.add(sim.PutDown(self.cur_objects[0], pos), signal=Action.SAVE)
            printlog("Execute PutDown")
        elif self.cur_actions[0][0] == 0:
            Action.add(Pick(self.cur_objects[0]), signal=Action.SAVE)
            self.cur_actions[0].insert(1, -1)  # An object that has been picked up will be put down in the next action
            printlog("Execute PickUp")
        elif self.cur_actions[0][0] == 1:
            Action.add(Push(self.cur_objects[0], direction="left", velocity=np.random.uniform(1, 1.5)),
                       signal=Action.SAVE)
            printlog("Execute PushLeft")
        elif self.cur_actions[0][0] == 2:
            Action.add(Push(self.cur_objects[0], direction="right", velocity=np.random.uniform(1, 1.5)),
                       signal=Action.SAVE)
            printlog("Execute PushRight")
        printlog("Exiting execute")

    """
    Reposition all objects in the scene
    """

    def force_reset(self):
        printlog("Entering reset")
        for obj in self.cur_objects:
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            action = sim.CombinedAction([Place(obj, pos)])
            action.append(Wait(np.random.uniform(0.2, 0.3)))
            Action.add(action, non_blocking=True)
        printlog("Exiting reset")


"""
Manages the objects to create a balanced data set.
It generates all the possible color - object combinations in a randomized order. For each of
the pairs it then performs the actions push left, push right and pick up in a random
order. The pick up action is always immediately followed by the put down action
"""


class OldObjectManager:

    def __init__(self, object_list, color_list, action_manager, max_objects_in_scene):
        self.object_list = object_list
        self.color_list = color_list
        self.action_manager = action_manager
        self.max_objects = max_objects_in_scene

        self.reset()

        self.objects = []  # Active objects in the scene
        self.actionTracker = []  # Tracks remaining actions for the active objects

    def start(self):
        for i in range(min(self.max_objects, len(self.pairs))):
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            pair = self.pairs.pop(0)
            obj = self.object_list[pair[0]](pos, color=self.color_list[pair[0]][pair[1]])
            action = sim.CombinedAction([Place(obj, pos)])
            action.append(Wait(np.random.uniform(0.1, 0.2)))
            Action.add(action, non_blocking=True)

            self.objects.append(obj)
            self.actionTracker.append(self.action_manager.get(obj))
        Action.add(Wait(np.random.uniform(.3, .5)))

    def next_action(self):
        self.actionTracker[0].pop(0)
        for i, obj in enumerate(self.objects):
            if not obj.active and len(
                    self.actionTracker[i]) > 0:  # Reset objects that are not on the table with actions remaining
                pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
                action = sim.CombinedAction([Place(obj, pos)])
                action.append(Wait(np.random.uniform(0.3, 0.5)))
                Action.add(action, non_blocking=True)
        for i, obj in enumerate(self.objects):
            if len(self.actionTracker[i]) == 0:  # Replace objects with no remaining actions
                self.replace(i)

    """
    Reposition all objects in the scene
    """

    def force_reset(self):
        printlog("Entering reset")
        for obj in self.objects:
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            action = sim.CombinedAction([Place(obj, pos)])
            action.append(Wait(np.random.uniform(0.2, 0.3)))
            Action.add(action, non_blocking=True)
        printlog("Exiting reset")

    """
    Execute the next action
    """

    def execute(self):
        printlog("Entering execute")
        if self.actionTracker[0][0] == -1:
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            Action.add(sim.PutDown(self.objects[0], pos), signal=Action.SAVE)
            printlog("Execute PutDown")
        elif self.actionTracker[0][0] == 0:
            Action.add(Pick(self.objects[0]), signal=Action.SAVE)
            self.actionTracker[0].insert(1, -1)  # An object that has been picked up will be put down in the next action
            printlog("Execute PickUp")
        elif self.actionTracker[0][0] == 1:
            Action.add(Push(self.objects[0], direction="left", velocity=np.random.uniform(1, 1.5)), signal=Action.SAVE)
            printlog("Execute PushLeft")
        elif self.actionTracker[0][0] == 2:
            Action.add(Push(self.objects[0], direction="right", velocity=np.random.uniform(1, 1.5)), signal=Action.SAVE)
            printlog("Execute PushRight")
        printlog("Exiting execute")

    """
    Replace the i-th object with the next object in the list
    """

    def replace(self, i):
        printlog("Entering replace", i)
        Action.add(Remove(self.objects[i]))
        self.objects.pop(i)
        self.actionTracker.pop(i)
        if len(self.pairs) == 0:
            self.reset()

        if len(self.pairs) > 0:
            pos = [np.random.uniform(*sim.table_bounds[0]), np.random.uniform(*sim.table_bounds[1]), 3]
            pair = self.pairs.pop(0)
            obj = self.object_list[pair[0]](pos, color=self.color_list[pair[0]][pair[1]])
            action = sim.CombinedAction([Place(obj, pos)])
            action.append(Wait(np.random.uniform(0.3, 0.6)))
            Action.add(action, non_blocking=True)
            self.actionTracker.append(self.action_manager.get(obj))
            self.objects.append(obj)
        # elif len(self.objects) == 0:
        #    self.reset()
        #    self.start()
        printlog("Exiting replace")

    """
    Generate color + objects pairs and randomize their order
    """

    def reset(self):
        printlog("Entering reset")
        self.pairs = [(i, j) for i in range(len(self.object_list)) for j in range(len(self.color_list[i]))]
        self.pairs = np.random.permutation(self.pairs)
        self.pairs = [tuple(x) for x in self.pairs]
        printlog("Exiting reset")


class OldActionManager:

    def __init__(self, exception_patterns=[], exceptions=[]):
        self.num_actions = 3
        self.exc_patterns = [re.compile(x) for x in exception_patterns]
        self.exceptions = exceptions  # Do not push bananas to the left, Do not pick up green rings

    """
    Get list of actions for the object
    """

    def get(self, obj):
        a = [i for i in range(self.num_actions)]
        for i, e in enumerate(self.exc_patterns):
            if e.match(obj.description.replace(",", " ")):
                for x in self.exceptions[i]:
                    a.remove(x)
        return list(np.random.permutation(a))


argparser = argparse.ArgumentParser(description="simulate actions")
argparser.add_argument("actions", metavar="N", type=int, help="Number of actions to be simulated")
argparser.add_argument("save_directory", help="Directory to which frames are saved")
argparser.add_argument("-rules", default=None, help="File that contains the data generation rules")
argparser.add_argument("-o", "--max_objects", type=int, default=2,
                       help="Maximum number of objects that can be in the scene")
argparser.add_argument("-framerate", default=5, type=int, help="Set the framerate")
argparser.add_argument("-x", "--resolution_X", default=398, type=int, help="Set the image resolution")
argparser.add_argument("-y", "--resolution_Y", default=224, type=int, help="Set the image resolution")
argparser.add_argument("-headless", dest="headless", action="store_true", help="Run on a headless machine")
argparser.add_argument("-intervall", default=100, type=int, help="Print stats every i (default=100) frames")
argparser.add_argument("-na", "--max_null_frames", default=200, type=int,
                       help="Maximum number of frames without a word being spoken before the program is stopped")
argparser.add_argument("-verbose", action="store_true")
argparser.set_defaults(color=False, headless=False, verbose=False)

args = argparser.parse_args()

SAVE_DIRECTORY = args.save_directory
if os.listdir(SAVE_DIRECTORY):
    print(os.listdir(SAVE_DIRECTORY))
    print("SAVE DIRECTORY NOT EMPTY")

import json

with open(os.path.join(SAVE_DIRECTORY, "commandline_args.txt"), "w") as file:
    json.dump(args.__dict__, file, indent=2)

import shutil

shutil.copy(__file__, os.path.join(SAVE_DIRECTORY, os.path.basename(__file__)))

HEADLESS = args.headless
VERBOSE = args.verbose

SCENE_FILE = join(dirname(abspath(__file__)),
                  'Env3.ttt')

# object_list = [sim.Apple, sim.Banana, sim.Cup, sim.Ball, sim.Book, sim.Star, sim.Bottle, sim.Pylon, sim.Ring] # Which objects to use
# color_list = [[]] * len(object_list)

combinations = [("red", "apple"), ("green", "apple"), ("white", "football")]

exc_patterns = []
exc = []

if args.rules is not None:
    color_object_combinations = []
    with open(args.rules, "r") as file:
        obj_lines, exc_lines = file.read().split("Exceptions:")
        obj_lines = obj_lines.strip().splitlines()
        exc_lines = exc_lines.strip().splitlines()
        for line in obj_lines:
            label, colors = line.split(" - ")
            color_object_combinations += list(itertools.product(colors.split(" "), [label]))

        for line in exc_lines:
            pattern, exceptions = line.split(" - ")
            print(pattern)
            print(exceptions)
            exc_patterns.append(pattern.strip("\""))
            exc.append(list(map(lambda x: {"left": 1, "right": 2, "pick": 0}[x], exceptions.split())))

    print(color_object_combinations)
actionManager = ActionManager(color_object_combinations, (exc_patterns, exc), args.max_objects)
# objectManager = ObjectManager(object_list, color_list, actionManager, max_objects_in_scene=args.max_objects)

if HEADLESS:
    import pyvirtualdisplay

    with pyvirtualdisplay.Display(visible=False, backend="xvfb", size=(128, 72)) as display:
        generate(args.save_directory, SCENE_FILE, actionManager,
                 args.actions, args.max_null_frames,
                 args.framerate, (args.resolution_X, args.resolution_Y), HEADLESS)
else:
    generate(args.save_directory, SCENE_FILE, actionManager,
             args.actions, args.max_null_frames,
             args.framerate, (args.resolution_X, args.resolution_Y), HEADLESS)
"""
display = None

if HEADLESS:
    import pyvirtualdisplay
    print("STARTING VIRTUAL DISPLAY")
    display = pyvirtualdisplay.Display(visible=False, backend="xvfb", size=(128, 72))
    display.start()
"""
