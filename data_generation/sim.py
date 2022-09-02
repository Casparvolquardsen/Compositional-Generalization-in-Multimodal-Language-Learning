#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyrep.pyrep import PyRep
from pyrep.objects import Shape
import numpy as np
import pandas
import PIL.Image
import math, random, re
import os.path

pr = None

delta_time = 0 #time step simulated in pr.step()
total_time = 0 #total time in simulation

#agent = None
tip = None
target = None

table_bounds = [(0.7, 3.0), (-2.25, 2.25), (2.7, 4.2)] # define area where objects can be placed

"""
Initialize pyrep
"""
def initialize(scene_file, headless=True):
    global pr
    pr = PyRep()
    pr.launch(scene_file, headless=headless)
    pr.start()
    
    #generate_colored_objects((Book, Ball, Bottle, Pylon, Apple), force_reload=True)


def setup_ik(agent, target_dummy):
    global tip, target#, tip_collision_box
    tip = agent.get_tip()
    #tip_collision_box = agent.get_object("mp_link6_visible")
    target = target_dummy
    
"""
end coppeliasim
"""
def stop():
    global pr
    pr.stop()
    pr.shutdown()
    
"""
Performs a simulation step by calling the update function of the Action class.
Returns the status received from the action update.

delta_time is updated and can be used globally, e.g. to make time-dependent translations

"""
def sim_update():
    global delta_time, total_time
    working = Action.update(delta_time)
    pr.step() # imulation step
    delta_time = pr.get_simulation_timestep()
    total_time += delta_time
    SceneObject.update()
    Teacher.update()
    return working

"""
Class containing information about the state of the interactable objects in the scene.

"""
class SceneObject():
    scene_objects = [] # List of all the interactable objects in the scene.
    object_directory = "./Objects"
    gen_directory = "./Objects/generated_objects"
    
    """
    label: label used in Teacher output
    color: String from colors list defined in this file
    convex_hull_file: Can be used to load a separate object file that will be used
        for the dynamics simulation
    """
    def __init__(self, label, obj_file, convex_hull_file = "", requires_textures = False, position=[0,0,0]):
        SceneObject.scene_objects.append(self)
        self.label = label
        self.lang_color = ""
        #self.description = self.color + "," + self.label
        self.active = False # Set inactive if object leaves table / is out of reach, so actions can be canceled
        self.using_convex_hull = False
        self.requires_textures = requires_textures
        
        if requires_textures:
            # Get texture path
            with open(os.path.join(SceneObject.object_directory, obj_file.replace(".obj", ".mtl"))) as mat_file:
                lines = mat_file.readlines()
                self.texture_path = os.path.join(SceneObject.object_directory, lines[-1].split(" ")[-1].strip())
        
        """
        If a 
        """
        if convex_hull_file != "":
            self.using_convex_hull = True
            
            vis = Shape.import_shape(os.path.join(self.object_directory, obj_file))
            vis.set_renderable(False)
            vis.set_dynamic(False)
            vis.set_detectable(False)
            vis.set_collidable(False)
            vis.set_measurable(False)
            vis.set_respondable(False)
            vis.set_position([0,0,0])
            
            self.shape = Shape.import_mesh(os.path.join(self.object_directory, convex_hull_file))
            self.shape.set_renderable(False)
            self.shape.set_position([0,0,0])
            
            vis.set_parent(self.shape)
            
            self.visual_shape = vis
            
        else:
            self.shape = Shape.import_shape(os.path.join(self.object_directory, obj_file))
            
        self.shape.set_renderable(False)
        self.shape.set_dynamic(False)
        self.shape.set_detectable(False)
        self.shape.set_collidable(False)
        self.shape.set_measurable(False)
        self.shape.set_respondable(False)
        
        self.shape.set_position(position)
        
    def description(self):
        return self.lang_color + "," + self.label
        
    def create(label, color):
        presets = {"apple": ["apple.obj", "apple_collision.stl", ["red", "green"], True],
                   "banana": ["banana.obj", "", ["yellow", "green", "brown"], False],
                   "cup": ["cup_01.obj", "cup_collision.stl", ["blue", "white", "red", "brown", "yellow", "green"], False],
                   "football": ["football.obj", "", ["white", "blue", "red", "yellow", "green"], True],
                   "book": ["book.obj", "book_collision.stl", ["blue", "red", "brown", "yellow", "green"], True],
                   "star": ["star.obj", "star_collision.stl", ["yellow", "red", "green", "blue", "white"], False],
                   "bottle": ["bottle.obj", "bottle_collision.stl", ["green", "red", "blue", "yellow", "brown"], True],
                   "pylon": ["pylon.obj", "pylon_collision.stl", ["red", "green", "blue", "yellow", "brown"], True],
                   "ring": ["ring.obj", "ring_collision.stl", ["brown", "green", "blue", "yellow", "white", "red"], False]}
        if color == "random":
            color = np.random.choice(presets[label][2])
        elif color == "default":
            color = presets[label][2][0]
            
        obj = SceneObject(label, presets[label][0], convex_hull_file = presets[label][1], requires_textures = presets[label][3])
        
        obj.set_color(Colors.get(color, True), color)
        
        return obj
        
    def activate(self):
        self.active = True
        if self.using_convex_hull:
            for o in self.shape.get_objects_in_tree():
                o.set_renderable(True)
        else:
            self.shape.set_renderable(True)
        self.shape.set_dynamic(True)
        self.shape.set_detectable(True)
        self.shape.set_collidable(True)
        self.shape.set_measurable(True)
        self.shape.set_respondable(True)
        
    def delete(self):
        for obj in self.shape.get_objects_in_tree():
            obj.remove()
        self.shape.remove()
        SceneObject.scene_objects.remove(self)
        
    """
    Access a SceneObject by its name
    """
    def get(name):
        name = name.lower()
        for o in SceneObject.scene_objects:
            if o.label.lower() == name:
                return o
    
    """
    Change the color of the object.
    If the object uses no textures, the base color is changed to the provided values
    via the set_color function of Shape.
    If the object is textured, the texture is replaced by a new texture that is generated
    by mixing the original texture with a colored background. Use the alpha values
    of the original texture to influence the mixing results.
    
    Params:
        color : The RGB color values
        color_string : The word used to describe the color
    """
    def set_color(self, color, color_string = ""):
        print("COLOR;", color)
        self.lang_color = color_string #provide color name
        
        if self.requires_textures:
            vis = self.shape
            if self.using_convex_hull:
                vis = self.visual_shape
            tex = PIL.Image.open(self.texture_path)
            bg = tex.copy()
            bg.paste(tuple(color), [0, 0, tex.size[0], tex.size[1]])
            mix = PIL.Image.alpha_composite(bg, tex)
            #mix.save(self.texture_path.replace(SceneObject.object_directory, SceneObject.gen_directory))
            #_, new_tex = pr.create_texture(self.texture_path.replace(SceneObject.object_directory, SceneObject.gen_directory), interpolate=False, decal_mode=True,repeat_along_u=True)
            tex_coords = vis.get_shape_viz(0).texture_coords
            vis.remove_texture()
            
            vis.apply_texture(tex_coords, np.asarray(mix.resize((512, 512))), is_rgba = True, flipv = True)
            
        else:
            # changed from 0-255 to 0.0-1.0 for correct color rendering.
            color_transformed = [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0]
            self.shape.set_color(color_transformed)
            #super().set_color(colors[color])
            for obj in self.shape.get_objects_in_tree():
                obj.set_color(color_transformed)
            
    def get_position(self):
        return self.shape.get_position()
    
    def set_position(self, position):
        self.shape.set_position(position)
        
    """
    Returns all SceneObjects with active = True
    """
    def getActiveObjects():
        return [x for x in SceneObject.scene_objects if x.active]
    
    """
    Returns all SceneObjects with active = False
    """
    def getInactiveObjects():
        return [x for x in SceneObject.scene_objects if not x.active]
    
    """
    Check on all SceneObjects
    """
    def update():
        for obj in SceneObject.scene_objects:
            obj.check_if_on_table()
            
    """
    Check if object left table and set inactive if true
    """
    def check_if_on_table(self):
        if self.active:
            if not (table_bounds[0][0] - 1.2 < self.get_position()[0] < table_bounds[0][1] + 1.2 and
                    table_bounds[1][0] - 1.2 < self.get_position()[1] < table_bounds[1][1] + 1.2 and
                    table_bounds[2][0] - 1.2 < self.get_position()[2] < table_bounds[2][1] + 15): # Not using the exact boundaries works better for me
                self.active = False
                #Action.addSubsequent(Place(random.choice([Apple, Banana, Cup])), Remove(self))

class Colors():
    colors = {"red": [.95, 0, 0], "green": [0, .95, 0], "blue": [0, 0, .95],
              "yellow": [.85, .85, 0], "white": [.95, .95, .95], "brown": [.5, .2, 0]}
    
    def names():
        return list(Colors.colors.keys())
    
    def get(name, as_int=False):
        if as_int:
            c = Colors.colors[name].copy()
            return [int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)]
        return Colors.colors[name]


"""
Obsolete
Used to generate .obj files with colored textures and save them
"""
def generate_colored_object(cls, color, force_reload=True):
    import shutil
    if not os.path.exists(SceneObject.gen_directory):
        os.mkdir(SceneObject.gen_directory)
    #for cls in classes:
    #    for color in cls.color_list:
            #p = cls.obj_path.replace(".obj", "_" + color + ".obj")
    obj_c = os.path.join(cls.gen_directory, color + "_" + cls.obj_file)
    if not os.path.exists(obj_c) or force_reload:
        obj = os.path.join(cls.object_directory, cls.obj_file)
        shutil.copyfile(obj, obj_c)
        
        with open(obj, "r") as file:
            data = file.readlines()

        mtl_file = color + "_" + cls.obj_file.replace(".obj", ".mtl")
        with open(obj_c, "w") as file:
            data[2] = "mtllib " + mtl_file + "\n"
            file.writelines(data)#.replace(".mtl", "_" + color + ".mtl"))
    
    mtl_c = obj_c.replace(".obj", ".mtl") #os.path.join(cls.gen_directory, mtl_file)# cls.obj_path.replace(".obj", "_"+color+".mtl")
    if not os.path.exists(mtl_c) or force_reload:
        mtl = obj.replace(".obj", ".mtl")
        shutil.copyfile(mtl, mtl_c)
    
        with open(mtl, "r") as file:
            lines = file.readlines()
            #tex_pth = re.sub("/.*\.obj", "/" +lines[-1].split(" ")[-1].strip(), cls.obj_path)
            tex_pth = os.path.join(cls.object_directory, lines[-1].split(" ")[-1].strip())
            lines[-1] = re.sub("\s.*\.png", "  "+color+"_"+cls.obj_file.replace(".obj", ".png"), lines[-1])
            
        with open(mtl_c, "w") as file:
            file.writelines(lines)
            
        tex_c = mtl_c.replace(".mtl", ".png")
            
        tex = PIL.Image.open(tex_pth)
        bg = tex.copy()
        bg.paste(tuple(Colors.get(color, True)), [0, 0, tex.size[0], tex.size[1]])
        mix = PIL.Image.alpha_composite(bg, tex)
        mix.save(os.path.join(tex_c))

"""
class Apple(SceneObject):
    default_color = "red"
    color_list = ["red", "green"]
    obj_file = "apple.obj"
    collision_file = "apple_collision.stl"
    label = "apple"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        generate_colored_object(Apple, self.color)
        super().__init__(Apple.label, obj_file=os.path.join("generated_objects", self.color + "_" + self.obj_file), convex_hull_file=self.collision_file, position = position)
        
class Banana(SceneObject):
    default_color = "yellow"
    color_list = ["yellow", "green", "brown"]
    obj_file = "banana.obj"
    label = "banana"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        super().__init__(Banana.label, obj_file=self.obj_file, position=position)
        self.set_color(Colors.get(self.color))
        
class Cup(SceneObject):
    default_color = "brown"
    color_list = ["blue", "white", "red", "brown", "yellow", "green"]
    obj_file = "cup_01.obj"
    hull_file = "cup_collision.stl"
    label = "cup"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        super().__init__(self.label, self.obj_file, convex_hull_file=self.hull_file, position=position)
        self.set_color(Colors.get(self.color))
        
class Ball(SceneObject):    
    default_color = "white"
    color_list = ["blue", "white", "red", "yellow", "green"]
    obj_file = "football.obj"
    label = "football"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        generate_colored_object(Ball, self.color)
        super().__init__(self.label, obj_file=os.path.join("generated_objects", self.color + "_" + self.obj_file), position=position)
        
class Book(SceneObject):
    default_color = "brown"
    color_list = ["blue", "red", "brown", "yellow", "green"]
    obj_file = "book.obj"
    collision_file = "book_collision.stl"
    label = "book"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        generate_colored_object(Book, self.color)
        super().__init__(self.label, obj_file=os.path.join("generated_objects", self.color + "_" + self.obj_file), convex_hull_file=self.collision_file,
             position=position)
        self.set_color(Colors.get("white"))
        
        
class Star(SceneObject):
    default_color = "yellow"
    color_list = ["red", "green", "blue", "yellow", "white"]
    obj_file = "star.obj"
    collision_file = "star_collision.stl"
    label = "star"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        super().__init__(self.label, obj_file=self.obj_file, convex_hull_file=self.collision_file, position=position)
        self.set_color(Colors.get(self.color))
        
class Bottle(SceneObject):
    default_color = "green"
    color_list = ["red", "green", "blue", "yellow", "brown"]
    obj_file = "bottle.obj"
    collision_file = "bottle_collision.stl"
    label = "bottle"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        generate_colored_object(Bottle, self.color)
        super().__init__(self.label, obj_file=os.path.join("generated_objects", self.color + "_" + self.obj_file), convex_hull_file=self.collision_file, position=position)
        self.set_color(Colors.get("white"))
        
class Pylon(SceneObject):
    default_color = "red"
    color_list = ["red", "green", "blue", "yellow", "brown"]
    obj_file = "pylon.obj"
    collision_file = "pylon_collision.stl"
    label = "pylon"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        generate_colored_object(Pylon, self.color)
        super().__init__(self.label, os.path.join("generated_objects", self.color + "_" + self.obj_file), 
              convex_hull_file=self.collision_file, position=position)
        self.set_color(Colors.get("white"))
        
class Ring(SceneObject):
    default_color = "green"
    color_list = ["red", "green", "blue", "yellow", "white", "brown"]
    obj_file = "ring.obj"
    collision_file = "ring_collision.stl"
    label = "ring"
    
    def __init__(self, position, color="random"):
        if color == "random":
            self.color = np.random.choice(self.color_list)
        elif color == "default":
            self.color = self.default_color
        else:
            self.color = color
        super().__init__(self.label, self.obj_file, convex_hull_file=self.collision_file, position=position)
        self.set_color(Colors.get(self.color))
"""
class Teacher():
    dic = ["N/A", "put down", "picked up", "pushed left", "pushed right", "dropped", 
           "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
           "red", "green", "blue", "yellow", "white", "brown"] #Dictionary
    actions = {} # Dictionary to count individual occurences of actions
    queued = [] # Queue of all the words yet to say
    time_per_word = 0.5 # Should not be lower than the recording framerate
    _time = time_per_word
    
    """
    Say some words
    """
    def say(text):
        for t in text.split(","):
            if t != "":
                Teacher.queued.append(t.lower())
    
    def to_sentence(array):
        sentence = ""
        for i in pandas.unique(array):
            sentence += Teacher.dic[i] + " "
        return sentence
    
    """
    count an occurence of an action
    """
    def countAction(text):
        if not Teacher.actions.get(text):
            Teacher.actions[text] = 1
        else:
            Teacher.actions[text] += 1
            
    def isTalking():
        return len(Teacher.queued) > 0
    
    """
    Get index of the current word in Teacher.dic
    """
    def getCurrentWordIndex():
        if len(Teacher.queued) > 0:
            #print(Teacher.dic[Teacher.dic.index(Teacher.queued[0])])
            return Teacher.dic.index(Teacher.queued[0])
        return 0
    
    """
    Get current word as one-hot-encoded vector (Based on the word layout in Teacher.dic)
    """
    def getCurrentOneHot():
        v = [0] * len(Teacher.dic)
        v[Teacher.getCurrentWordIndex()] = 1
        return v
    
    """
    Update the word that is currently being spoken
    """
    def update():
        global delta_time
        if Teacher.queued:
            Teacher._time -= delta_time
            if Teacher._time < 0:
                Teacher._time = Teacher.time_per_word
                Teacher.queued.pop(0)

                

"""
Manages actions in a queue system
"""
class Action():
    STEPPED = 0
    FINISHED = 1
    CANCELED = 2
    PAUSED = 3
    BLOCKED = 4
    SAVE = 5
    PAUSE = False
    WAIT = False
    action_queue = [] # Action queue
    blocking_flags_queue = []
    signal_queue = []
    
    #def __init__(self):
        #Action.action_queue.append(self)
        
        
    """
    Add an action at the end of the action queue
    """
    def add(action, signal = FINISHED, non_blocking = False):
        Action.action_queue.append(action)
        Action.blocking_flags_queue.append(not non_blocking)
        Action.signal_queue.append(signal)
        if len(Action.action_queue) == 1:
            Action.action_queue[0].start()
        
    """
    Add action at position 1 in the action queue.
    If actionqueue is empty, it will be added at position 0 instead
    """
    """
    def addSubsequent(action, *actions):
        if len(Action.action_queue) > 0:
            Action.action_queue.insert(1, action)
            for i, a in enumerate(actions):
                Action.action_queue.insert(2 + i, a)
        else:
            Action.add(action)
            for i, a in enumerate(actions):
                Action.action_queue.insert(1 + i, a)
    """
    def pause():
        Action.PAUSE = True
        
    def unpause():
        Action.PAUSE = False
        
    def wait_language():
        Action.WAIT = True
    
    """
    This function is called to initialize an action.
    Should be overwritten in the definition of an action, e.g. to initialize some parameters
    """
    def start(self):
        pass
        
    """
    Steps into an action if there is one in the action queue.
    Returns True after a successful step into an action.
    """
    def update(deltaTime):
        if Action.PAUSE:
            return Action.PAUSED
        if Action.WAIT:
            if Teacher.isTalking():
                return Action.BLOCKED
            Action.WAIT = False
            Action.action_queue.pop(0)
            sig = Action.signal_queue.pop(0)
            if len(Action.action_queue) > 0:
                Action.action_queue[0].start()
            return sig
        if len(Action.action_queue) > 0:
            s = Action.action_queue[0].step()
            if s > 0: # action returned FINISHED or CANCELED
                if Action.blocking_flags_queue.pop(0):
                    Action.wait_language()
                    if s == Action.CANCELED:
                        Action.signal_queue[0] = s
                    return Action.BLOCKED
                Action.action_queue.pop(0)
                sig = Action.signal_queue.pop(0)
                if len(Action.action_queue) > 0:
                    Action.action_queue[0].start()
                if s == Action.FINISHED:
                    return sig
            return s
        else:
            return Action.STEPPED
    
    
    """
    Advances the action.
    Must be overwritten.
    Return True 
    Return False if the action is finished. The action will be removed from the action queue and step
    will not be called again until the action is added back to the action queue. Before adding finished
    actions back to the queue, make sure the start function reset all relevant variables
    """
    def step(self):
        return Action.FINISHED # OLD: Return False if no action was executed (because action is finished)
        
"""
Wait for time in seconds
"""
class Wait(Action):
    def __init__(self, time):
        self.time = time
        
    def start(self):
        self.timer = self.time
    
    def step(self):
        self.timer -= delta_time
        if self.timer <= 0:
            return Action.FINISHED
        return Action.STEPPED

"""

"""    
class Remove(Action):
    def __init__(self, sceneObject):
        self.obj = sceneObject
        
    def start(self):
        self.obj.delete()
        
    def step(self):
        return Action.FINISHED

"""
Place a new object on the table
"""
class Place(Action):
    def __init__(self, obj, pos, produce_language=False):
        self.obj = obj
        self.counter = 5
        self.pos = pos
        self.produce_language = produce_language
        
    def start(self):
        self.obj.activate()
        self.obj.shape.set_position(self.pos)
        #self.sceneobj.set_position(self.pos)
        if self.produce_language:
            Teacher.say("placed")
            Teacher.say(self.sceneobj.description())
            Teacher.countAction("placed" + " " + self.sceneobj.description().replace(",", " "))
        
    def step(self):
        return Action.FINISHED
"""
Move the arm to a position
"""       
class MoveTo(Action):
    def __init__(self, pos, velocity = 1):
        self.velocity = velocity
        self.t_pos = pos
        
    def start(self):
        self.path = LinPath(tip.get_position(), self.t_pos)
        
    def step(self):
        new_pos = self.path.next(delta_time, self.velocity)
        if new_pos != 0:
            target.set_position(new_pos)
            return Action.STEPPED
        return Action.FINISHED
    
"""
Navigate arm to a position relative to the specified object. If dynamic is True, the path will
update according to the movements of the object
"""
class MoveRelativeTo(Action):
    def __init__(self, sceneObject, relative_pos, velocity = 1, dynamic = True, dist = 0.2):
        self.object = sceneObject
        self.velocity = velocity
        self.relPos = relative_pos
        self.dynamic = dynamic
        self.dist = dist
        
    def start(self):
        self.dynFlag = self.dynamic
        if not self.dynFlag:
            self.path = LinPath(tip.get_position(), self.object.get_position() + self.relPos)
        
    def step(self):
        if self.dynFlag:
            if np.linalg.norm((tip.get_position() - self.object.get_position()), ord=2) < self.dist:
                self.dynFlag = False
            self.path = LinPath(tip.get_position(), self.object.get_position() + self.relPos)
        new_pos = self.path.next(delta_time, self.velocity)
        if new_pos != 0:
            target.set_position(new_pos)
            return Action.STEPPED
        return Action.FINISHED
"""
Define an action from a list of actions thst should be executed consecutively

Params:
    actionlist - list of Actions
"""
class CombinedAction(Action):
    def __init__(self, actionlist):
        self.actions = actionlist
        
    def start(self):
        if len(self.actions) > 0:
            self.actions[0].start()
            
    def append(self, action):
        self.actions.append(action)
            
    def step(self):
        if len(self.actions) > 1:
            ret = self.actions[0].step()
            if ret > 0:
                self.actions.pop(0)
                self.actions[0].start()
                return self.step()
        elif len(self.actions) == 1:
            return self.actions[0].step()
        return Action.STEPPED
                
        

class Push(Action): 
    def __init__(self, sceneObject, direction = "random", velocity = 1.5):
        direction = direction.lower()
        #super().__init__()
        self.object = sceneObject
        self.velocity = velocity
        self.collision = False #We check for collision to see if object was pushed
        self.language_output = "pushed"
        if direction == "random":
            direction = random.choice(["left", "right"])
        self.language_output += " " + direction
        #self.language_output += "," + sceneObject.color + "," + sceneObject.get_name().lower()
        self.direction = direction
        self.waypoints = []
        self.cur_path = None
        
    def start(self):
        dx = random.uniform(-0.3, 0.3)
        dy = random.uniform(.45, .75)
        dz = random.uniform(.6, .8)
        
        self.dz = dz
        
        if self.direction == "right":
            o_pos = self.object.get_position()
            pos = o_pos.copy()
            pos[0] += dx
            pos[1] -= dy
            pos[2] += dz
            self.waypoints.append(pos.copy())
            pos[2] -= dz
            self.waypoints.append(pos.copy())
            pos = [pos[0] + (o_pos[0] - pos[0]) * 2, pos[1] + (o_pos[1] - pos[1]) * 2, pos[2]]
            #pos[1] += .35
            self.waypoints.append(pos.copy())
            pos[2] += 1
            self.waypoints.append(pos.copy())
            
        elif self.direction == "left":
            o_pos = self.object.get_position()
            pos = o_pos.copy()
            pos[0] += dx
            pos[1] += dy
            pos[2] += dz
            self.waypoints.append(pos.copy())
            pos[2] -= dz
            self.waypoints.append(pos.copy())
            pos = [pos[0] + (o_pos[0] - pos[0]) * 2, pos[1] + (o_pos[1] - pos[1]) * 2, pos[2]]
            #pos[1] -= .35
            self.waypoints.append(pos.copy())
            pos[2] += 1
            self.waypoints.append(pos.copy())
            
        self.initial_velocity_y = int(self.object.shape.get_velocity()[0][1] * 10)
        self.initial_position_y = self.object.shape.get_position()[1]
        self.pushed = False
        self.error = False
                    
    def step(self):
        global delta_time
        if not self.pushed and not self.object.active:
            self.error = True
            #return Action.CANCELED
        if self.cur_path is None:
            if len(self.waypoints) > 0:
                if len(self.waypoints) == 1:
                    if self.pushed:
                        if self.direction == "right":
                            if self.object.shape.get_position()[1] < self.initial_position_y:
                                self.error = True
                        elif self.direction == "left":
                            if self.object.shape.get_position()[1] > self.initial_position_y:
                                self.error = True
                        
                        if not self.error:
                            Teacher.say(self.language_output)
                            Teacher.say(self.object.description())
                            Teacher.countAction(self.language_output + " " + self.object.description().replace(","," "))
                    else: # If object hasn't moved, change its position
                        self.error = True
                        #pos = target.get_position().copy()
                        #pos[2] += self.dz
                        #self.waypoints = [pos]
                        #self.object.set_position([random.uniform(*table_bounds[0]), random.uniform(*table_bounds[1]), 1.05])
                self.cur_path = LinPath(tip.get_position(), self.waypoints.pop(0))
                return self.step()
            else:
                if self.error:
                    return Action.CANCELED
                return Action.FINISHED
        #print("Dist: ", euclidean_distance(tip.get_position(), target.get_position()))
        new_pos = self.cur_path.next(delta_time, self.velocity)
        if new_pos != 0:
            target.set_position(new_pos)
            if self.direction == "right":
                if (int(self.object.shape.get_velocity()[0][1] * 10) > self.initial_velocity_y):
                    self.pushed = True
            elif self.direction == "left":
                if (int(self.object.shape.get_velocity()[0][1] * 10) < self.initial_velocity_y):
                    self.pushed = True
            #if self.object.get_velocity != self.initial_velocity:
             #   self.pushed = True
            #self.collision = self.collision or self.object.check_collision(collision_boxes[0]) or self.object.check_collision(collision_boxes[1])
        else:
            self.cur_path = None
            return self.step()
        return Action.STEPPED

class Pick(Action): 
    def __init__(self, sceneObject, velocity = 1.0):
        self.object = sceneObject
        self.velocity = velocity
        self.collision = False #We check for collision to see if object was pushed
        self.language_output = "picked up"
        self.waypoints = [0, 1, 2, 3]
        self.cur_path = None
        
    def start(self):
        self.grabbed = False        
        pos = self.object.get_position()
        pos[2] = max(3.2, pos[2])
        self.waypoints[0] = pos.copy()
        pos = self.object.get_position()
        pos[2] = 2.3
        self.waypoints[1] = pos.copy()
        self.cur_path = LinPath(tip.get_position(), self.waypoints[0])
        self.waypoint_idx = 0

        self.object.shape.set_dynamic(False)
           
                    
    def step(self):        
        if not self.grabbed:
            if not self.object.active:
                return Action.CANCELED #Object left playground
            #pos[2] += .03
            #self.cur_path = LinPath(tip.get_position(), pos)
            
        if self.waypoint_idx == 1:
            if self.object.shape.check_distance(tip) < 0.05:
                self.grabbed = True
                self.object.shape.set_parent(tip)
                pos = tip.get_position()
                pos[2] += 0.3
                self.waypoints[2] = pos.copy()
                pos[2] += 0.5
                self.waypoints[3] = pos.copy()
                self.waypoint_idx = 2
                self.cur_path = LinPath(tip.get_position(), self.waypoints[self.waypoint_idx])
                                        
        new_pos = self.cur_path.next(delta_time, self.velocity)
        if new_pos != 0:
            target.set_position(new_pos)
        else:
            if self.waypoint_idx == 1:
                if not self.grabbed:
                    self.object.shape.set_dynamic(True)
                    return Action.CANCELED
            elif self.waypoint_idx == 2:
                Teacher.say(self.language_output)
                Teacher.say(self.object.description())
                Teacher.countAction(self.language_output + " " + self.object.description().replace(","," "))
            elif self.waypoint_idx == 3:
                #Action.addSubsequent(Drop(self.object, [random.uniform(*table_bounds[0]), random.uniform(*table_bounds[1]), 1.3]))
                return Action.FINISHED
            self.waypoint_idx += 1
            self.cur_path = LinPath(tip.get_position(), self.waypoints[self.waypoint_idx])
            return self.step()

        return Action.STEPPED
    
class PutDown(Action):
    def __init__(self, sceneObject, pos, velocity = 1):
        self.object = sceneObject
        self.pos = pos
        self.velocity = velocity
        self.language = "put down"
        
    def start(self):
        self.cur_path = LinPath(tip.get_position(), self.pos)
        self.waypoint_idx = 0

    def step(self):
        new_pos = self.cur_path.next(delta_time, self.velocity)
        if self.waypoint_idx == 1 and self.object.shape.check_collision(Shape.get_object("diningTable")):
            Shape
            new_pos = 0
        if new_pos != 0:
            target.set_position(new_pos)
        else:
            if self.waypoint_idx == 0:
                self.cur_path = LinPath(tip.get_position(), [self.pos[0], self.pos[1], 2.3])
                self.waypoint_idx = 1
            elif self.waypoint_idx == 1:
                self.object.shape.set_dynamic(True)
                self.object.shape.set_parent(None)
                Teacher.say(self.language)
                Teacher.say(self.object.description())
                Teacher.countAction(self.language + " " + self.object.description().replace(","," "))
                self.cur_path = LinPath(tip.get_position(), [tip.get_position()[0], tip.get_position()[1], tip.get_position()[2] + .35])
                self.waypoint_idx = 2
            elif self.waypoint_idx == 2:
                return Action.FINISHED
        return Action.STEPPED
    
class Drop(Action):
    def __init__(self, sceneObject, pos, velocity = 1):
        self.object = sceneObject
        self.pos = pos
        self.velocity = velocity
        self.counter = .25
        self.language = "dropped"
        
    def start(self):
        self.cur_path = LinPath(tip.get_position(), self.pos)
        self.c = self.counter
        
    def step(self):
        new_pos = self.cur_path.next(delta_time, self.velocity)
        if new_pos != 0:
            target.set_position(new_pos)
        else:
            self.object.shape.set_dynamic(True)
            self.object.shape.set_parent(None)
            self.c -= delta_time
            self.c <= 0
            Teacher.say(self.language)
            Teacher.say(self.object.description())
            Teacher.countAction(self.language + " " + self.object.description().replace(",", " "))
            return Action.FINISHED
        return Action.STEPPED
        

"""
Linear Path
"""
class LinPath:
    
    def __init__(self, start_pos, end_pos):
        self.start = start_pos
        self.end = end_pos
        self.dist = euclidean_distance(self.start, self.end)
        self.counter = 0
            
    
    def next(self, delta_time, velocity = 1):
        self.counter += delta_time * velocity
        if self.counter > self.dist:
            return 0
        return linear_interpolation(self.start, self.end, self.counter/self.dist)
                
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def linear_interpolation(a, b, s):
    vec = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]  # vector from a to b
    for i in range(len(vec)):
        vec[i] = vec[i] * s + a[i]
        
    return vec # returns position
