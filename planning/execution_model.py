import random
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from impedance_controller import *

def parse_xml_to_dict(xml_file: str) -> Dict:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    model_dict = {}
    for obj_elem in root.findall("object"):
        obj_id = obj_elem.get("id")
        
        task_elem = obj_elem.find("task")
        task_id = task_elem.find("id").text
        task_skill = task_elem.find("skill").text

        dependencies = []
        for dep_elem in obj_elem.find("dependencies").findall("dependency"):
            dep_id = dep_elem.find("id").text
            dep_skill = dep_elem.find("skill").text
            dep_conf = float(dep_elem.find("confidence").text)
            dependencies.append(((dep_id, dep_skill), dep_conf))

        model_dict[obj_id] = {
            "task": (task_id, task_skill),
            "termination": obj_elem.find("termination").text,
            "dependencies": dependencies,
            "object_confidence": float(obj_elem.find("object_confidence").text)
        }
    
    return model_dict

class DisassemblySystem:
    def __init__(self, model_dict: Dict = None):
        self.usable_parts: List[str] = []          
        self.dependency_tasks: Dict[str, List[str]] = {} 
        self.executed_tasks: Set[str] = set()      
        self.valid_objects: List[str] = []        
        self.skill_library: Dict[str, str] = {     
            # S0 skills
            "S_(0x+)": "Direct pull along +X axis",
            "S_(0x-)": "Direct pull along -X axis",
            "S_(0y+)": "Direct pull along +Y axis",
            "S_(0y-)": "Direct pull along -Y axis",
            "S_(0z+)": "Direct pull along +Z axis",
            "S_(0z-)": "Direct pull along -Z axis",
            
            # S1 skills
            "S_(1x+)": "Twist-pull along +X axis",
            "S_(1x-)": "Twist-pull along -X axis",
            "S_(1y+)": "Twist-pull along +Y axis",
            "S_(1y-)": "Twist-pull along -Y axis",
            "S_(1z+)": "Twist-pull along +Z axis",
            "S_(1z-)": "Twist-pull along -Z axis"
        }
        
        self.disassembled_parts: List[str] = []    
        self.model_dict = model_dict or {}
        self.impedance_controller = impedance_controller         

    def load_model_from_dict(self, model_dict: Dict):
        self.model_dict = model_dict
        self.usable_parts = list(model_dict.keys())

        for obj_id, data in model_dict.items():
            task_id = data["task"][0]
            self.dependency_tasks[obj_id] = [
                dep[0][0] for dep in data["dependencies"]]

    def is_part_valid(self, part: str) -> bool:
        return all(
            dep in self.executed_tasks 
            for dep in self.dependency_tasks.get(part, [])
        )

    def filter_and_sort_objects(self):
        self.valid_objects = [
            part for part in self.usable_parts 
            if self._is_part_valid(part)
        ]
        self.valid_objects.sort(key=lambda x: len(self.dependency_tasks[x]))

    def is_skill_successful(self,
        part_id: str,
        initial_pos: np.ndarray,  
        current_pos: np.ndarray,   
        collection_area_bounds: Dict[str, Tuple[float, float]],  
        fixed_part_id: str,       
        remaining_parts: List[str]  
    ) -> bool:
        
        in_collection_area = all(
            collection_area_bounds[axis][0] <= current_pos[i] <= collection_area_bounds[axis][1]
            for i, axis in enumerate(["x", "y", "z"])
        )
        
        displacement = np.linalg.norm(current_pos - initial_pos)
        MIN_DISPLACEMENT = 0.1 
   
        is_last_removable_part = (len(remaining_parts) == 2 and fixed_part_id in remaining_parts)
        
        return (
            in_collection_area 
            and displacement >= MIN_DISPLACEMENT
            and (not is_last_removable_part or part_id != fixed_part_id)
        )

    def physical_exploration(self):
        time.sleep(1) 

        if random.random() < 0.2 and len(self.usable_parts) < 5:
            new_part = f"O_{len(self.usable_parts)+1}"
            self.usable_parts.append(new_part)
            self.dependency_tasks[new_part] = {
                "skill": random.choice(list(self.skill_library.keys())),
                "dependencies": [],
                "confidence": round(random.uniform(0.5, 0.9), 4)
            }
            return True

        remaining_deps = set()
        for part, data in self.dependency_tasks.items():
            for dep_id, _ in data["dependencies"]:
                if dep_id not in self.executed_tasks:
                    remaining_deps.add(dep_id)
        
        if remaining_deps and random.random() < 0.3:
            resolved_dep = random.choice(list(remaining_deps))
            self.executed_tasks.add(resolved_dep)
            return True

        return False

    def attempt_disassembly(self, part: str) -> bool:
        part_data = self.model_dict[part]
        task_id, skill_code = part_data["task"]
        skill = self.get_human_readable_skill(skill_code)

        success_rate = part_data["object_confidence"] * 0.8 + 0.2
        success = random.random() < success_rate
        
        if success:
            self.executed_tasks.add(part)
            self.disassembled_parts.append(part)
            print(" success！")
        else:
            print(" fail！")
        return success

    def fixed_loop_execution(self,
        part_id: str,
        target_position: Tuple[float, float, float],  
        place_area: height
    ) -> bool:
        
        # 1. RESET
        self.impedance_controller.initalrobot()  
        
        # 2. POSITIONING and grasping
        self.impedance_controller.robot_control_grasptarget(target_position)
        
        # 3. disassembling
        self.attempt_disassembly(part_id)
       
        # 4. placing
        self.impedance_controller.robot_control_place(place_area)
