import xml.etree.ElementTree as ET
from typing import Dict, Any
import time

def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def save_model_to_xml(model_dict: Dict[str, Any], filename: str = "product.xml") -> None:
    root = ET.Element("product")
    
    for obj_id, data in model_dict.items():
        obj_elem = ET.SubElement(root, "object", id=obj_id)

        task_elem = ET.SubElement(obj_elem, "task")
        ET.SubElement(task_elem, "id").text = data["task"][0]
        ET.SubElement(task_elem, "skill").text = data["task"][1]

        ET.SubElement(obj_elem, "termination").text = data["termination"]

        deps_elem = ET.SubElement(obj_elem, "dependencies")
        for dep, conf in data["dependencies"]:
            dep_elem = ET.SubElement(deps_elem, "dependency")
            ET.SubElement(dep_elem, "id").text = dep[0]
            ET.SubElement(dep_elem, "skill").text = dep[1]
            ET.SubElement(dep_elem, "confidence").text = str(conf)

        ET.SubElement(obj_elem, "object_confidence").text = str(data["object_confidence"])

    indent(root)
    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"XML saved to {filename} with formatting.")

def update_model_loop(initial_model: Dict[str, Any], 
                     update_interval: int = 5,
                     max_iterations: int = 3,
                     output_file: str = "product.xml") -> None:
    current_model = initial_model.copy()
    
    for iteration in range(1, max_iterations + 1):
        for obj_id in current_model:
            current_model[obj_id]["object_confidence"] = min(
                1.0, 
                current_model[obj_id]["object_confidence"] * (1 + 0.05 * iteration)
            )
            updated_deps = []
            for dep, conf in current_model[obj_id]["dependencies"]:
                updated_deps.append((dep, min(1.0, conf * (1 + 0.03 * iteration))))
            current_model[obj_id]["dependencies"] = updated_deps

        print(f"current_model ({iteration}):")
        for obj_id, data in current_model.items():
            print(f"object {obj_id}: object_confidence={data['object_confidence']:.4f}")
        timestamp = int(time.time())
        versioned_file = f"{output_file.split('.')[0]}_v{iteration}_{timestamp}.xml"
        save_model_to_xml(current_model, versioned_file)

        if iteration < max_iterations:
            time.sleep(update_interval)

    save_model_to_xml(current_model, output_file)