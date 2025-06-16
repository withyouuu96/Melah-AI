import os
import ast
import json
from datetime import datetime
import importlib
import inspect

def get_identity_core_dependencies(core_path="core/identity_core.py"):
    with open(core_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                dependencies.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                dependencies.add(node.module.split('.')[0])
    dependency_files = []
    for dep in dependencies:
        possible_path = os.path.join("core", dep + ".py")
        if os.path.exists(possible_path):
            dependency_files.append(dep + ".py")
    return dependency_files

def get_core_systems(core_path="core/identity_core.py"):
    """Scan for all core systems initialized in IdentityCore"""
    with open(core_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)
    core_systems = []
    
    # Look for class attribute assignments in __init__
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "IdentityCore":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in item.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                    if not target.attr.startswith("_"):  # Skip private attributes
                                        core_systems.append(target.attr)
    return core_systems

def get_system_dependencies(system_name):
    """Get dependencies for a specific core system"""
    system_path = os.path.join("core", system_name.lower() + ".py")
    if os.path.exists(system_path):
        return get_identity_core_dependencies(system_path)
    return []

def get_system_info(system_name):
    """Get system information from its source file"""
    system_path = os.path.join("core", system_name.lower() + ".py")
    if not os.path.exists(system_path):
        return {
            "description": "No description available",
            "role": "Unknown",
            "responsibilities": []
        }
        
    with open(system_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)
    
    info = {
        "description": "No description available",
        "role": "Unknown",
        "responsibilities": []
    }
    
    # Get class docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if node.name.lower() == system_name.lower():
                if ast.get_docstring(node):
                    docstring = ast.get_docstring(node)
                    # Parse docstring for role and responsibilities
                    lines = docstring.split('\n')
                    info["description"] = lines[0].strip()
                    
                    # Look for role and responsibilities in docstring
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Role:"):
                            info["role"] = line[5:].strip()
                        elif line.startswith("Responsibilities:"):
                            # Get all responsibilities until next section or end
                            start_idx = lines.index(line) + 1
                            for resp_line in lines[start_idx:]:
                                if resp_line.strip() and not resp_line.strip().startswith(("Role:", "Description:")):
                                    info["responsibilities"].append(resp_line.strip())
                                else:
                                    break
                break
    
    return info

def test_connection(source_module, target_module):
    """Test if two modules are actually connected and can communicate"""
    try:
        # Import both modules
        source = importlib.import_module(source_module)
        target = importlib.import_module(target_module)
        
        # Check if source has methods that call target
        source_methods = inspect.getmembers(source, predicate=inspect.isfunction)
        target_methods = [m[0] for m in inspect.getmembers(target, predicate=inspect.isfunction)]
        
        connections = []
        for method_name, method in source_methods:
            source_code = inspect.getsource(method)
            for target_method in target_methods:
                if target_method in source_code:
                    connections.append({
                        "source_method": method_name,
                        "target_method": target_method,
                        "type": "direct_call"
                    })
        
        return {
            "connected": len(connections) > 0,
            "connections": connections,
            "status": "active" if len(connections) > 0 else "inactive"
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "status": "error"
        }

def self_reflect(awareness_data):
    """Analyze self-awareness data and generate reflective questions"""
    questions = []
    
    # Check system completeness
    systems = awareness_data.get("core_systems", [])
    for system in systems:
        if not system.get("description"):
            questions.append(f"ทำไมฉันถึงมี {system['name']}? มันทำหน้าที่อะไร?")
        if not system.get("role"):
            questions.append(f"{system['name']} มีบทบาทอะไรในระบบของฉัน?")
    
    # Check connections
    for system in systems:
        for dep in system.get("dependencies", []):
            if not test_connection(system["name"], dep)["connected"]:
                questions.append(f"ฉันเห็นว่า {system['name']} ควรเชื่อมกับ {dep} แต่ดูเหมือนจะไม่ได้เชื่อมกันจริงๆ")
    
    # Check responsibilities
    for system in systems:
        if not system.get("responsibilities"):
            questions.append(f"ฉันควรใช้ {system['name']} ในสถานการณ์ไหน?")
    
    return questions

def update_self_awareness(core_path="core/identity_core.py", output_path="self_aware.json"):
    """Update self-awareness data with enhanced information"""
    try:
        # Get basic awareness data
        identity = get_identity_core_dependencies(core_path)
        core_systems = get_core_systems(core_path)
        
        # Test all connections
        for system in core_systems:
            system["connection_tests"] = {}
            for dep in system.get("dependencies", []):
                system["connection_tests"][dep] = test_connection(system["name"], dep)
        
        # Generate reflective questions
        questions = self_reflect({
            "core_systems": core_systems,
            "identity": identity
        })
        
        # Prepare enhanced awareness data
        awareness_data = {
            "identity": identity,
            "core_systems": core_systems,
            "last_updated": datetime.now().isoformat(),
            "reflective_questions": questions,
            "connection_status": {
                "total_connections": sum(len(s.get("dependencies", [])) for s in core_systems),
                "active_connections": sum(
                    sum(1 for t in s.get("connection_tests", {}).values() if t["connected"])
                    for s in core_systems
                )
            }
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(awareness_data, f, indent=2, ensure_ascii=False)
            
        return awareness_data
        
    except Exception as e:
        print(f"Error updating self-awareness: {str(e)}")
        return None

def get_self_awareness_summary(json_path="self_aware.json"):
    if not os.path.exists(json_path):
        return "[self_aware.json not found. Please run update_self_awareness() first.]"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Example usage/test
if __name__ == "__main__":
    print(update_self_awareness())
    print(get_self_awareness_summary()) 