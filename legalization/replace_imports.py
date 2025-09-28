import os
import re

# import dreamplace.xxx -> import legalization.dreamplace.dreamplace.xxx as xxx
# from dreamplace.xxx -> from legalization.dreamplace.dreamplace.xxx
# import xxx-> import legalization.dreamplace.dreamplace.xxx as xxx (xxx in dreamplace)
# from xxx-> from legalization.dreamplace.dreamplace.xxx (xxx in dreamplace)

def replace_imports_in_py_files(root_dir):
    pattern = re.compile(r'^(.*\b(?:import|from)\s+)dreamplace', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                modified = False
                new_lines = []
                for line in lines:
                    if 'import dreamplace' in line or 'from dreamplace' in line:
                        new_line = pattern.sub(r'\1legalization.dreamplace.dreamplace', line)
                        if new_line != line:
                            modified = True
                            new_lines.append(new_line)
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                if modified:
                    print(f'Updating: {filepath}')
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                        

    py_files = [os.path.splitext(f)[0] for f in os.listdir(root_dir) if f.endswith(".py")]
    prefix = "legalization.dreamplace.dreamplace."
    for filename in os.listdir(root_dir):
        if not filename.endswith(".py"):
            continue

        filepath = os.path.join(root_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        for mod in py_files:
            code = re.sub(
                rf"(^|\n)\s*import\s+{mod}(\s|$)",
                rf"\1import {prefix}{mod}\2",
                code
            )
            code = re.sub(
                rf"(^|\n)\s*from\s+{mod}(\s|\.|$)",
                rf"\1from {prefix}{mod}\2",
                code
            )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        
        
    pattern = re.compile(r'^\s*import\s+legalization\.dreamplace\.dreamplace\.([a-zA-Z_][a-zA-Z0-9_]*)\s*$')
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue

            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            modified = False
            new_lines = []
            for line in lines:
                if re.search(r'\s+as\s+', line):
                    new_lines.append(line)
                    continue

                m = pattern.match(line.strip())
                if m:
                    module = m.group(1)
                    new_line = f"import legalization.dreamplace.dreamplace.{module} as {module}\n"
                    new_lines.append(new_line)
                    modified = True
                else:
                    new_lines.append(line)

            if modified:
                print(f"Updating: {filepath}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

if __name__ == '__main__':
    root_dir = 'dreamplace/dreamplace'
    replace_imports_in_py_files(root_dir)
