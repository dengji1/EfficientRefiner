import os
import re

# import dreamplace.xxx -> import legalization.dreamplace.dreamplace.xxx
# from dreamplace.xxx -> from legalization.dreamplace.dreamplace.xxx
def replace_imports_in_py_files(root_dir):
    pattern = re.compile(r'^(.*\b(?:import|from)\s+)dreamplace\.', re.IGNORECASE)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)

                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                modified = False
                new_lines = []
                for line in lines:
                    if 'import dreamplace.' in line or 'from dreamplace.' in line:
                        new_line = pattern.sub(r'\1legalization.dreamplace.dreamplace.', line)
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

if __name__ == '__main__':
    replace_imports_in_py_files('dreamplace')
