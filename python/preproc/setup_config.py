from os import path as op

def insert_subject_blocks(template_path, output_path, subjectID, blocks):
    # Read the template .m file
    with open(template_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    inserted_subject = False
    inserted_blocks = False

    for line in lines:
        new_lines.append(line)

        # After finding the right place, insert the new subject ID
        if not inserted_subject and 'INFO.file_labels.Subjects' in line:
            new_subject_line = f'INFO.file_labels.Subjects = {{"{subjectID}"}};\n'
            new_lines[-1] = '%' + line  # Comment out the original line
            new_lines.append(new_subject_line)
            inserted_subject = True

        # After finding the right place, insert the new blocks
        if not inserted_blocks and 'tmp.Block{1}' in line:
            formatted_blocks = ','.join(f'"{b}"' for b in blocks)
            new_block_line = f'tmp.Block{{1}} = {{{formatted_blocks}}};\n'
            new_lines[-1] = '%' + line  # Comment out the original line
            new_lines.append(new_block_line)
            inserted_blocks = True

    # Save the new version
    with open(output_path, 'w') as file:
        file.writelines(new_lines)

    print(f"Saved new config to {output_path}")