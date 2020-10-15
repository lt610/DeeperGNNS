import os
if __name__ == '__main__':
    with open('../shells/MLP_result_cora_citeseer_pubmed_chameleon_cornell_texas_wisconsin.txt', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)