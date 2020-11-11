import os
if __name__ == '__main__':
    with open('../shells/VBlockGCN_drop_important_cora.sh', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)