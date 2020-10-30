import os
if __name__ == '__main__':
    with open('../shells/missing_VBlockGCN_ATT_search_cora.sh', 'r') as f:
        cmds = f.readlines()
        for cmd in cmds:
            os.system (cmd)